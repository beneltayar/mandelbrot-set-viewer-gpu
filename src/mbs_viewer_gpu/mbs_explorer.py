import contextlib
import dataclasses
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from typing import Optional

import cv2
import numpy as np
from PyQt6 import QtGui
from PyQt6.QtCore import Qt, QPoint, QRectF, QPointF, QTimer
from PyQt6.QtGui import QPixmap, QImage, QPainter, QFont
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QSlider

from .big_float.classes import BigFloat, WindowDimension
from .config import MIN_NUM_ITERATIONS, MIN_IMAGE_HEIGHT, GRID_HEIGHT, GRID_WIDTH, MAX_IMAGE_HEIGHT, RESOLUTION_UPSCALE_FACTOR, MAX_NUM_ITERATIONS, STARTING_WINDOW_HEIGHT
from .mbs_image_calculator import get_mandel_image
from .utils import colorize_mandel_image


@dataclasses.dataclass(frozen=True)
class ViewConfig:
    dimension: WindowDimension
    num_iterations: int


class ViewConfigChange(Exception):
    pass


class MandelExplorer(QWidget):
    def __init__(self):
        super().__init__()
        self.dimension_lock = threading.Lock()
        self.setWindowTitle('Mandelbrot Set Explorer GPU')
        self.setGeometry(50, 50, round(STARTING_WINDOW_HEIGHT * 1.5), STARTING_WINDOW_HEIGHT)

        self.calculated_height: int = 0
        self.dimensions: list[WindowDimension] = [WindowDimension(
            x_min=BigFloat.from_num(-2),
            x_max=BigFloat.from_num(+1),
            y_min=BigFloat.from_num(-1),
            y_max=BigFloat.from_num(+1),
        )]
        self.image_label = QLabel(self)
        self.image_label.setScaledContents(True)
        self.overlay_drawer = OverlayDrawer(self)
        self.drag_start: Optional[QPoint] = None
        self.mandel_image = np.zeros((self.height(), self.width(), 3), dtype=np.uint8)
        self.num_iterations_slider = QSlider(self)
        self.num_iterations_slider.setRange(MIN_NUM_ITERATIONS, MAX_NUM_ITERATIONS)
        # noinspection PyUnresolvedReferences
        self.num_iterations_slider.valueChanged.connect(self._update_iterations_label)
        self.scale_label = QLabel(self)
        self.scale_label.setFont(QFont('Ariel', pointSize=20))
        self.scale_label.setStyleSheet('color: white')
        self.quality = 0.
        self.time_stated_calculating = time.monotonic()
        self.calculation_timer = QTimer(self)
        self.calculation_timer.setInterval(100)
        # noinspection PyUnresolvedReferences
        self.calculation_timer.timeout.connect(self.update_timer)
        self.calculation_timer.start()
        self.quality_label = QLabel(self)
        self.quality_label.setFont(QFont('Ariel', pointSize=20))
        self.quality_label.setStyleSheet('color: white')
        self.timer_label = QLabel(self)
        self.timer_label.setFont(QFont('Ariel', pointSize=20))
        self.timer_label.setStyleSheet('color: white')
        self.iterations_label = QLabel(self)
        self.iterations_label.setFont(QFont('Ariel', pointSize=20))
        self.iterations_label.setStyleSheet('color: white')
        self._update_iterations_label()
        self.updating_thread = Thread(target=self.calc_set_continuously, daemon=True)
        self.updating_thread.start()

    def _update_iterations_label(self):
        self.iterations_label.setText(f'Iterations: {self.num_iterations_slider.value()}')

    def update_timer(self):
        if self.quality < 1.:
            self.timer_label.setText(f'Render Time: {time.monotonic() - self.time_stated_calculating:.2f}s')

    def resizeEvent(self, a0: QtGui.QResizeEvent) -> None:
        self.num_iterations_slider.setGeometry(30, 30, 30, self.height() - 60)
        self.scale_label.setGeometry(100, 30, 3000, 100)
        self.quality_label.setGeometry(100, 60, 3000, 100)
        self.timer_label.setGeometry(100, 90, 3000, 100)
        self.iterations_label.setGeometry(100, 120, 3000, 100)

        with self._on_new_dimension(self.dimension) as new_dimension:
            self.dimensions[-1] = new_dimension
            self.mandel_image = np.zeros((a0.size().height(), a0.size().width(), 3), dtype=np.uint8)
        self.image_label.resize(a0.size())
        self._update_pixmap()

    def crop_and_zoom_to_new_dimension(self, dimension: WindowDimension):
        image_height, image_width, _ = self.mandel_image.shape
        width = self.dimension.width()
        left = (dimension.x_min - self.dimension.x_min) / width
        right = (dimension.x_max - self.dimension.x_min) / width
        top = (dimension.y_min - self.dimension.y_min) / width
        bottom = (dimension.y_max - self.dimension.y_min) / width

        im_left = round(left * image_width)
        im_right = round(right * image_width)
        im_top = round(top * image_width)
        im_bottom = round(bottom * image_width)

        if im_right < 0 or im_bottom < 0 or image_width < im_left or image_height < im_top:
            # we are at a completely different area with co common pixels
            self.mandel_image[:] = 0
            self._update_pixmap()
            return

        actual_left = max(0, im_left)
        actual_top = max(0, im_top)
        actual_right = min(image_width, im_right)
        actual_bottom = min(image_height, im_bottom)

        im_slice = self.mandel_image[actual_top:actual_bottom, actual_left:actual_right]
        slice_height, slice_width, _ = im_slice.shape

        if actual_left == im_left and actual_top == im_top and actual_right == im_right and actual_bottom == im_bottom:
            # noinspection PyUnresolvedReferences
            cv2.resize(im_slice.copy(), (image_width, image_height), dst=self.mandel_image, interpolation=cv2.INTER_CUBIC)
        else:
            pad_left = round(image_width * (actual_left - im_left) / (im_right - im_left))
            pad_right = round(image_width * (im_right - actual_right) / (im_right - im_left))
            pad_top = round(image_height * (actual_top - im_top) / (im_bottom - im_top))
            pad_bottom = round(image_height * (im_bottom - actual_bottom) / (im_bottom - im_top))
            dst = self.mandel_image[pad_top:image_height - pad_bottom, pad_left:image_width - pad_right]
            if dst.size:
                # noinspection PyUnresolvedReferences
                cv2.resize(im_slice.copy(), (image_width - pad_left - pad_right, image_height - pad_top - pad_bottom), dst=dst, interpolation=cv2.INTER_AREA)
            self.mandel_image[:pad_top] = 0
            self.mandel_image[pad_top:, :pad_left] = 0
            self.mandel_image[pad_top:, -pad_right:] = 0
            self.mandel_image[-pad_bottom:, pad_left:-pad_right] = 0
        self._update_pixmap()

    @property
    def dimension(self) -> WindowDimension:
        return self.dimensions[-1]

    @property
    def view_config(self) -> ViewConfig:
        return ViewConfig(self.dimension, self.num_iterations_slider.value())

    def calc_set_continuously(self):
        while True:
            self.quality = 0.
            self.time_stated_calculating = time.monotonic()
            self.quality_label.setText('Quality: 0%')
            self.timer_label.setText(f'Render Time: 0.00s')
            view_config = self.view_config
            try:
                self.calc_set_in_increasing_resolution(view_config=view_config)
            except ViewConfigChange:
                continue
            while view_config == self.view_config:
                time.sleep(0.1)

    def calc_set_in_increasing_resolution(self, view_config: ViewConfig):
        image_height = MIN_IMAGE_HEIGHT
        while image_height <= MAX_IMAGE_HEIGHT:
            self.calc_set_cells(view_config, image_height)
            image_height *= RESOLUTION_UPSCALE_FACTOR

    def calc_set_cells(self, view_config: ViewConfig, image_height: int):
        cells_coords = [(col, row) for row in range(GRID_HEIGHT) for col in range(GRID_WIDTH)]
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(self.calc_set_with_dimension, view_config=view_config, image_height=image_height, image_coord=cell_coords) for cell_coords in sorted(cells_coords, key=lambda coord: (coord[0] - GRID_WIDTH / 2) ** 2 + (coord[1] - GRID_HEIGHT / 2) ** 2)]
            for future in futures:
                try:
                    future.result()
                    self.quality += (((image_height / MAX_IMAGE_HEIGHT) ** 2) - ((((image_height / RESOLUTION_UPSCALE_FACTOR) / MAX_IMAGE_HEIGHT) ** 2) if image_height != MIN_IMAGE_HEIGHT else 0)) / (GRID_WIDTH * GRID_HEIGHT)
                    self.quality_label.setText(f'Quality: {round(self.quality * 100)}%')
                    self.timer_label.setText(f'Render Time: {time.monotonic() - self.time_stated_calculating:.2f}s')
                except ViewConfigChange:
                    executor.shutdown(cancel_futures=True)
                    raise

    def calc_set_with_dimension(self, view_config: ViewConfig, image_height: int, image_coord: tuple[int, int]):
        with self.dimension_lock:
            if view_config != self.view_config:
                raise ViewConfigChange

        grid_x, grid_y = image_coord
        image_dimension = view_config.dimension.rescaled_by_rect(QRectF(QPointF(grid_x / GRID_WIDTH, grid_y / GRID_HEIGHT),
                                                                        QPointF((grid_x + 1) / GRID_WIDTH, (grid_y + 1) / GRID_HEIGHT)))
        mandel_image = get_mandel_image(image_dimension, image_height=image_height, num_iterations=self.num_iterations_slider.value())
        colorized_image = colorize_mandel_image(mandel_image)
        height, width, _ = self.mandel_image.shape
        dest_x_min = int(width * grid_x / GRID_WIDTH)
        dest_x_max = int(width * (grid_x + 1) / GRID_WIDTH)
        dest_y_min = int(height * grid_y / GRID_HEIGHT)
        dest_y_max = int(height * (grid_y + 1) / GRID_HEIGHT)
        dest = self.mandel_image[dest_y_min:dest_y_max, dest_x_min:dest_x_max]
        # noinspection PyUnresolvedReferences
        interpolation = cv2.INTER_AREA if (dest_y_max - dest_y_min) < image_height else cv2.INTER_CUBIC
        with self.dimension_lock:
            if view_config != self.view_config:
                raise ViewConfigChange
            # noinspection PyUnresolvedReferences
            cv2.resize(colorized_image, (dest_x_max - dest_x_min, dest_y_max - dest_y_min), dst=dest, interpolation=interpolation)
        self._update_pixmap()

    def _update_pixmap(self):
        height, width, n_channels = self.mandel_image.shape
        bytes_per_line = n_channels * width
        qimage = QImage(self.mandel_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        self.image_label.setPixmap(pixmap)

    def mouseReleaseEvent(self, a0: QtGui.QMouseEvent) -> None:
        super().mouseReleaseEvent(a0)
        if a0.button() == Qt.MouseButton.LeftButton:
            self._set_viewpoint(self.get_normalized_rect(self.drag_start, a0.pos()))
            self.drag_start = None

    def mousePressEvent(self, a0: QtGui.QMouseEvent) -> None:
        if a0.button() == Qt.MouseButton.LeftButton:
            self.drag_start = a0.pos()
            a0.accept()
        elif a0.button() == Qt.MouseButton.RightButton:
            self._reset_dimensions()

    def _reset_dimensions(self):
        if len(self.dimensions) > 1:
            with self._on_new_dimension(self.dimensions[-2]) as new_dimension:
                self.dimensions[-2] = new_dimension
                self.dimensions.pop()

    @contextlib.contextmanager
    def _on_new_dimension(self, new_dimension: WindowDimension):
        with self.dimension_lock:
            new_dimension = new_dimension.re_ratio(self.width() / self.height())
            self.crop_and_zoom_to_new_dimension(new_dimension)
            yield new_dimension
            self.scale_label.setText(f'Scale: {new_dimension.width().to_num():.3e}')

    def wheelEvent(self, a0: QtGui.QWheelEvent) -> None:
        position = a0.position()
        position_x = position.x() / self.width()
        position_y = position.y() / self.height()

        delta = 1.003 ** -a0.angleDelta().y()
        new_dimension = self.dimension.rescaled_around_point(position_x, position_y, delta)
        with self._on_new_dimension(new_dimension) as new_dimension:
            self.dimensions[-1] = new_dimension

    def _set_viewpoint(self, rect: QRectF):
        if not rect.width() or not rect.height():
            return

        converted_rect = QRectF(
            QPointF(rect.left() / self.width(), (rect.top() / self.height())),
            QPointF(rect.right() / self.width(), (rect.bottom() / self.height())),
        )

        new_dimension = self.dimension.rescaled_by_rect(converted_rect)
        with self._on_new_dimension(new_dimension) as new_dimension:
            self.dimensions.append(new_dimension)

    def get_normalized_rect(self, p1: QPoint, p2: QPoint) -> QRectF:
        x1 = p1.x()
        y1 = p1.y()
        x2 = p2.x()
        y2 = p2.y()

        y2 = y1 + max(abs(y2 - y1), abs(x2 - x1) * self.height() / self.width()) * (np.sign(y2 - y1) or 1)
        x2 = x1 + max(abs(y2 - y1) * self.width() / self.height(), abs(x2 - x1)) * (np.sign(x2 - x1) or 1)

        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        return QRectF(QPointF(x1, y1), QPointF(x2, y2))

    def keyPressEvent(self, a0: QtGui.QKeyEvent) -> None:
        if a0.key() == Qt.Key.Key_Escape.value:
            self.close()
        if a0.key() == Qt.Key.Key_F11.value:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()


class OverlayDrawer(QWidget):
    def __init__(self, app: MandelExplorer):
        super().__init__(app)
        self.app = app
        self.setFixedWidth(self.app.width())
        self.setFixedHeight(self.app.height())
        self.rect: Optional[QRectF] = None

    def paintEvent(self, a0: QtGui.QPaintEvent) -> None:
        super().paintEvent(a0)
        painter = QPainter(self)
        if self.rect:
            painter.drawRect(self.rect)

    def mouseMoveEvent(self, a0: QtGui.QMouseEvent) -> None:
        if self.app.drag_start:
            self.rect = self.app.get_normalized_rect(self.app.drag_start, a0.pos())
            self.repaint()

    def mouseReleaseEvent(self, a0: QtGui.QMouseEvent) -> None:
        a0.accept()
        self.rect = None
        self.repaint()
        super().mouseReleaseEvent(a0)


def main():
    app = QApplication(sys.argv)
    explorer = MandelExplorer()
    explorer.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
