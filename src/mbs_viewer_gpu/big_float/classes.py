import dataclasses
from typing import Optional

import numpy as np
from PyQt6.QtCore import QRectF

from ..config import FACTOR_PER_CELL, DIVMOD_BUFFER, PRECISION


class BigFloat:
    def __init__(self, arr: Optional[np.ndarray] = None):
        self.arr = np.empty(PRECISION, dtype=np.int64) if arr is None else arr

    def __add__(self, other) -> 'BigFloat':
        if isinstance(other, BigFloat):
            result = BigFloat(self.arr + other.arr)
        elif isinstance(other, (float, int, np.ndarray)):
            result = BigFloat(self.arr + other)
        else:
            return NotImplemented
        result.normalize()
        return result

    def __sub__(self, other) -> 'BigFloat':
        if isinstance(other, BigFloat):
            result = BigFloat(self.arr - other.arr)
        elif isinstance(other, (float, int, np.ndarray)):
            result = self.arr - other
        else:
            return NotImplemented
        result.normalize()
        return result

    def __mul__(self, other) -> 'BigFloat':
        if isinstance(other, (int, float)):
            result = BigFloat()
            leftover = 0
            for i in range(self.arr.size - 1, -1, -1):
                leftover *= FACTOR_PER_CELL
                num = self.arr[i] * other + leftover
                result.arr[i], leftover = divmod(num, 1)
            result.normalize()
            return result
        if isinstance(other, BigFloat):
            result = BigFloat()
            leftover = 0
            for i in range(result.arr.size):
                leftover, result.arr[i] = divmod(self.arr[i] * other.arr[i] + leftover, FACTOR_PER_CELL)
            assert not leftover
            return result
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, BigFloat):
            for i in range(PRECISION - 1, -1, -1):
                if self.arr[i] > other.arr[i]:
                    return True
                elif self.arr[i] < other.arr[i]:
                    return False
            return False
        return NotImplemented

    def normalize(self):
        assert self.arr.dtype == np.int64
        leftover = 0
        for i, n in enumerate(self.arr):
            leftover, self.arr[i] = divmod(n + leftover + DIVMOD_BUFFER, FACTOR_PER_CELL)
            self.arr[i] -= DIVMOD_BUFFER
        assert not leftover

    def to_num(self) -> float:
        total = 0.0
        for element in self.arr:
            total /= FACTOR_PER_CELL
            total += element
        return total

    def get_scale_factor(self) -> int:
        for i, x in enumerate(reversed(self.arr)):
            if x:
                return i
        return self.arr.size

    def get_scaled_by_factor(self, factor: int) -> 'BigFloat':
        new = self.from_num(0)
        for i in range(PRECISION - factor):
            new.arr[i + factor] = self.arr[i]
        return new

    def __truediv__(self, other) -> float:
        if isinstance(other, (int, float)):
            return self.to_num() / other
        if isinstance(other, BigFloat):
            scale_factor = min(self.get_scale_factor(), other.get_scale_factor())
            return self.get_scaled_by_factor(scale_factor).to_num() / other.get_scaled_by_factor(scale_factor).to_num()
        return NotImplemented

    @classmethod
    def from_num(cls, num: float) -> 'BigFloat':
        result = cls()
        for i in range(PRECISION):
            result.arr[PRECISION - 1 - i] = int(num)
            num %= 1
            num *= FACTOR_PER_CELL
        return result

    def __repr__(self):
        return str(self.to_num())


@dataclasses.dataclass(frozen=True)
class WindowDimension:
    x_min: BigFloat
    x_max: BigFloat
    y_min: BigFloat
    y_max: BigFloat

    def ratio(self) -> float:
        width = self.width()
        height = self.height()
        scale_factor = min(width.get_scale_factor(), height.get_scale_factor())
        return width.get_scaled_by_factor(scale_factor).to_num() / height.get_scaled_by_factor(scale_factor).to_num()

    def rescaled_around_point(self, x: float, y: float, delta: float) -> 'WindowDimension':
        center_x = self.x_min + self.width() * x
        center_y = self.y_min + self.height() * y
        x_min = center_x - (center_x - self.x_min) * delta
        x_max = center_x - (center_x - self.x_max) * delta
        y_min = center_y - (center_y - self.y_min) * delta
        y_max = center_y - (center_y - self.y_max) * delta
        return WindowDimension(x_min, x_max, y_min, y_max)

    def rescaled_by_rect(self, rect: QRectF) -> 'WindowDimension':
        width = self.x_max - self.x_min
        height = self.y_max - self.y_min
        x_max = self.x_min + width * rect.right()
        x_min = self.x_min + width * rect.left()
        y_max = self.y_min + height * rect.bottom()
        y_min = self.y_min + height * rect.top()
        return WindowDimension(x_min, x_max, y_min, y_max)

    def width(self) -> BigFloat:
        return self.x_max - self.x_min

    def height(self) -> BigFloat:
        return self.y_max - self.y_min

    def re_ratio(self, ratio: float) -> 'WindowDimension':
        y_max = self.y_min + self.width() * (1 / ratio)
        return WindowDimension(self.x_min, self.x_max, self.y_min, y_max)
