import matplotlib
import numpy as np

from .config import NUM_ITERATIONS_PER_COLOR_CYCLE


def colorize_mandel_image(image: np.ndarray) -> np.ndarray:
    ones = np.zeros_like(image, dtype=np.float32)
    ones[image != 0] = 1
    new = np.stack([(image / NUM_ITERATIONS_PER_COLOR_CYCLE) % 1, ones, ones], axis=-1)
    gimage = matplotlib.colors.hsv_to_rgb(new)
    return (gimage * (256 - 1e-9)).astype(np.uint8)
