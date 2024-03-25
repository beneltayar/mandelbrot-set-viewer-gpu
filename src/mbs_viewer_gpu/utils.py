import matplotlib
import numpy as np


def colorize_mandel_image(image: np.ndarray, num_iterations_per_color_cycle: int = 1000) -> np.ndarray:
    ones = np.zeros_like(image, dtype=np.float32)
    ones[image != 0] = 1
    new = np.stack([(image / num_iterations_per_color_cycle) % 1, ones, ones], axis=-1)
    gimage = matplotlib.colors.hsv_to_rgb(new)
    return (gimage * (256 - 1e-9)).astype(np.uint8)
