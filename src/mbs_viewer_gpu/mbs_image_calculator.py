import math

import numpy as np
from numba import cuda

from .big_float.classes import WindowDimension
from .big_float.operations import to_num_gpu, poly_mul_gpu
from .config import PRECISION, FACTOR_PER_CELL


@cuda.jit(device=True)
def mandel(real: np.ndarray, imag: np.ndarray, max_iters: int) -> int:
    """
    Given the real and imaginary parts of a complex number,
    determine if it is a candidate for membership in the Mandelbrot
    set given a fixed number of iterations.
    """
    # noinspection PyTypeChecker
    z_real = cuda.local.array(PRECISION, dtype=np.int64)
    # noinspection PyTypeChecker
    z_imag = cuda.local.array(PRECISION, dtype=np.int64)
    for i in range(PRECISION):
        z_real[i] = 0
        z_imag[i] = 0

    for i in range(max_iters):
        poly_mul_gpu(z_real, z_imag)
        for j in range(PRECISION):
            z_real[j] = z_real[j] + real[j]
            z_imag[j] = z_imag[j] + imag[j]

        z_real_scale = to_num_gpu(z_real)
        z_imag_scale = to_num_gpu(z_imag)
        if (z_real_scale * z_real_scale + z_imag_scale * z_imag_scale) >= 4:
            return i
    return 0


@cuda.jit(device=True)
def weighted_average(left: np.ndarray, right: np.ndarray, right_weight: int, total_weight: int, out: np.ndarray):
    leftover = 0
    for i in range(PRECISION - 1, -1, -1):
        out[i], leftover = divmod(left[i] * (total_weight - right_weight) + right[i] * right_weight + leftover, total_weight)
        leftover *= FACTOR_PER_CELL
    return out


@cuda.jit
def mandel_kernel(min_x: np.ndarray, max_x: np.ndarray, min_y: np.ndarray, max_y: np.ndarray, image: np.ndarray, iters: int):
    height = image.shape[0]
    width = image.shape[1]

    # noinspection PyUnresolvedReferences
    pixel_x = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
    # noinspection PyUnresolvedReferences
    pixel_y = cuda.blockDim.y * cuda.blockIdx.y + cuda.threadIdx.y
    if pixel_x >= width or pixel_y >= height:
        return

    # noinspection PyTypeChecker
    real = cuda.local.array(PRECISION, np.int64)
    # noinspection PyTypeChecker
    imag = cuda.local.array(PRECISION, np.int64)
    weighted_average(min_x, max_x, pixel_x, width, real)
    weighted_average(min_y, max_y, pixel_y, height, imag)
    cuda.syncthreads()
    image[pixel_y, pixel_x] = mandel(real, imag, iters)


def get_mandel_image(dimension: WindowDimension, image_height: int = 512, num_iterations: int = 2000):
    ratio = dimension.ratio()
    image_width = round(image_height * ratio)

    block_dim = (16, 16)
    grid_dim = (math.ceil(image_width / block_dim[1]), math.ceil(image_height / block_dim[0]))

    x_min = cuda.to_device(dimension.x_min.arr)
    x_max = cuda.to_device(dimension.x_max.arr)
    y_min = cuda.to_device(dimension.y_min.arr)
    y_max = cuda.to_device(dimension.y_max.arr)

    d_image = cuda.device_array((image_height, image_width), dtype=np.uint32)
    mandel_kernel[grid_dim, block_dim](x_min, x_max, y_min, y_max, d_image, num_iterations)
    return d_image.copy_to_host()
