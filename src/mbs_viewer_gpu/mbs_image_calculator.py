import math

import numpy as np
from numba import cuda

from .big_float.classes import WindowDimension
from .big_float.operations import to_num_gpu, sub_gpu, div_gpu, poly_mul_gpu
from .config import PRECISION


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
def calc_step(min_: np.ndarray, pixel_size: np.ndarray, val: int, out: np.ndarray):
    for i in range(PRECISION):
        out[i] = min_[i] + pixel_size[i] * val


@cuda.jit
def mandel_kernel(min_x: np.ndarray, max_x: np.ndarray, min_y: np.ndarray, max_y: np.ndarray, image: np.ndarray, iters: int):
    height = image.shape[0]
    width = image.shape[1]

    # noinspection PyUnresolvedReferences
    start_x = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
    # noinspection PyUnresolvedReferences
    start_y = cuda.blockDim.y * cuda.blockIdx.y + cuda.threadIdx.y
    if start_x >= width or start_y >= height:
        return

    # noinspection PyTypeChecker
    pixel_size_x = cuda.local.array(PRECISION, np.int64)
    # noinspection PyTypeChecker
    pixel_size_y = cuda.local.array(PRECISION, np.int64)
    sub_gpu(max_x, min_x, pixel_size_x)
    sub_gpu(max_y, min_y, pixel_size_y)
    div_gpu(pixel_size_x, width)
    div_gpu(pixel_size_y, height)

    # noinspection PyTypeChecker
    real = cuda.local.array(PRECISION, np.int64)
    # noinspection PyTypeChecker
    imag = cuda.local.array(PRECISION, np.int64)
    calc_step(min_x, pixel_size_x, start_x, real)
    calc_step(min_y, pixel_size_y, start_y, imag)
    cuda.syncthreads()
    image[start_y, start_x] = mandel(real, imag, iters)


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
