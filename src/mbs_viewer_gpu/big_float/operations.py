import numba as nb
import numpy as np
from numba import cuda

from .fft import fft, ifft
from ..config import PRECISION, DOUBLE_PRECISION, FACTOR_PER_CELL, DIVMOD_BUFFER

_MUL_PRECISION = PRECISION * 2 - 1


def to_num_cpu(arr: np.ndarray) -> float:
    total = 0.0
    for element in arr:
        total /= FACTOR_PER_CELL
        total += element
    return total


@cuda.jit(device=True)
def poly_mul_gpu_with_fft(z_real: np.ndarray, z_imag: np.ndarray):
    combined = cuda.local.array(DOUBLE_PRECISION, nb.complex128)
    for i in range(PRECISION):
        combined[i] = complex(z_real[i], z_imag[i])
    for i in range(PRECISION, DOUBLE_PRECISION):
        combined[i] = 0
    fft(combined)

    for i in range(DOUBLE_PRECISION):
        combined[i] = combined[i] * combined[i]
    ifft(combined)

    real_leftover = 0
    imag_leftover = 0
    for i in range(PRECISION):
        real = real_leftover + round(combined[i + PRECISION - 1].real)
        real_leftover, z_real[i] = divmod(real + DIVMOD_BUFFER, FACTOR_PER_CELL)
        z_real[i] -= DIVMOD_BUFFER

        imag = imag_leftover + round(combined[i + PRECISION - 1].imag)
        imag_leftover, z_imag[i] = divmod(imag + DIVMOD_BUFFER, FACTOR_PER_CELL)
        z_imag[i] -= DIVMOD_BUFFER


@cuda.jit(device=True)
def poly_mul_gpu(z_real: np.ndarray, z_imag: np.ndarray):
    # noinspection PyTypeChecker
    tmp_real = cuda.local.array(_MUL_PRECISION, nb.int64)
    # noinspection PyTypeChecker
    tmp_imag = cuda.local.array(_MUL_PRECISION, nb.int64)

    total_real = 0
    total_imag = 0
    for tmp_index in range(_MUL_PRECISION):
        for index1 in range(max(0, tmp_index - PRECISION + 1), min(tmp_index + 1, PRECISION)):
            index2 = tmp_index - index1
            total_real += z_real[index1] * z_real[index2]
            total_real -= z_imag[index1] * z_imag[index2]
            total_imag += z_imag[index1] * z_real[index2]
            total_imag += z_real[index1] * z_imag[index2]
        total_real, tmp_real[tmp_index] = divmod(total_real + DIVMOD_BUFFER, FACTOR_PER_CELL)
        total_imag, tmp_imag[tmp_index] = divmod(total_imag + DIVMOD_BUFFER, FACTOR_PER_CELL)
        tmp_real[tmp_index] -= DIVMOD_BUFFER
        tmp_imag[tmp_index] -= DIVMOD_BUFFER

    for i in range(PRECISION):
        z_real[i] = tmp_real[PRECISION - 1 + i]
        z_imag[i] = tmp_imag[PRECISION - 1 + i]


def sub_(arr1: np.ndarray, arr2: np.ndarray, out: np.ndarray) -> np.ndarray:
    leftover = 0
    for i in range(out.size):
        leftover, out[i] = divmod(arr1[i] + leftover - arr2[i] + DIVMOD_BUFFER, FACTOR_PER_CELL)
        out[i] -= DIVMOD_BUFFER
    assert not leftover
    return out


def add_(arr1: np.ndarray, arr2: np.ndarray, out: np.ndarray) -> np.ndarray:
    leftover = 0
    for i in range(out.size):
        leftover, out[i] = divmod(arr1[i] + leftover + arr2[i] + DIVMOD_BUFFER, FACTOR_PER_CELL)
        out[i] -= DIVMOD_BUFFER
    assert not leftover
    return out


def div_cpu(arr: np.ndarray, num: int) -> np.ndarray:
    leftover = 0
    for i in range(arr.size - 1, -1, -1):
        leftover *= FACTOR_PER_CELL
        arr[i], leftover = divmod(arr[i] + leftover + num // 2, num)
        leftover -= num // 2
    return arr


@cuda.jit(device=True)
def mul_gpu(arr: np.ndarray, num: int) -> np.ndarray:
    leftover = 0
    for i in range(arr.size):
        leftover, arr[i] = divmod(arr[i] * num + leftover + DIVMOD_BUFFER, FACTOR_PER_CELL)
        arr[i] -= DIVMOD_BUFFER
    assert not leftover
    return arr


sub_gpu = cuda.jit(device=True)(sub_)
add_gpu = cuda.jit(device=True)(add_)
div_gpu = cuda.jit(device=True)(div_cpu)
to_num_gpu = cuda.jit(device=True)(to_num_cpu)
