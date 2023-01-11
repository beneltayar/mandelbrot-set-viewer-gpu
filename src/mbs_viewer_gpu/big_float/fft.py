import cmath
import math

import numba as nb
import numpy as np
from numba import cuda

from ..config import PRECISION, PRECISION_LOG

PRECISION_LOG = PRECISION_LOG + 1
_HALF_PRECISION = PRECISION
PRECISION = PRECISION * 2


@cuda.jit(device=True)
def reverse(n: int):
    p = 0
    for i in range(1, PRECISION_LOG + 1):
        if n & (1 << (PRECISION_LOG - i)):
            p |= 1 << (i - 1)
    return p


@cuda.jit(device=True)
def ordina(arr: np.ndarray):
    tmp = cuda.local.array(PRECISION, nb.complex128)
    for i in range(PRECISION):
        tmp[i] = arr[reverse(i)]
    for i in range(PRECISION):
        arr[i] = tmp[i]


@cuda.jit(device=True)
def fft(arr: np.ndarray):
    ordina(arr)
    roots = cuda.local.array(_HALF_PRECISION, nb.complex128)
    factor = cmath.rect(1., -math.tau / PRECISION)
    for i in range(_HALF_PRECISION):
        roots[i] = factor ** i

    n = 1
    a = _HALF_PRECISION
    for j in range(PRECISION_LOG):
        for i in range(PRECISION):
            if not (i & n):
                temp = arr[i]
                temp2 = roots[(i * a) % (n * a)] * arr[i + n]
                arr[i] = temp + temp2
                arr[i + n] = temp - temp2
        n *= 2
        a = a // 2


@cuda.jit(device=True)
def ifft(arr: np.ndarray):
    for i in range(PRECISION):
        arr[i] = arr[i].conjugate()
    fft(arr)
    for i in range(PRECISION):
        arr[i] = complex(arr[i].real / PRECISION, arr[i].imag / -PRECISION)
    return arr
