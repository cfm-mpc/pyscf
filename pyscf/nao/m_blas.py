from __future__ import division
import numpy as np
from scipy.linalg import blas
import numba as nb

def gemv(alpha, A, x, trans=0):

    if A.dtype == np.float32:
        return blas.sgemv(alpha, A, x, trans=trans)
    elif A.dtype == np.float64:
        return blas.dgemv(alpha, A, x, trans=trans)
    else:
        print(A.dtype)
        raise ValueError("unsupported dtype")

@nb.jit(nopython=True, parallel=True)
def matvec(A, x, trans=False):

    if not trans:
        led = A.shape[0]
        fol = A.shape[1]
    else:
        led = A.shape[1]
        fol = A.shape[0]

    y = np.zeros((led), dtype=x.dtype)
    for i in nb.prange(led):
        for j in range(fol):
            if trans:
                y[i] += A[j, i]*x[j]
            else:
                y[i] += A[i, j]*x[j]

    return y
