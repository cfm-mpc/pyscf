# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import division
import numba as nb
import numpy as np
from numba import cuda

@cuda.jit(nb.void(nb.float32[:], nb.float32[:], nb.int32, nb.int32, nb.float64, nb.float64,\
           nb.float32[:, :], nb.float32[:, :]))
def div_eigenenergy_gpu_float32(n2e, n2f, nfermi, vstart, omega_real, omega_imag, nm2v_re, nm2v_im):
    """
    multiply the temporary matrix by (fn - fm) (frac{1.0}{w - (Em-En) -1} -
        frac{1.0}{w + (Em - En)})
    using numba
    """

    neigv = n2e.shape[-1]
    n = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    m = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if n < nfermi and m < neigv-vstart:
        a0 = omega_real**2 - omega_imag**2
        b = 2*omega_real*omega_imag
        en = n2e[n]
        fn = n2f[n]

        em = n2e[m+vstart]
        fm = n2f[m+vstart]

        a = a0 - (em -en)**2
        factor = 2*(fn - fm)*(em - en)
        den = a**2 + b**2

        nm2v_re_new = factor*(a*nm2v_re[n, m] + b*nm2v_im[n, m])/den
        nm2v_im_new = factor*(a*nm2v_im[n, m] - b*nm2v_re[n, m])/den

        nm2v_re[n, m] = nm2v_re_new
        nm2v_im[n, m] = nm2v_im_new

    if n > vstart and n < nfermi:
        if m < n-vstart:
            nm2v_re[n, m] = 0.0
            nm2v_im[n, m] = 0.0

@cuda.jit(nb.void(nb.float64[:], nb.float64[:], nb.int32, nb.int32, nb.float64, nb.float64,\
           nb.float64[:, :], nb.float64[:, :]))
def div_eigenenergy_gpu_float64(n2e, n2f, nfermi, vstart, omega_real, omega_imag, nm2v_re, nm2v_im):
    """
    multiply the temporary matrix by (fn - fm) (frac{1.0}{w - (Em-En) -1} -
        frac{1.0}{w + (Em - En)})
    using numba
    """

    neigv = n2e.shape[-1]
    n = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    m = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if n < nfermi and m < neigv-vstart:
        a0 = omega_real**2 - omega_imag**2
        b = 2*omega_real*omega_imag
        en = n2e[n]
        fn = n2f[n]

        em = n2e[m+vstart]
        fm = n2f[m+vstart]

        a = a0 - (em -en)**2
        factor = 2*(fn - fm)*(em - en)
        den = a**2 + b**2

        nm2v_re_new = factor*(a*nm2v_re[n, m] + b*nm2v_im[n, m])/den
        nm2v_im_new = factor*(a*nm2v_im[n, m] - b*nm2v_re[n, m])/den

        nm2v_re[n, m] = nm2v_re_new
        nm2v_im[n, m] = nm2v_im_new

    if n > vstart and n < nfermi:
        if m < n-vstart:
            nm2v_re[n, m] = 0.0
            nm2v_im[n, m] = 0.0
