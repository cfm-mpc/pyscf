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

@nb.jit(nopython=True, parallel=True)
def div_eigenenergy_numba(n2e, n2f, nfermi, vstart, comega, nm2v_re, nm2v_im):
    """
    multiply the temporary matrix by (fn - fm) (frac{1.0}{w - (Em-En) -1} -
        frac{1.0}{w + (Em - En)})
    using numba
    """

    neigv = n2e.shape[-1]
    a0 = comega.real**2 - comega.imag**2
    b = 2*comega.real*comega.imag
    
    for n in nb.prange(nfermi):
        en = n2e[n]
        fn = n2f[n]

        for m in range(neigv-vstart):
            em = n2e[m+vstart]
            fm = n2f[m+vstart]

            a = a0 - (em -en)**2
            factor = 2*(fn - fm)*(em - en)
            den = a**2 + b**2

            nm2v_re_nm = factor*(a*nm2v_re[n, m] + b*nm2v_im[n, m])/den
            nm2v_im_nm = factor*(a*nm2v_im[n, m] - b*nm2v_re[n, m])/den
            #nm2v = nm2v_re[n, m] + 1.0j*nm2v_im[n, m]
            #nm2v = nm2v * (fn-fm) * \
            #  ( 1.0 / (comega - (em - en)) - 1.0 / (comega + (em - en)) )
            #
            nm2v_re[n, m] = nm2v_re_nm
            nm2v_im[n, m] = nm2v_im_nm

    for n in nb.prange(vstart+1, nfermi):
        for m in range(n-vstart):
            nm2v_re[n, m] = 0.0 
            nm2v_im[n, m] = 0.0

@cuda.jit()
def div_eigenenergy_gpu(n2e, n2f, nfermi, vstart, comega, nm2v_re, nm2v_im):
    """
    multiply the temporary matrix by (fn - fm) (frac{1.0}{w - (Em-En) -1} -
        frac{1.0}{w + (Em - En)})
    using numba
    """

    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    #for i in range(start, x.shape[0], stride):
    #    out[i] = x[i] + y[i]

    neigv = n2e.shape[-1]
    a0 = comega.real**2 - comega.imag**2
    b = 2*comega.real*comega.imag

    for n in range(start, nfermi, stride):
        en = n2e[n]
        fn = n2f[n]

        for m in range(neigv-vstart):
            em = n2e[m+vstart]
            fm = n2f[m+vstart]

            a = a0 - (em -en)**2
            factor = 2*(fn - fm)*(em - en)
            den = a**2 + b**2

            nm2v_re_nm = factor*(a*nm2v_re[n, m] + b*nm2v_im[n, m])/den
            nm2v_im_nm = factor*(a*nm2v_im[n, m] - b*nm2v_re[n, m])/den
            #nm2v = nm2v_re[n, m] + 1.0j*nm2v_im[n, m]
            #nm2v = nm2v * (fn-fm) * \
            #  ( 1.0 / (comega - (em - en)) - 1.0 / (comega + (em - en)) )
            #
            nm2v_re[n, m] = nm2v_re_nm
            nm2v_im[n, m] = nm2v_im_nm

    #for n in range(vstart+1, nfermi):
    for n in range(start, nfermi, stride):
        if n < vstart+1: continue
        for m in range(n-vstart):
            nm2v_re[n, m] = 0.0
            nm2v_im[n, m] = 0.0

@nb.jit(nopython=True)
def mat_mul_numba(a, b):
    return a*b
