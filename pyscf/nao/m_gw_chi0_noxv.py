from __future__ import division
import sys
from timeit import default_timer as timer
import numpy as np
from pyscf.nao.m_chi0_noxv import calc_sab, div_eigenenergy

def gw_chi0_mv(self, dvin, comega=1j*0.0):

    # real part
    sab_real = calc_sab(self.cc_da_csr, self.v_dab_trans,
                        dvin.real).reshape(self.norbs,self.norbs)

    # imaginary
    sab_imag = calc_sab(self.cc_da_csr, self.v_dab_trans,
                        dvin.imag).reshape(self.norbs,self.norbs)

    ab2v_re = np.zeros((self.norbs, self.norbs), dtype=self.dtype)
    ab2v_im = np.zeros((self.norbs, self.norbs), dtype=self.dtype)

    for spin in range(self.nspin):
        
        nb2v = self.gemm(1.0, self.xocc[spin], sab_real)
        nm2v_re = self.gemm(1.0, nb2v, self.xvrt[spin], trans_b=1)

        nb2v = self.gemm(1.0, self.xocc[spin], sab_imag)
        nm2v_im = self.gemm(1.0, nb2v, self.xvrt[spin], trans_b=1)

        vs, nf = self.vstart[spin], self.nfermi[spin]

        div_eigenenergy(self.ksn2e, self.ksn2f, spin, nf, vs, comega, nm2v_re,
                        nm2v_im, div_numba=self.div_numba,
                        use_numba=self.use_numba)

        # real part
        nb2v = self.gemm(1.0, nm2v_re, self.xvrt[spin])
        ab2v_re = self.gemm(1.0, self.xocc[spin], nb2v, 1.0, ab2v_re, trans_a=1)

        # imag part
        nb2v = self.gemm(1.0, nm2v_im, self.xvrt[spin])
        ab2v_im = self.gemm(1.0, self.xocc[spin], nb2v, 1.0, ab2v_im, trans_a=1)

    chi0_re = calc_sab(self.v_dab_csr, self.cc_da_trans, ab2v_re.reshape(self.norbs*self.norbs))
    chi0_im = calc_sab(self.v_dab_csr, self.cc_da_trans, ab2v_im.reshape(self.norbs*self.norbs))

    return chi0_re + 1.0j*chi0_im
