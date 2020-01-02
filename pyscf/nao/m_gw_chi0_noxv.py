from __future__ import division
import sys
from timeit import default_timer as timer
import numpy as np
from pyscf.nao.m_chi0_noxv import calc_sab, div_eigenenergy, get_ab2v
from pyscf.nao.m_sparsetools import csr_matvec

def gw_chi0_mv(self, dvin, comega=1j*0.0, timing=None):

    # real part
    sab_re = calc_sab(self.cc_da_csr, self.v_dab_trans,
                      dvin.real, timing[0:2]).reshape(self.norbs,self.norbs)

    # imaginary
    sab_im = calc_sab(self.cc_da_csr, self.v_dab_trans,
                      dvin.imag, timing[2:4]).reshape(self.norbs,self.norbs)

    # This loop is oddly done and useless, it just take the last spin
    # component, MUST be checked
    for spin in range(self.nspin):

        ab2v_re, ab2v_im = get_ab2v(self, sab_re, sab_im, spin, comega,
                                    timing[4:13])
        
    chi0_re = calc_sab(self.v_dab_csr, self.cc_da_trans, ab2v_re,
                       timing[13:15])
    chi0_im = calc_sab(self.v_dab_csr, self.cc_da_trans, ab2v_im,
                       timing[15:17])

    return chi0_re + 1.0j*chi0_im

#
#
#
def gw_chi0_mv_gpu(self, v, comega=1j*0.0):
# check with nspin=2
    """
        Apply the non-interacting response function to a vector using gpu for
        matrix-matrix multiplication
    """
    assert self.nspin == 1

    if self.dtype != np.float32:
        print(self.dtype)
        raise ValueError("GPU version only with single precision")

    # real part
    sab = calc_sab(self.cc_da_csr, self.v_dab_trans,
                   v.real).reshape([self.norbs, self.norbs])
    self.td_GPU.cpy_sab_to_device(sab, Async = 1)
    self.td_GPU.calc_nb2v_from_sab(reim=0)

    # nm2v_real
    self.td_GPU.calc_nm2v_real()

    # start imaginary part
    sab = calc_sab(self.cc_da_csr, self.v_dab_trans,
                   v.imag).reshape([self.norbs, self.norbs])
    self.td_GPU.cpy_sab_to_device(sab, Async = 2)

    self.td_GPU.calc_nb2v_from_sab(reim=1)
    # nm2v_imag
    self.td_GPU.calc_nm2v_imag()

    self.td_GPU.div_eigenenergy_gpu(comega)
    # real part
    self.td_GPU.calc_nb2v_from_nm2v_real()
    self.td_GPU.calc_sab(reim=0)
    self.td_GPU.cpy_sab_to_host(sab, Async = 1)

    # start calc_ imag to overlap with cpu calculations
    self.td_GPU.calc_nb2v_from_nm2v_imag()

    vdp = csr_matvec(self.v_dab_csr, sab)

    self.td_GPU.calc_sab(reim=1)

    # finish real part
    chi0_re = csr_matvec(self.cc_da_trans, vdp)

    # imag part
    self.td_GPU.cpy_sab_to_host(sab)

    vdp = csr_matvec(self.v_dab_csr, sab)
    chi0_im = csr_matvec(self.cc_da_trans, vdp)

    return chi0_re + 1.0j*chi0_im
