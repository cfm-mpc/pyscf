from __future__ import division
import sys
from timeit import default_timer as timer
import numpy as np
from pyscf.lib import misc
from pyscf.nao.m_chi0_noxv import calc_sab, div_eigenenergy, get_ab2v

def gw_chi0_mv(self, dvin, comega=1j*0.0, timing=None):

    # real part
    sab_re = calc_sab(self.cc_da_csr, self.v_dab_trans,
                      dvin.real, timing[0:2]).reshape(self.norbs,self.norbs)

    # imaginary
    sab_im = calc_sab(self.cc_da_csr, self.v_dab_trans,
                      dvin.imag, timing[2:4]).reshape(self.norbs,self.norbs)

    ab2v_re = None
    ab2v_im = None
    for spin in range(self.nspin):

        if ab2v_re is None:
            ab2v_re, ab2v_im = get_ab2v(self.xocc[spin], self.xvrt[spin],
                                        self.vstart[spin], self.nfermi[spin],
                                        self.norbs, self.ksn2e[0, spin],
                                        self.ksn2f[0, spin],
                                        sab_re, sab_im, comega, self.div_numba,
                                        self.use_numba, timing[4:13])
        else:
            matre, matim = get_ab2v(self.xocc[spin], self.xvrt[spin],
                                    self.vstart[spin], self.nfermi[spin],
                                    self.norbs, self.ksn2e[0, spin],
                                    self.ksn2f[0, spin],
                                    sab_re, sab_im, comega, self.div_numba,
                                    self.use_numba, timing[4:13])
            ab2v_re += matre
            ab2v_im += matim

    # real part
    chi0_re = calc_sab(self.v_dab_csr, self.cc_da_trans, ab2v_re, timing[13:15])

    # imag part
    chi0_im = calc_sab(self.v_dab_csr, self.cc_da_trans, ab2v_im, timing[15:17])

    return chi0_re + 1.0j*chi0_im

def gw_chi0_mv_gpu(self, dvin, comega=1j*0.0, timing=None):
    """
    """
    import cupy as cp

    # real part
    sab_re = calc_sab(self.cc_da_csr, self.v_dab_trans,
                      dvin.real, timing[0:2]).reshape(self.norbs,self.norbs)
    t1 = timer()
    sab_re_gpu = cp.asarray(sab_re)
    t2 = timer()
    timing[1] += t2-t1

    # imaginary
    sab_im = calc_sab(self.cc_da_csr, self.v_dab_trans,
                      dvin.imag, timing[2:4]).reshape(self.norbs,self.norbs)
    t1 = timer()
    sab_im_gpu = cp.asarray(sab_im)
    t2 = timer()
    timing[3] += t2-t1

    for spin in range(self.nspin):

        if spin == 0:
            ab2v_re_gpu, ab2v_im_gpu = get_ab2v(self.xocc_gpu[spin],
                                                self.xvrt_gpu[spin],
                                                self.vstart[spin], self.nfermi[spin],
                                                self.norbs, self.ksn2e_gpu[0, spin],
                                                self.ksn2f_gpu[0, spin],
                                                sab_re_gpu, sab_im_gpu, comega,
                                                self.div_numba,
                                                self.use_numba, timing[4:13],
                                                GPU=True,
                                                blockspergrid=self.block_size[spin],
                                                threadsperblock=self.grid_size[spin])
        else:
            matre, matim = get_ab2v(self.xocc_gpu[spin], self.xvrt_gpu[spin],
                                    self.vstart[spin], self.nfermi[spin],
                                    self.norbs, self.ksn2e_gpu[0, spin],
                                    self.ksn2f_gpu[0, spin],
                                    sab_re_gpu, sab_im_gpu, comega,
                                    self.div_numba,
                                    self.use_numba, timing[4:13],
                                    GPU=True, blockspergrid=self.block_size[spin],
                                    threadsperblock=self.grid_size[spin])
            ab2v_re_gpu += matre
            ab2v_im_gpu += matim

    # real part
    t1 = timer()
    ab2v_re = cp.asnumpy(ab2v_re_gpu)
    t2 = timer()
    timing[13] += t2-t1
    chi0_re = calc_sab(self.v_dab_csr, self.cc_da_trans, ab2v_re, timing[13:15])

    # imag part
    t1 = timer()
    ab2v_im = cp.asnumpy(ab2v_im_gpu)
    t2 = timer()
    timing[15] += t2-t1
    chi0_im = calc_sab(self.v_dab_csr, self.cc_da_trans, ab2v_im, timing[15:17])

    return chi0_re + 1.0j*chi0_im
