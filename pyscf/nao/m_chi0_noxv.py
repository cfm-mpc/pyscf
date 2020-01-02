from __future__ import division
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from scipy.linalg import blas
from pyscf.nao.m_sparsetools import csr_matvec, csc_matvec, csc_matvecs
import math
  
def chi0_mv(self, dvin, comega=1j*0.0, dnout=None):
    """
        Apply the non-interacting response function to a vector
        Input Parameters:
        -----------------
            self : tddft_iter or tddft_tem class
            sp2v : vector describing the effective perturbation [spin*product] --> value
            comega: complex frequency
    """
    if dnout is None:
        dnout = np.zeros_like(dvin, dtype=self.dtypeComplex)

    sp2v  = dvin.reshape((self.nspin,self.nprod))
    sp2dn = dnout.reshape((self.nspin,self.nprod))
    
    for s in range(self.nspin):

        # real part
        sab = calc_sab(self.cc_da_csr, self.v_dab_trans,
                       sp2v[s].real).reshape((self.norbs,self.norbs))
    
        nb2v = self.gemm(1.0, self.xocc[s], sab)
        nm2v_re = self.gemm(1.0, nb2v, self.xvrt[s], trans_b=1)

        # imaginary
        sab = calc_sab(self.cc_da_csr, self.v_dab_trans,
                       sp2v[s].imag).reshape((self.norbs,self.norbs))
      
        nb2v = self.gemm(1.0, self.xocc[s], sab)
        nm2v_im = self.gemm(1.0, nb2v, self.xvrt[s], trans_b=1)

        vs,nf = self.vstart[s],self.nfermi[s]
        if self.use_numba:
            self.div_numba(self.ksn2e[0,s], self.ksn2f[0,s], nf, vs, comega, nm2v_re, nm2v_im)
        else:
            for n,(en,fn) in enumerate(zip(self.ksn2e[0,s,:nf], self.ksn2f[0,s,:nf])):
                for m,(em,fm) in enumerate(zip(self.ksn2e[0,s,vs:],self.ksn2f[0,s,vs:])):
                    nm2v = nm2v_re[n, m] + 1.0j*nm2v_im[n, m]
                    nm2v = nm2v * (fn - fm) * \
                    ( 1.0 / (comega - (em - en)) - 1.0 / (comega + (em - en)) )
                    nm2v_re[n, m] = nm2v.real
                    nm2v_im[n, m] = nm2v.imag

            # padding m<n i.e. negative occupations' difference
            for n in range(vs+1,nf):
                for m in range(n-vs):
                    nm2v_re[n,m], nm2v_im[n,m] = 0.0,0.0

        # real part
        nb2v = self.gemm(1.0, nm2v_re, self.xvrt[s])
        ab2v = self.gemm(1.0, self.xocc[s], nb2v, trans_a=1).reshape(self.norbs*self.norbs)
        chi0_re = calc_sab(self.v_dab_csr, self.cc_da_trans, ab2v)

        # imag part
        nb2v = self.gemm(1.0, nm2v_im, self.xvrt[s])
        ab2v = self.gemm(1.0, self.xocc[s], nb2v, trans_a=1).reshape(self.norbs*self.norbs)
        chi0_im = calc_sab(self.v_dab_csr, self.cc_da_trans, ab2v)

        sp2dn[s] = chi0_re + 1.0j*chi0_im
      
    return dnout

#
#
#
def chi0_mv_gpu(self, v, comega=1j*0.0):
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

def calc_sab(mat1, mat2, vec):
    vdp = csr_matvec(mat1, vec)
    return csr_matvec(mat2, vdp)
