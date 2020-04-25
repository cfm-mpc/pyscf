from __future__ import division
from timeit import default_timer as timer
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from scipy.linalg import blas
from pyscf.nao.m_sparsetools import csr_matvec, csc_matvec, csc_matvecs
import math
  
def chi0_mv(self, dvin, comega=1j*0.0, dnout=None, timing=None):
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
    
    for spin in range(self.nspin):

        # real part
        sab_re = calc_sab(self.cc_da_csr, self.v_dab_trans,
                       sp2v[spin].real, timing[0:2]).reshape((self.norbs,self.norbs))
    
        # imaginary
        sab_im = calc_sab(self.cc_da_csr, self.v_dab_trans,
                       sp2v[spin].imag, timing[2:4]).reshape((self.norbs,self.norbs))

        ab2v_re, ab2v_im = get_ab2v(self, sab_re, sab_im, spin, comega,
                                    timing[4:13])

        chi0_re = calc_sab(self.v_dab_csr, self.cc_da_trans, ab2v_re,
                           timing[13:15])
        chi0_im = calc_sab(self.v_dab_csr, self.cc_da_trans, ab2v_im,
                           timing[15:17])

        sp2dn[spin] = chi0_re + 1.0j*chi0_im
      
    return dnout

def chi0_mv_gpu(self, dvin, comega=1j*0.0, timing=None):
    """
    TODO: nspin == 2
    """
    import cupy as cp

    assert self.nspin == 1
    spin = 0

    # real part
    vdp = csr_matvec(self.cc_da_csr, dvin.real)
    sab_re = csr_matvec(self.v_dab_trans, vdp).reshape((self.norbs,self.norbs))
    sab_re_gpu = cp.asarray(sab_re)

    nb2v = self.xocc_gpu[spin].dot(sab_re_gpu)
    nm2v_re = nb2v.dot(self.xvrt_gpu[spin].T)

    # imaginary
    vdp = csr_matvec(self.cc_da_csr, dvin.imag)
    sab_im = csr_matvec(self.v_dab_trans, vdp).reshape((self.norbs,self.norbs))
    sab_im_gpu = cp.asarray(sab_im)

    nb2v = self.xocc_gpu[spin].dot(sab_im_gpu)
    nm2v_im = nb2v.dot(self.xvrt_gpu[spin].T)

    vs, nf = self.vstart[spin], self.nfermi[spin]
    div_eigenenergy(self.ksn2e_gpu, self.ksn2f_gpu, spin, nf, vs, comega,
                    nm2v_re, nm2v_im, div_numba=self.div_numba,
                    use_numba=self.use_numba)

    # real part
    nb2v = nm2v_re.dot(self.xvrt_gpu[spin])
    ab2v = self.xocc_gpu[spin].T.dot(nb2v)
    ab2v_re = cp.asnumpy(ab2v).reshape(self.norbs*self.norbs)
    
    vdp = csr_matvec(self.v_dab_csr, ab2v_re)
    chi0_re = csr_matvec(self.cc_da_trans, vdp)

    # imag part
    nb2v = nm2v_im.dot(self.xvrt_gpu[spin])
    ab2v = self.xocc_gpu[spin].T.dot(nb2v)
    ab2v_im = cp.asnumpy(ab2v).reshape(self.norbs*self.norbs)

    vdp = csr_matvec(self.v_dab_csr, ab2v_im)
    chi0_im = csr_matvec(self.cc_da_trans, vdp)

    return chi0_re + 1.0j*chi0_im

def calc_sab(mat1, mat2, vec, timing):

    t1 = timer()
    vdp = csr_matvec(mat1, vec)
    t2 = timer()
    timing[0] += t2 - t1
    
    t1 = timer()
    sab = csr_matvec(mat2, vdp)
    t2 = timer()
    timing[1] += t2 - t1

    return sab

def div_eigenenergy(ksn2e, ksn2f, spin, nf, vs, comega, nm2v_re, nm2v_im,
                    div_numba=None, use_numba=False):
    
    if use_numba and div_numba is not None:
        div_numba(ksn2e[0, spin], ksn2f[0, spin], nf, vs, comega,
                  nm2v_re, nm2v_im)
    else:
        for n, (en, fn) in enumerate(zip(ksn2e[0, spin, :nf], ksn2f[0, spin, :nf])):
            for m, (em, fm) in enumerate(zip(ksn2e[0, spin, vs:], ksn2f[0, spin, vs:])):
                nm2v = nm2v_re[n, m] + 1.0j*nm2v_im[n, m]
                nm2v = nm2v * (fn - fm) * \
                ( 1.0 / (comega - (em - en)) - 1.0 / (comega + (em - en)) )
                nm2v_re[n, m] = nm2v.real
                nm2v_im[n, m] = nm2v.imag

        # padding m<n i.e. negative occupations' difference
        for n in range(vs+1, nf):
            for m in range(n-vs):
                nm2v_re[n,m], nm2v_im[n,m] = 0.0, 0.0

def get_ab2v(self, sab_re, sab_im, spin, comega, timing):

    t1 = timer()
    #nb2v = self.gemm(1.0, self.xocc[spin], sab_re)
    nb2v = self.xocc[spin].dot(sab_re)
    t2 = timer()
    timing[0] += t2 - t1

    t1 = timer()
    #nm2v_re = self.gemm(1.0, nb2v, self.xvrt[spin], trans_b=1)
    nm2v_re = nb2v.dot(self.xvrt[spin].T)
    t2 = timer()
    timing[1] += t2 - t1

    t1 = timer()
    #nb2v = self.gemm(1.0, self.xocc[spin], sab_im)
    nb2v = self.xocc[spin].dot(sab_im)
    t2 = timer()
    timing[2] += t2 - t1
    
    t1 = timer()
    #nm2v_im = self.gemm(1.0, nb2v, self.xvrt[spin], trans_b=1)
    nm2v_im = nb2v.dot(self.xvrt[spin].T)
    t2 = timer()
    timing[3] += t2 - t1

    t1 = timer()
    vs, nf = self.vstart[spin], self.nfermi[spin]
    div_eigenenergy(self.ksn2e, self.ksn2f, spin, nf, vs, comega, nm2v_re,
                    nm2v_im, div_numba=self.div_numba,
                    use_numba=self.use_numba)
    t2 = timer()
    timing[4] += t2 - t1

    # real part
    t1 = timer()
    #nb2v = self.gemm(1.0, nm2v_re, self.xvrt[spin])
    nb2v = nm2v_re.dot(self.xvrt[spin])
    t2 = timer()
    timing[5] += t2 - t1

    t1 = timer()
    #ab2v_re = self.gemm(1.0, self.xocc[spin], nb2v, trans_a=1).reshape(self.norbs*self.norbs)
    ab2v_re = self.xocc[spin].T.dot(nb2v).reshape(self.norbs*self.norbs)
    t2 = timer()
    timing[6] += t2 - t1

    # imag part
    t1 = timer()
    #nb2v = self.gemm(1.0, nm2v_im, self.xvrt[spin])
    nb2v = nm2v_im.dot(self.xvrt[spin])
    t2 = timer()
    timing[7] += t2 - t1

    t1 = timer()
    #ab2v_im = self.gemm(1.0, self.xocc[spin], nb2v, trans_a=1).reshape(self.norbs*self.norbs)
    ab2v_im = self.xocc[spin].T.dot(nb2v).reshape(self.norbs*self.norbs)
    t2 = timer()
    timing[8] += t2 - t1

    return ab2v_re, ab2v_im
