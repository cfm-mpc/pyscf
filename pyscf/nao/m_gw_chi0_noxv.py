from __future__ import division
import sys
from timeit import default_timer as timer
import numpy as np
from pyscf.lib import misc
from pyscf.nao.m_chi0_noxv import calc_sab, div_eigenenergy, get_ab2v
from pyscf.nao.m_sparsetools import csr_matvec
from ctypes import POINTER, c_double, c_int, c_int64, c_float, c_int

libsparsetools = misc.load_library("libsparsetools")

def csr_matrix_ctype(csr):
    """
    store a csr matrix into a dictionary alreay getting the ctype data
    """

    dico = {}
    dico["nrow"] = c_int(csr.shape[0])
    dico["ncol"] = c_int(csr.shape[1])
    dico["nnz"] = c_int(csr.nnz)
    dico["indptr"] = csr.indptr.ctypes.data_as(POINTER(c_int))
    dico["indices"] = csr.indices.ctypes.data_as(POINTER(c_int))
    dico["dtype"] = csr.dtype

    if csr.dtype == np.float32:
        dico["data"] = csr.data.ctypes.data_as(POINTER(c_float))
    elif csr.dtype == np.float64:
        dico["data"] = csr.data.ctypes.data_as(POINTER(c_double))
    else:
        raise ValueError("Unsupported dtype")

    return dico

def csr_matvec_ctype(csr, x, y):

    xx = np.require(x, requirements=["A", "O"], dtype=csr["dtype"])

    if csr["dtype"] == np.float32:
        libsparsetools.scsr_matvec(csr["nrow"], csr["ncol"], csr["nnz"],
                                   csr["indptr"], csr["indices"], csr["data"],
                                   xx.ctypes.data_as(POINTER(c_float)),
                                   y.ctypes.data_as(POINTER(c_float)))
    elif csr["dtype"] == np.float64:
        libsparsetools.dcsr_matvec(csr["nrow"], csr["ncol"], csr["nnz"],
                                   csr["indptr"], csr["indices"], csr["data"],
                                   xx.ctypes.data_as(POINTER(c_double)),
                                   y.ctypes.data_as(POINTER(c_double)))
    else:
        raise ValueError("Unsupported dtype")

class gw_chi0_matvec():

    def __init__(self, ksn2e, ksn2f, vstart, nfermi, xocc, xvrt, cc_da_csr, cc_da_trans,
                 v_dab_csr, v_dab_trans, norbs, nspin, div_numba):

        self.cc_da_csr = csr_matrix_ctype(cc_da_csr)
        self.cc_da_trans = csr_matrix_ctype(cc_da_trans)
        self.v_dab_csr = csr_matrix_ctype(v_dab_csr)
        self.v_dab_trans = csr_matrix_ctype(v_dab_trans)

        self.ksn2e = ksn2e
        self.ksn2f = ksn2f
        self.vstart = vstart
        self.nfermi = nfermi
        self.xocc = xocc
        self.xvrt = xvrt

        self.norbs = norbs
        self.nspin = nspin
        self.div_numba = div_numba
        self.vdp = np.zeros((cc_da_csr.shape[0]), dtype=cc_da_csr.dtype)
        self.vdp2 = np.zeros((v_dab_csr.shape[0]), dtype=cc_da_csr.dtype)
        
        self.sab_re = np.zeros((v_dab_trans.shape[0]), dtype=cc_da_csr.dtype)
        self.sab_im = np.zeros((v_dab_trans.shape[0]), dtype=cc_da_csr.dtype)
        
        self.chi0_re = np.zeros((cc_da_trans.shape[0]), dtype=cc_da_csr.dtype)
        self.chi0_im = np.zeros((cc_da_trans.shape[0]), dtype=cc_da_csr.dtype)

        self.timing = np.zeros((17), dtype=np.float64)

        try:
            import numba
            self.use_numba = True
        except:
            self.use_numba = False


    def chi0_mv(self, dvin, comega=1j*0.0):

        # real part
        #sab_re = calc_sab(self.cc_da_csr, self.v_dab_trans,
        #                  dvin.real, timing[0:2]).reshape(self.norbs,self.norbs)

        self.vdp.fill(0.0)
        self.sab_re.fill(0.0)
        t1 = timer()
        csr_matvec_ctype(self.cc_da_csr, dvin.real, self.vdp)
        t2 = timer()
        self.timing[0] += t2-t1

        t1 = timer()
        csr_matvec_ctype(self.v_dab_trans, self.vdp, self.sab_re)
        t2 = timer()
        self.timing[1] += t2-t1

        # imaginary
        #sab_im = calc_sab(self.cc_da_csr, self.v_dab_trans,
        #                  dvin.imag, timing[2:4]).reshape(self.norbs,self.norbs)
        self.vdp.fill(0.0)
        self.sab_im.fill(0.0)
        t1 = timer()
        csr_matvec_ctype(self.cc_da_csr, dvin.imag, self.vdp)
        t2 = timer()
        self.timing[2] += t2-t1

        t1 = timer()
        csr_matvec_ctype(self.v_dab_trans, self.vdp, self.sab_im)
        t2 = timer()
        self.timing[3] += t2-t1

        ab2v_re = None
        ab2v_im = None
        for spin in range(self.nspin):

            if ab2v_re is None:
                ab2v_re, ab2v_im = get_ab2v(self, self.sab_re.reshape(self.norbs, self.norbs),
                                            self.sab_im.reshape(self.norbs, self.norbs),
                                            spin, comega, self.timing[4:13])
            else:
                matre, matim = get_ab2v(self, self.sab_re.reshape(self.norbs, self.norbs),
                                            self.sab_im.reshape(self.norbs, self.norbs),
                                            spin, comega, self.timing[4:13])
                ab2v_re += matre
                ab2v_im += matim

        # real part
        self.vdp2.fill(0.0)
        self.chi0_re.fill(0.0)
        t1 = timer()
        csr_matvec_ctype(self.v_dab_csr, ab2v_re, self.vdp2)
        t2 = timer()
        self.timing[13] += t2-t1

        t1 = timer()
        csr_matvec_ctype(self.cc_da_trans, self.vdp2, self.chi0_re)
        t2 = timer()
        self.timing[14] += t2-t1

        # imag part
        self.vdp2.fill(0.0)
        self.chi0_im.fill(0.0)
        t1 = timer()
        csr_matvec_ctype(self.v_dab_csr, ab2v_im, self.vdp2)
        t2 = timer()
        self.timing[15] += t2-t1

        t1 = timer()
        csr_matvec_ctype(self.cc_da_trans, self.vdp2, self.chi0_im)
        t2 = timer()
        self.timing[16] += t2-t1

        return self.chi0_re + 1.0j*self.chi0_im

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
            ab2v_re, ab2v_im = get_ab2v(self, sab_re,
                                        sab_im,
                                        spin, comega, timing[4:13])
        else:
            matre, matim = get_ab2v(self, sab_re,
                                    sab_im,
                                    spin, comega, timing[4:13])
            ab2v_re += matre
            ab2v_im += matim

    # real part
    chi0_re = calc_sab(self.v_dab_csr, self.cc_da_trans, ab2v_re, timing[13:15])

    # imag part
    chi0_im = calc_sab(self.v_dab_csr, self.cc_da_trans, ab2v_im, timing[15:17])

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
