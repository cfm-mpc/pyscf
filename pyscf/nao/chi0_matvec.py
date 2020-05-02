from __future__ import print_function, division
from copy import copy
import numpy as np
from numpy import array, argmax
from scipy.sparse import csr_matrix, coo_matrix
from timeit import default_timer as timer
from scipy.linalg import blas
import scipy.sparse.linalg as splin

from pyscf.nao import mf
from pyscf.nao.m_chi0_noxv import chi0_mv_gpu, chi0_mv
from pyscf.data.nist import HARTREE2EV

class chi0_matvec(mf):
    """
    A class to organize the application of non-interacting response to a vector
    """

    def __init__(self, **kw):
        from pyscf.nao.m_fermi_dirac import fermi_dirac_occupations

        if "use_initial_guess_ite_solver" in kw:
            self.use_initial_guess_ite_solver = kw["use_initial_guess_ite_solver"]
        else:
            self.use_initial_guess_ite_solver = False

        krylov_solvers = {"lgmres": splin.lgmres,
                          "gmres": splin.gmres,
                          "gcrotmk": splin.gcrotmk,
                          "qmr": splin.qmr,
                          "minres": splin.minres,
                          "bicgstab": splin.bicgstab,
                          "bicg": splin.bicg,
                          "cg": splin.cg,
                          "cgs": splin.cgs}
        if "krylov_solver" in kw:
            self.krylov_solver = krylov_solvers[kw["krylov_solver"]]
        else:
            self.krylov_solver = krylov_solvers["lgmres"]

        if "krylov_options" in kw:
            self.krylov_options = kw["krylov_options"]
        else:
            self.krylov_options = {"tol": 1.0e-3,
                                   "atol": 1.0e-3,
                                   "maxiter": 1000}

        mf.__init__(self, **kw)

        if self.dtype == np.float32:
            self.gemm = blas.sgemm
            self.gemv = blas.sgemv
        elif self.dtype == np.float64:
            self.gemm = blas.dgemm
            self.gemv = blas.dgemv
        else:
            raise ValueError("dtype can be only float32 or float64")

        self.dealloc_hsx = kw['dealloc_hsx'] if 'dealloc_hsx' in kw else True
        self.eps = kw['iter_broadening'] if 'iter_broadening' in kw else 0.00367493
        self.GPU = kw['GPU'] if 'GPU' in kw else False
        self.nfermi_tol = nfermi_tol = kw['nfermi_tol'] if 'nfermi_tol' in kw else 1e-5
        self.telec = kw['telec'] if 'telec' in kw else self.telec
        self.fermi_energy = kw['fermi_energy'] if 'fermi_energy' in kw else self.fermi_energy

        self.chi0_timing = np.zeros((17), dtype=np.float64)

        assert isinstance(self.eps, float)

        if self.GPU:
            if not self.use_numba:
                raise ValueError("GPU calculations require Numba")

            if self.dtype == np.float32:
                from pyscf.nao.m_div_eigenenergy_numba_gpu import div_eigenenergy_gpu_float32
                self.div_numba = div_eigenenergy_gpu_float32

            else:
                from pyscf.nao.m_div_eigenenergy_numba_gpu import div_eigenenergy_gpu_float64
                self.div_numba = div_eigenenergy_gpu_float64
        elif self.use_numba:
            from pyscf.nao.m_div_eigenenergy_numba import div_eigenenergy_numba
            self.div_numba = div_eigenenergy_numba

        else:
            self.div_numba = None

        # deallocate hsx
        if hasattr(self, 'hsx') and self.dealloc_hsx: self.hsx.deallocate()

        self.ksn2e = self.mo_energy # Just a pointer here. Is it ok?
        
        if 'fermi_energy' in kw:
            if self.verbosity > 0:
                print(__name__, 'Fermi energy is specified => recompute occupations')

            ksn2fd = fermi_dirac_occupations(self.telec, self.ksn2e, self.fermi_energy)
            for s,n2fd in enumerate(ksn2fd[0]):
                if not all(n2fd>self.nfermi_tol): continue
                print(self.telec, s, self.nfermi_tol, n2fd)
                raise RuntimeError(__name__, 'telec is too high?')
            ksn2f = self.ksn2f = (3-self.nspin)*ksn2fd
        else:
            ksn2f = self.ksn2f = self.mo_occ
        
        self.nfermi = array([argmax(ksn2f[0,s]<self.nfermi_tol)\
                            for s in range(self.nspin)], dtype=int)
        self.vstart = array([argmax(1.0-ksn2f[0,s]>=self.nfermi_tol)\
                            for s in range(self.nspin)], dtype=int)
        
        # should not be this list arrays ??
        self.xocc = [self.mo_coeff[0,s,:nfermi,:,0]\
                             for s,nfermi in enumerate(self.nfermi)]
        self.xvrt = [self.mo_coeff[0,s,vstart:,:,0]\
                             for s,vstart in enumerate(self.vstart)]
     
        if self.verbosity > 4:
            print(__name__, '\t====> self.dtype ', self.dtype)
            print(__name__, '\t====> self.xocc[0].dtype ', self.xocc[0].dtype)
            print(__name__, '\t====> self.xvrt[0].dtype ', self.xvrt[0].dtype)
            print(__name__, '\t====> MO energies (ksn2e) (eV):\n{},\tType: {}'.format(self.ksn2e*HARTREE2EV,self.ksn2e.dtype))
            print(__name__, '\t====> Occupations (ksn2f):\n{},\tType: {}'.format(self.ksn2f,self.ksn2f.dtype))

        self.chi0mv_ncalls = 0
        self.chi0mv_ncalls_ite = 0
        if not hasattr(self, 'pb'):
            print('no pb?')
            return

        pb = self.pb
        self.moms0,self.moms1 = pb.comp_moments(dtype=self.dtype)

        if self.GPU:
            self.initialize_chi0_matvec_GPU()

    def initialize_chi0_matvec_GPU(self):
        """
        Initialize GPU variable.
        Copy xocc, xvrt, ksn2e and ksn2f to the device
        """

        try:
            import cupy as cp
        except RuntimeError as err:
            raise RuntimeError("Could not import cupy: {}".format(err))

        block_size = 32
        self.block_size = [] # np.array([32, 32], dtype=np.int32) # threads by block
        self.grid_size = [] #np.array([0, 0], dtype=np.int32) # number of blocks

        self.xocc_gpu = []
        self.xvrt_gpu = []
        self.ksn2e_gpu = cp.asarray(self.ksn2e)
        self.ksn2f_gpu = cp.asarray(self.ksn2f)

        for spin in range(self.nspin):
            dimensions = [self.nfermi[spin], self.nprod]
            self.block_size.append([block_size, block_size])
            self.grid_size.append([0, 0])
            for i in range(2):
                if dimensions[i] <= block_size:
                    self.block_size[spin][i] = dimensions[i]
                    self.grid_size[spin][i] = 1
                else:
                    self.grid_size[spin][i] = dimensions[i]//self.block_size[spin][i] + 1

            print("spin {}: block_size: ({}, {}); grid_size: ({}, {})".format(
                spin, self.block_size[spin][0], self.block_size[spin][1],
                self.grid_size[spin][0], self.grid_size[spin][1]))
            self.xocc_gpu.append(cp.asarray(self.xocc[spin]))
            self.xvrt_gpu.append(cp.asarray(self.xvrt[spin]))

    def apply_rf0(self, sp2v, comega=1j*0.0):
        """
        This applies the non-interacting response function to a vector
        (a set of vectors?)
        """
        
        expect_shape=tuple([self.nspin*self.nprod])
        assert np.all(sp2v.shape == expect_shape),\
                "{} {}".format(sp2v.shape,expect_shape)
        self.chi0mv_ncalls_ite += 1
        self.chi0mv_ncalls += 1

        if self.GPU:
            return chi0_mv_gpu(self, sp2v, comega, timing=self.chi0_timing)
        else:
            return chi0_mv(self, sp2v, comega, timing=self.chi0_timing)


    def comp_polariz_nonin_xx_atom_split(self, comegas):
        """
        Compute the non interacting polarizability along the xx direction
        """
        aw2pxx = np.zeros((self.natoms, comegas.shape[0]), dtype=self.dtypeComplex)

        vext = np.transpose(self.moms1)
        for iw, comega in enumerate(comegas):
            dn0 = self.apply_rf0(vext[0], comega)
            for ia in range(self.natoms):
                dn0_atom = np.zeros(self.nprod, dtype=self.dtypeComplex)
                st = self.pb.c2s[ia]
                fn = self.pb.c2s[ia+1]
                dn0_atom[st:fn] = dn0[st:fn]
                aw2pxx[ia, iw] = dn0_atom.dot(vext[0])
        
        self.write_chi0_mv_timing("tddft_iter_polariz_nonin_split_chi0_mv.txt")
        return aw2pxx

    def write_chi0_mv_timing(self, fname):

        with open(fname, "w") as fl:
            fl.write("# Total number call chi0_mv: {}\n".format(self.chi0mv_ncalls))
            fl.write("# step  time [s]\n")
            for it, time in enumerate(self.chi0_timing):
                fl.write("{}: {}\n".format(it, time))

    def comp_dens_along_Eext(self, comegas, Eext=np.array([1.0, 0.0, 0.0]),
                             tmp_fname=None, inter=False):
        """ 
        Compute a the average interacting polarizability along the Eext direction
        for the frequencies comegas.
        
        Input Parameters:
            comegas (1D array, complex): the real part contains the frequencies at which the polarizability
                        should be computed. The imaginary part id the width of the polarizability define as self.eps
            Eext (1D xyz array, real): direction of the external field
            maxiter (integer): max number of iteration before to exit iteration loop in GMRES
        
        Other Calculated quantity:
            self.p_mat (complex array, dim: [3, 3, comega.size]): store the (3, 3) polarizability matrix 
                                [[Pxx, Pxy, Pxz],
                                 [Pyx, Pyy, Pyz],
                                 [Pzx, Pzy, Pzz]] for each frequency.
            self.dn (complex array, dim: [3, comegas.size, self.nprod]): store the density change
        """

        if tmp_fname is not None:
            if not isinstance(tmp_fname, str):
                raise ValueError("tmp_fname must be a string")
            else:
                tmp_re = open(tmp_fname+".real", "w")
                tmp_re.write("# All atomic units\n")
                tmp_re.write("# w (Ha)    Pxx    Pxy    Pxz    Pyx    Pyy    Pyz    Pzx    Pzy    Pzz\n")
            
                tmp_im = open(tmp_fname+".imag", "w")
                tmp_im.write("# All atomic units\n")
                tmp_im.write("# w    Pxx    Pxy    Pxz    Pyx    Pyy    Pyz    Pzx    Pzy    Pzz\n")

        if isinstance(Eext, list):
            Eext = np.array(Eext.size)

        assert Eext.size == 3
        
        nww = len(comegas)
        p_mat = np.zeros((3, 3, nww), dtype=self.dtypeComplex)
        dn = np.zeros((3, nww, self.nspin*self.nprod), dtype=self.dtypeComplex)
        Edir = Eext/np.dot(Eext, Eext)
    
        vext = np.zeros((self.nspin*self.nprod, 3), dtype=self.moms1.dtype)
        #veff = np.zeros((self.nspin*self.nprod, 3), dtype=self.dtypeComplex)
        for ixyz in range(3):
            vext[:, ixyz] = np.concatenate([self.moms1[:, ixyz] for s in range(self.nspin)])

        for iw, comega in enumerate(comegas):

            dn[:, iw, :], p_mat[:, :, iw] = \
                    self.calc_dens_Edir_omega(iw, nww, comega, Edir, vext,
                                              tmp_fname=tmp_fname, inter=inter)

        if inter:
            fname = "tddft_iter_dens_chng_inter_chi0_mv.txt"
        else:
            fname = "tddft_iter_dens_chng_nonin_chi0_mv.txt"
        self.write_chi0_mv_timing(fname)
        print("Total number of iterations: ", self.chi0mv_ncalls)
        return dn, p_mat
        
    def calc_dens_Edir_omega(self, iw, nww, w, Edir, vext, tmp_fname=None,
                             inter=False):
        """
        Calculate the density change and polarizability for a specific frequency
        """

        Pmat = np.zeros((3, 3), dtype=self.dtypeComplex)
        dn = np.zeros((3, self.nspin*self.nprod), dtype=self.dtypeComplex)
        eV = 27.211386024367243

        for xyz, Exyz in enumerate(Edir):
            if abs(Exyz) < 1.0e-12:
                continue

            self.chi0mv_ncalls_ite = 0
            if inter:
                veff = self.comp_veff(vext[:, xyz], w)#, x0=veff[:, xyz])
                dn[xyz, :] = self.apply_rf0(veff, w)
            else:
                dn[xyz, :] = self.apply_rf0(vext[:, xyz], w)

            if self.verbosity > 1:
                print("dir: {0}, iw: {1}/{2}; w: {3:.4f}; nite: {4}".format(xyz,
                                                                            iw,
                                                                            nww,
                                                                            w.real*eV,
                                                                            self.chi0mv_ncalls_ite))

            for xyzp in range(Edir.size):
                Pmat[xyz, xyzp] = vext[:, xyzp].dot(dn[xyz, :])

        if tmp_fname is not None:
            tmp_re = open(tmp_fname + ".real", "a")
            tmp_re.write("{0:.6e}   ".format(w.real))

            tmp_im = open(tmp_fname + ".imag", "a")
            tmp_im.write("{0:.6e}   ".format(w.real))
        
            for i in range(3):
                for j in range(3):
                    tmp_re.write("{0:.6e}    ".format(Pmat[i, j].real))
                    tmp_im.write("{0:.6e}    ".format(Pmat[i, j].imag))
            tmp_re.write("\n")
            tmp_im.write("\n")
            tmp_re.close()
            tmp_im.close()
            # Need to open and close the file at every freq, otherwise
            # tmp is written only at the end of the calculations, therefore,
            # it is useless

        return dn, Pmat
