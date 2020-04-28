from __future__ import print_function, division
import sys, copy
from timeit import default_timer as timer
import time
import numpy as np
from pyscf.nao import scf, gw
from pyscf.data.nist import HARTREE2EV
from pyscf.nao.m_gw_chi0_noxv import gw_chi0_mv, gw_chi0_mv_gpu
from pyscf.nao.m_sparsetools import spmat_denmat

def profile(fnc):
    """
    Profiles any function in following class just by adding @profile above function
    """
    import cProfile, pstats, io
    def inner (*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc (*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'   #Ordered
        ps = pstats.Stats(pr,stream=s).strip_dirs().sort_stats(sortby)
        n=20                    #reduced the list to be monitored
        ps.print_stats(n)
        #ps.dump_stats("profile.prof")
        print(s.getvalue())
        return retval
    return inner


class gw_iter(gw):
  """
  Iterative G0W0 with integration along imaginary axis
  """

  def __init__(self, **kw):
    gw.__init__(self, **kw)

    self.gw_iter_tol = kw['gw_iter_tol'] if 'gw_iter_tol' in kw else 1e-4
    self.maxiter = kw['maxiter'] if 'maxiter' in kw else 1000
    self.gw_xvx_algo = kw['gw_xvx_algo'] if 'gw_xvx_algo' in kw else "ac_blas"

    self.limited_nbnd = kw['limited_nbnd'] if 'limited_nbnd' in kw else False
    if (self.limited_nbnd and min (self.vst) < 50 ):
        print('Too few virtual states, limited_nbnd ignored!')
        self.limited_nbnd= False

    self.pass_dupl = kw['pass_dupl'] if 'pass_dupl' in kw else False

    #if not hasattr(self, 'h0_vh_x_expval'):
    #    self.h0_vh_x_expval = self.get_h0_vh_x_expval()
    #    if self.write_R:    
    #        self.write_data(step = 'H0EXP')

    self.ncall_chi0_mv_ite = 0
    self.ncall_chi0_mv_total = 0
    self.lgmres_time_per_step = []

    # Store the ac product basis if necessary (ac_sparse)
    self.v_pab = None

  def si_c_iter(self,ww):
    """
    This computes the correlation part of the screened interaction using LinearOpt and lgmres
    lgmres method is much slower than np.linalg.solve !!
    """
    from scipy.sparse.linalg import lgmres, LinearOperator 
    si0 = np.zeros((ww.size, self.nprod, self.nprod), dtype=self.dtypeComplex)
    k_c_opt = LinearOperator((self.nprod,self.nprod), matvec=self.gw_vext2veffmatvec, dtype=self.dtypeComplex) 
    for iw,w in enumerate(ww):                                 
      self.comega_current = w 
       
      for m in range(self.nprod):  
        k_c = self.kernel_sq[m] 
        b = self.chi0_mv(k_c, self.comega_current) 
        a = self.kernel_sq.dot(b) 
        si0[iw,m,:],exitCode = lgmres(k_c_opt, a, atol=self.gw_iter_tol, maxiter=self.maxiter)    
        if exitCode != 0: print("LGMRES has not achieved convergence: exitCode = {}".format(exitCode))
    return si0

  def si_c_check (self, tol = 1e-5):
    """
    This compares np.solve and LinearOpt-lgmres methods for solving linear equation (1-v\chi_{0}) * W_c = v\chi_{0}v
    """
    import time
    import numpy as np
    ww = 1j*self.ww_ia
    t = time.time()
    si0_1 = self.si_c(ww)      #method 1:  numpy.linalg.solve
    t1 = time.time() - t
    print('numpy: {} sec'.format(t1))
    t2 = time.time()
    si0_2 = self.si_c2(ww)     #method 2:  scipy.sparse.linalg.lgmres
    t3 = time.time() - t2
    print('lgmres: {} sec'.format(t3))
    summ = abs(si0_1 + si0_2).sum()
    diff = abs(si0_1 - si0_2).sum() 
    if diff/summ < tol and diff/si0_1.size < tol:
       print('OK! scipy.lgmres methods and np.linalg.solve have identical results')
    else:
       print('Results (W_c) are NOT similar!')     
    return [[diff/summ] , [np.amax(abs(diff))] ,[tol]]

  def gw_xvx(self, algo=None):
    """
     calculates basis products
     \Psi(r')\Psi(r') = XVX[spin,(nn, norbs, nprod)] = X_{a}^{n}V_{\nu}^{ab}X_{b}^{m}
     
     using 4-methods:
        1- direct multiplication by using np.dot and np.einsum via swapping between axis
        2- using atom-centered product basis
        3- using atom-centered product basis and BLAS multiplication
        4- using sparse atom-centered product basis and BLAS multiplication
        5- using dominant product basis
        6- using sparse dominant product basis
    """

    algol = algo.lower() if algo is not None else 'ac_blas'  
    print("gw_xvx algo: ", algol)

    # 1-direct multiplication with np and einsum
    if algol=='simple':

        from pyscf.nao.m_gw_xvx import gw_xvx_ac_simple
        xvx = gw_xvx_ac_simple(self)

    # 2-atom-centered product basis
    elif algol=='ac':

        from pyscf.nao.m_gw_xvx import gw_xvx_ac
        xvx = gw_xvx_ac(self)

    # 3-atom-centered product basis and BLAS
    elif algol=='ac_blas':
        
        from pyscf.nao.m_gw_xvx import gw_xvx_ac_blas
        xvx = gw_xvx_ac_blas(self)

    # 4-sparse-atom-centered product basis and BLAS
    elif algol=='ac_sparse':

        from pyscf.nao.m_gw_xvx import gw_xvx_ac_sparse
        xvx = gw_xvx_ac_sparse(self)
        
    # 5-dominant product basis
    elif algol=='dp':

        from pyscf.nao.m_gw_xvx import gw_xvx_dp
        xvx = gw_xvx_dp(self)

    # 6-dominant product basis with scipy sparse COO format
    elif algol=='dp_coo':

        from pyscf.nao.m_gw_xvx import gw_xvx_dp_ndcoo
        xvx = gw_xvx_dp_ndcoo(self)

    # 7-using sparcity of dominant product basis and Numba library
    elif algol=='dp_sparse':

        from pyscf.nao.m_gw_xvx import gw_xvx_dp_sparse
        xvx = gw_xvx_dp_sparse(self)

    else:
      raise ValueError("Unknow algo {}".format(algol))

    return xvx

  def dupl_E (self, spin, thrs=None):
    """
    This returns index of states whose eigenenergy has difference less than thrs with former state
    """
    thrs=1e-05 if thrs is None else thrs #it's in Hartree, self.tol_ev is too stringent!!
    dup=[]
    #diff between mf's eigenvalues
    del_E = np.diff(self.ksn2e[0,spin])
    #print('number of values with difference less than {}: {}'.format(thrs, sum(i < thrs for i in del_E))

    #index of elements less than thrs   
    for x, val in enumerate(del_E):
      if (val < thrs):
        dup.append(x+1)
    return dup

  def precond_lgmres (self, omega=None):
    """
    For a simple V_eff, this calculates linear operator of M = (1-chi_0(omega) = kc_opt)^-1, 
    as a preconditioner for lgmres
    """
    from scipy.sparse import csc_matrix
    import scipy.sparse.linalg as spla 

    omega = 1j*10.0 if omega is None else omega
    v = np.ones(self.nprod, dtype=self.dtypeComplex)                           
    vX = self.chi0_mv (v, comega=omega)
    
    vX2 = 1 - vX.real  
    l = np.zeros((self.nprod,self.nprod), dtype=self.dtype) 
    np.fill_diagonal(l, vX2)

    #from numpy.linalg import pinv
    #M=pinv(l)
    l = csc_matrix (l)
    M_x = lambda x: spla.spsolve(l, x) 
    M = spla.LinearOperator((self.nprod, self.nprod), M_x)
    return M
    

  def get_snmw2sf_iter(self, nbnd=None, optimize="greedy"):
    """ 
    This computes a matrix elements of W_c:
        <\Psi(r)\Psi(r) | W_c(r,r',\omega) |\Psi(r')\Psi(r')>.
        sf[spin,n,m,w] = X^n V_mu X^m W_mu_nu X^n V_nu X^m,
    
    where   n runs from s...f,
            m runs from 0...norbs,
            w runs from 0...nff_ia,
            spin=0...1 or 2.
    1- XVX is calculated using dominant product: gw_xvx('dp_coo')
    2- I_nm = W XVX = (1-v\chi_0)^{-1}v\chi_0v
    3- S_nm = XVX W XVX = XVX * I_nm
    """


    from scipy.sparse.linalg import LinearOperator, lgmres

    if self.GPU:
        self.initialize_chi0_matvec_GPU()
    
    self.time_gw[10] = timer();    
    ww = 1j*self.ww_ia

    if not hasattr(self, 'xvx'):
        self.xvx = self.gw_xvx(self.gw_xvx_algo)

    snm2i = []
    # convert k_c as full matrix into Operator
    k_c_opt = LinearOperator((self.nprod,self.nprod),
                             matvec=self.gw_vext2veffmatvec,
                             dtype=self.dtypeComplex)

    # preconditioning could be using 1- kernel
    # not sure ...
    x0 = None
    M0 = None #self.precond_lgmres ()
    for s in range(self.nspin):
        sf_aux = np.zeros((len(self.nn[s]), self.norbs, self.nprod), dtype=self.dtypeComplex)
        inm = np.zeros((len(self.nn[s]), self.norbs, len(ww)), dtype=self.dtypeComplex)
        dup = self.dupl_E(s)
        if self.pass_dupl : print('duplicate states:', dup)
        # w is complex plane
        for iw, w in enumerate(ww):
            self.comega_current = w
            #M0 = self.precond_lgmres (w)

            self.ncall_chi0_mv_ite = 0
            if self.verbosity>3:
                print("spin: {}; freq: {}; nn = {}; norbs = {}".format(s+1, iw, len(self.nn[s]),
                                                             self.norbs))

            t1 = timer()
            for n in range(len(self.nn[s])):    
                for m in range(self.norbs):
                    
                    if (self.pass_dupl and m in dup ):
                        #copies sf for m whose mf's energy is almost similar, how about n??
                        sf_aux[n,m,:]= copy.deepcopy(sf_aux[n,m-1,:])
                        if self.verbosity>3: print('m#{} copied_duplicate, pass_dupl'.format(m))

                    else:
                        # v XVX
                        a = self.kernel_sq.dot(self.xvx[s][n,m,:])
                        
                        # \chi_{0}v XVX by using matrix vector
                        tt1 = timer()
                        b = self.chi0_mv(a, self.comega_current)
                        tt2 = timer()               
                        self.time_gw[15] += tt2 - tt1

                        # v\chi_{0}v XVX, this should be equals to bxvx in last approach
                        a = self.kernel_sq.dot(b)

                        #considers only part v\chi_{0}v XVX for virtual states above nbnd
                        if ( self.limited_nbnd and m >= nbnd[s]):
                            sf_aux[n,m,:] = a
                            if self.verbosity > 3:
                                print('m#{} skiped LGMRS, limited_nbnd'.format(m))

                        else:

                            # initial guess works pretty well!!
                            if self.use_initial_guess_ite_solver:
                                if iw == 0:
                                    x0 = None
                                else:
                                    x0 = copy.deepcopy(prev_sol[n, m, :])
                            sf_aux[n,m,:], exitCode = lgmres(k_c_opt, a,
                                                             atol=self.gw_iter_tol,
                                                             maxiter=self.maxiter,
                                                             x0=x0, M=M0)
                            if exitCode != 0:
                              print("LGMRES has not achieved convergence: exitCode = {}".format(exitCode))
                

            t2 = timer()
            if self.use_initial_guess_ite_solver:
                prev_sol = copy.deepcopy(sf_aux)

            if self.verbosity>3:
                print("time for lgmres loop: ", round(t2-t1,2))
                self.lgmres_time_per_step.append(round(t2-t1,2))
                print("number call chi0_mv:  ", self.ncall_chi0_mv_ite)
                print("Average call chi0_mv: ", int(self.ncall_chi0_mv_ite/(len(self.nn[s])*self.norbs)))

            self.ncall_chi0_mv_total += self.ncall_chi0_mv_ite

            # I = XVX I_aux
            inm[:,:,iw] = np.einsum('nmp,nmp->nm', self.xvx[s], sf_aux, optimize=optimize)
        snm2i.append(np.real(inm))

    print("Total call chi0_mv: ", self.ncall_chi0_mv_total)
    self.time_gw[11] = timer();

    return snm2i


  def gw_vext2veffmatvec(self, vin):
    t1 = timer()
    dn0 = self.chi0_mv(vin, self.comega_current)
    t2 = timer()               
    self.time_gw[23] += t2 - t1
    vcre, vcim = self.gw_applykernel_nspin1(dn0)
    return vin - (vcre + 1.0j*vcim)         #1 - v\chi_0

  def gw_vext2veffmatvec2(self,vin):
    dn0 = self.chi0_mv(vin, self.comega_current)
    vcre,vcim = self.gw_applykernel_nspin1(dn0)
    return 1- (vin - (vcre + 1.0j*vcim))    #1- (1-v\chi_0)

  def chi0_mv(self, dvin, comega):

      self.ncall_chi0_mv_ite += 1
      
      if self.GPU:
          return gw_chi0_mv_gpu(self, dvin, comega=comega, timing=self.chi0_timing)
      else:
          return gw_chi0_mv(self, dvin, comega=comega, timing=self.chi0_timing)

  def gw_applykernel_nspin1(self,dn):
    daux  = np.zeros(self.nprod, dtype=self.dtype)
    daux[:] = np.require(dn.real, dtype=self.dtype, requirements=["A","O"])
    vcre = self.spmv(self.nprod, 1.0, self.kernel, daux)
    
    daux[:] = np.require(dn.imag, dtype=self.dtype, requirements=["A","O"])
    vcim = self.spmv(self.nprod, 1.0, self.kernel, daux)
    return vcre,vcim

  def gw_comp_veff(self, vext, comega=1j*0.0):
    """
    This computes an effective field (scalar potential) given the external
    scalar potential as follows:
        (1-v\chi_{0})V_{eff} = V_{ext} = X_{a}^{n}V_{\mu}^{ab}X_{b}^{m} * 
                                         v\chi_{0}v * X_{a}^{n}V_{nu}^{ab}X_{b}^{m}
    
    returns V_{eff} as list for all n states(self.nn[s]).
    """
    
    from scipy.sparse.linalg import LinearOperator

    if self.GPU:
        self.initialize_chi0_matvec_GPU()

    self.comega_current = comega
    veff_op = LinearOperator((self.nprod,self.nprod),
                             matvec=self.gw_vext2veffmatvec,
                             dtype=self.dtypeComplex)

    from scipy.sparse.linalg import lgmres
    resgm, info = lgmres(veff_op,
                         np.require(vext, dtype=self.dtypeComplex, requirements='C'),
                         atol=self.gw_iter_tol, maxiter=self.maxiter)
    if info != 0:
      print("LGMRES has not achieved convergence: exitCode = {}".format(info))
    return resgm

  def check_veff(self, optimize="greedy"):
    """
    This checks the equality of effective field (scalar potential) given the external
    scalar potential obtained from lgmres(linearopt, v_ext) and np.solve(dense matrix, vext). 
    """

    from numpy.linalg import solve

    ww = 1j*self.ww_ia
    rf0 = self.rf0(ww)
    #V_{\mu}^{ab}
    if self.v_pab is None:
        v_pab = self.pb.get_ac_vertex_array(matformat="dense",
                                            dtype=self.dtype)
    else:
        v_pab = self.v_pab

    for s in range(self.nspin):
      v_eff = np.zeros((len(self.nn[s]), self.nprod), dtype=self.dtype)
      v_eff_ref = np.zeros((len(self.nn[s]), self.nprod), dtype=self.dtype)
      # X_{a}^{n}
      xna = self.mo_coeff[0,s,self.nn[s],:,0]
      # X_{b}^{m}
      xmb = self.mo_coeff[0,s,:,:,0]
      # X_{a}^{n}V_{\mu}^{ab}X_{b}^{m}
      xvx = np.einsum('na,pab,mb->nmp', xna, v_pab, xmb, optimize=optimize)
      for iw,w in enumerate(ww):     
          # v\chi_{0} 
          k_c = self.kernel_sq.dot(rf0[iw,:,:])
          # v\chi_{0}v 
          b = k_c.dot(self.kernel_sq)
          #(1-v\chi_{0})
          k_c = np.eye(self.nprod)-k_c
          
          #v\chi_{0}v * X_{a}^{n}V_{\nu}^{ab}X_{b}^{m}
          bxvx = np.einsum('pq,nmq->nmp', b, xvx, optimize=optimize)
          #V_{ext}=X_{a}^{n}V_{\mu}^{ab}X_{b}^{m} * v\chi_{0}v * X_{a}^{n}V_{\nu}^{ab}X_{b}^{m}
          xvxbxvx = np.einsum ('nmp,nlp->np', xvx, bxvx, optimize=optimize)
          
          for n in range (len(self.nn[s])):
              # compute v_eff in tddft_iter class as referance
              v_eff_ref[n,:] = self.gw_comp_veff(xvxbxvx[n,:])
              # linear eq. for finding V_{eff} --> (1-v\chi_{0})V_{eff}=V_{ext}
              v_eff[n,:]=solve(k_c, xvxbxvx[n,:])

    # compares both V_{eff}
    if np.allclose(v_eff,v_eff_ref,atol=1e-4)== True:
      return v_eff

  def gw_corr_int_iter(self, sn2w, eps=None):
    """
    This computes an integral part of the GW correction at GW class while
    uses get_snmw2sf_iter
    """

    if self.GPU:
        self.initialize_chi0_matvec_GPU()

    if not hasattr(self, 'snmw2sf'):

        if self.restart: 
            from pyscf.nao.m_restart import read_rst_h5py
            self.snmw2sf, msg = read_rst_h5py(value='screened_interactions',
                                              filename= 'RESTART.hdf5')
            print(msg)  

        else:
            if self.limited_nbnd:
                nbnd = [int(self.nfermi[s] + self.vst[s]*0.7) for s in range(self.nspin)] #considers 70% of virtual states
                self.snmw2sf = self.get_snmw2sf_iter(nbnd)
                print('Limited number of virtual states are considered in full matrix of W_c')

            else:
                self.snmw2sf = self.get_snmw2sf_iter()
        
    if self.write_R: self.write_data(step = 'W_c')

    return self.gw_corr_int(sn2w, eps=None)

  def gw_corr_res_iter(self, sn2w):
    """
    This computes a residue part of the GW correction at energies in 
    iterative procedure
    """
    
    from scipy.sparse.linalg import lgmres, LinearOperator

    if not hasattr(self, 'xvx'): self.xvx = self.gw_xvx(self.gw_xvx_algo)


    t1 = timer()

    sn2res = [np.zeros_like(n2w, dtype=self.dtype) for n2w in sn2w ]   
    k_c_opt = LinearOperator((self.nprod,self.nprod),
                             matvec=self.gw_vext2veffmatvec,
                             dtype=self.dtypeComplex)

    prev_sol = None
    M0 = self.precond_lgmres ()
    for spin, ww in enumerate(sn2w):
      
        #x = self.mo_coeff[0, spin, :, :, 0]
        for nl, (n, w) in enumerate(zip(self.nn[spin], ww)):
            lsos = self.lsofs_inside_contour(self.ksn2e[0, spin ,:], w, self.dw_excl)
            zww = np.array([pole[0] for pole in lsos])
            if self.verbosity > 3:
                stw = np.array([pole[1] for pole in lsos])
                print('spin {}, states located inside contour: {}'.format(spin ,str(stw)))
            #xv = v_pab.dot(x[n])

            for pole, z_real in zip(lsos, zww):
                self.comega_current = z_real
                #M0 = self.precond_lgmres (z_real)
                #xvx = xv.dot(x[pole[1]])
                a = self.kernel_sq.dot(self.xvx[spin][nl, pole[1], :])
                tt1 = timer()
                b = self.chi0_mv(a, self.comega_current)
                tt2 = timer()               
                self.time_gw[21] += tt2 - tt1
                a = self.kernel_sq.dot(b)

                si_xvx, exitCode = lgmres(k_c_opt, a, atol=self.gw_iter_tol,
                                          maxiter=self.maxiter)#, x0 = prev_sol, M=M0)
                if exitCode != 0:
                    print("LGMRES has not achieved convergence: exitCode = {}".format(exitCode))

                if self.use_initial_guess_ite_solver:
                    #if nl > 0: print('prev_sol ,si_xvx', (abs(prev_sol - si_xvx).sum()))
                    prev_sol = si_xvx

                contr = self.xvx[spin][nl, pole[1], :].dot(si_xvx)
                sn2res[spin][nl] += pole[2]*contr.real

    t2 = timer()
    self.time_gw[19] += t2 - t1 
    return sn2res

  def g0w0_eigvals_iter(self):
    """
    This computes the G0W0 corrections to the eigenvalues
    """
    self.time_gw[8] = timer(); 
    #self.ksn2e = self.mo_energy
    sn2eval_gw = [np.copy(self.ksn2e[0,s,nn]) for s,nn in enumerate(self.nn) ]
    sn2eval_gw_prev = copy.copy(sn2eval_gw)

    self.nn_conv = []           # self.nn_conv - list of states to converge
    for nocc_0t,nocc_conv,nvrt_conv in zip(self.nocc_0t, self.nocc_conv, self.nvrt_conv):
      self.nn_conv.append( range(max(nocc_0t-nocc_conv,0), min(nocc_0t+nvrt_conv,self.norbs)))

    # iterations to converge the qp-energies 
    if self.verbosity>0:
      print('='*48,'| G0W0_iter corrections of eigenvalues |','='*48)
      mess = """        
      MAXIMUM number of iterations (Input file): {},
      Number of grid points (frequency): {},
      Number of states to be corrected: {},
      Tolerance to convergence: {}\n
      GW corection for eigenvalues STARTED:
      """.format(self.niter_max_ev, self.nff_ia, len(self.nn[0]), self.tol_ev)
      print(mess)  
  
    perv1 = [np.zeros(len(self.nn[s])) for s in range(self.nspin)]
    perv2 = [np.zeros(len(self.nn[s])) for s in range(self.nspin)]
    sn2i_conv = False
    sn2r_conv = False
    #perv3 = np.array([])

    for i in range(self.niter_max_ev):

      if sn2i_conv:   sn2i = perv1
      else:           sn2i = self.gw_corr_int_iter(sn2eval_gw)
      if sn2r_conv:   sn2r = perv2
      else:           sn2r = self.gw_corr_res_iter(sn2eval_gw)

      if (all([np.allclose(x, y, atol = self.tol_ev) for x, y in zip(sn2i, perv1)])): 
        sn2i_conv = True      
        if (self.verbosity > 3): print('Integral part converged!') 
 
      if (all([np.allclose(x, y, atol = self.tol_ev) for x, y in zip(sn2r, perv2)])): 
        sn2r_conv = True
        if (self.verbosity > 3): print('Residual part converged!')

      #states = self.gw_corr_res_states(sn2eval_gw)
      #if (self.verbosity > 3 and np.array_equal(states, perv3)): 
      #  print('States inside contour are identical')
    
      perv1 = sn2i
      perv2 = sn2r
      #perv3 = states

      sn2eval_gw = []
      for s,(evhf,n2i,n2r,nn) in enumerate(zip(self.h0_vh_x_expval,sn2i,sn2r,self.nn)):
        sn2eval_gw.append(evhf[nn]+n2i+n2r)
        
      sn2mismatch = np.zeros((self.nspin,self.norbs))
      for s, nn in enumerate(self.nn): sn2mismatch[s,nn] = sn2eval_gw[s][:]-sn2eval_gw_prev[s][:]
      sn2eval_gw_prev = copy.copy(sn2eval_gw)
      err = 0.0
      for s,nn_conv in enumerate(self.nn_conv): err += abs(sn2mismatch[s,nn_conv]).sum()/len(nn_conv)

      if self.verbosity>0:
        np.set_printoptions(linewidth=1000, suppress=True, precision=5)
        print('Iteration #{:3d}, Relative Error: {:.8f}, Time spent up to now: {:.1f} secs'
                .format(i+1, err, timer()-self.time_gw[0]))
      if self.verbosity>1:
        #print(sn2mismatch)
        for s,n2ev in enumerate(sn2eval_gw):
          print('Spin{}: {}'.format(s+1, n2ev[:]*HARTREE2EV)) 
      
      if err<self.tol_ev : 
        if self.verbosity>0:
          print('-'*42,
                ' |  Convergence has been reached at iteration#{}  | '.format(i+1),
                '-'*42,'\n')
        break

      if err>=self.tol_ev and i+1==self.niter_max_ev:
        if self.verbosity>0:
          print('='*28,
                ' |  TAKE CARE! Convergence to tolerance {} not achieved after {}-iterations  | '.format(self.tol_ev,self.niter_max_ev),
                '='*28,'\n')

    self.time_gw[9] = timer();    
    return sn2eval_gw

  #@profile  
  def make_mo_g0w0_iter(self):
    """
    This creates the fields mo_energy_g0w0, and mo_coeff_g0w0
    """

    if not hasattr(self, 'h0_vh_x_expval'):
        if self.restart: 
            from pyscf.nao.m_restart import read_rst_h5py
            self.kmat , msg= read_rst_h5py(value='K_matrix', filename= 'RESTART.hdf5', arr=True)
            self.jmat , msg= read_rst_h5py(value='J_matrix', filename= 'RESTART.hdf5', arr=True)
            self.h0_vh_x_expval , msg= read_rst_h5py(value='H0_EXP', filename= 'RESTART.hdf5', arr=True)
            msg = 'RESTART: self.kmat, self.kmat and and self.h0_vh_x_expval read from RESTART.hdf5'
            print(msg)
        else:
            self.h0_vh_x_expval = self.get_h0_vh_x_expval()
            if self.write_R:
                self.write_data(step = 'H0EXP')

    if self.verbosity>2: self.report_mf()

    self.time_gw[12] = timer();
    if not hasattr(self, 'xvx'):
        if self.restart: 
            from pyscf.nao.m_restart import read_rst_h5py 
            self.xvx , msg= read_rst_h5py(value='XVX', filename= 'RESTART.hdf5')
            print(msg)
        else:
            self.xvx = self.gw_xvx(self.gw_xvx_algo)
    self.time_gw[13] = timer();

     
    if not hasattr(self,'sn2eval_gw'): 
        self.sn2eval_gw=self.g0w0_eigvals_iter() # Comp. GW-corrections


    # Update mo_energy_gw, mo_coeff_gw after the computation is done
    self.mo_energy_gw = np.copy(self.mo_energy)
    self.mo_coeff_gw = np.copy(self.mo_coeff)
    self.argsort = []

    self.time_gw[24] = timer();
    for s,nn in enumerate(self.nn):
      
      self.mo_energy_gw[0,s,nn] = self.sn2eval_gw[s]
      nn_occ = [n for n in nn if n<self.nocc_0t[s]]
      nn_vrt = [n for n in nn if n>=self.nocc_0t[s]]
      scissor_occ = (self.mo_energy_gw[0,s,nn_occ] - self.mo_energy[0,s,nn_occ]).sum()/len(nn_occ)
      scissor_vrt = (self.mo_energy_gw[0,s,nn_vrt] - self.mo_energy[0,s,nn_vrt]).sum()/len(nn_vrt)
      #print(scissor_occ, scissor_vrt)
      mm_occ = list(set(range(self.nocc_0t[s]))-set(nn_occ))
      mm_vrt = list(set(range(self.nocc_0t[s],self.norbs)) - set(nn_vrt))
      self.mo_energy_gw[0,s,mm_occ] +=scissor_occ
      self.mo_energy_gw[0,s,mm_vrt] +=scissor_vrt
      #print(self.mo_energy_g0w0)
      argsrt = np.argsort(self.mo_energy_gw[0,s,:])
      self.argsort.append(argsrt)

      if self.verbosity>2: 
        order = self.argsort[s][self.start_st[s]:self.finish_st[s]]        
        print(__name__, '\t\t====> Spin {}: energy-sorted MO indices: {}'.format(str(s+1),order))
      
      self.mo_energy_gw[0,s,:] = np.sort(self.mo_energy_gw[0,s,:])
      for n,m in enumerate(argsrt): self.mo_coeff_gw[0,s,n] = self.mo_coeff[0,s,m]
 
    self.time_gw[25] = timer();
    self.xc_code = 'GW'
    if self.verbosity>4:
      print(__name__,'\t\t====> Performed xc_code: {}\n '.format(self.xc_code))
      print('\nConverged GW-corrected eigenvalues (Ha):\n',
        [self.mo_energy_gw[0,s][self.start_st[s]:self.finish_st[s]] for s in range(self.nspin)])

    if self.write_R:    self.write_data(step='G0W0')
    self.write_chi0_mv_timing("gw_iter_chi0_mv.txt")

    return self.etot_gw()
        
  # This line is odd !!!
  kernel_gw_iter = make_mo_g0w0_iter

if __name__=='__main__':
    from pyscf import gto, scf
    from pyscf.nao import gw_iter   

    mol = gto.M(atom='''O 0.0, 0.0, 0.622978 ; O 0.0, 0.0, -0.622978''', basis='ccpvtz',spin=2)
    mf = scf.UHF(mol)
    mf.kernel()

    gw = gw_iter(mf=mf, gto=mol, verbosity=1, niter_max_ev=1, nff_ia=5, nvrt=1, nocc=1, 
                 use_initial_guess_ite_solver=False, 
                 limited_nbnd=False, pass_dupl=False, write_R = False)

    gw_ref = gw.get_snmw2sf()
    gw_it = gw.get_snmw2sf_iter()
    #if limited_nbnd: 
    #nbnd = [int(gw.nfermi[s] + gw.vst[s]*0.7) for s in range(gw.nspin)]
    #gw_it = gw.get_snmw2sf_iter(nbnd)

    print('Comparison between matrix element of W obtained from gw_iter and gw classes: ', 
            np.allclose(gw_it, gw_ref, atol= gw.gw_iter_tol)) 
    print([abs(gw_it[s]-gw_ref[s]).sum() for s in range(gw.nspin)])  

    sn2w = [np.copy(gw.ksn2e[0,s,nn]) for s,nn in enumerate(gw.nn)]
    t1 = timer()
    sn2r_it  = gw.gw_corr_res_iter(sn2w)
    t2 = timer()
    sn2r_ref = gw.gw_corr_res(sn2w)
    t3 = timer()
    print('Comparison between energies in residue part obtained from gw_iter and gw classes: ',
            np.allclose(sn2r_it, sn2r_ref, atol= gw.gw_iter_tol))
    print([abs(sn2r_it[s]-sn2r_ref[s]).sum() for s in range(gw.nspin)])
    print('iter Vs. ref', t2-t1, t3-t2)
    #gw.kernel_gw_iter()
    #gw.report()
