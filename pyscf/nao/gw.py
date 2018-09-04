from __future__ import print_function, division
import sys, numpy as np
from numpy import stack, dot, zeros, einsum, pi, log, array, require
from pyscf.nao import scf
from copy import copy
from pyscf.nao.m_pack2den import pack2den_u, pack2den_l
from timeit import default_timer as timer
from pyscf.nao.m_rf0_den import rf0_den, rf0_cmplx_ref_blk, rf0_cmplx_ref, rf0_cmplx_vertex_dp, rf0_cmplx_vertex_ac
from pyscf.nao.m_rf_den import rf_den
from pyscf.nao.m_rf_den_pyscf import rf_den_pyscf
from pyscf.data.nist import HARTREE2EV

starting_time=timer()
class gw(scf):
  """ G0W0 with integration along imaginary axis """
  
  def __init__(self, **kw):
    from pyscf.nao.m_log_mesh import log_mesh
    """ Constructor G0W0 class """
    # how to exclude from the input the dtype and xc_code ?
    scf.__init__(self, **kw)
    #print(__name__, ' dtype ', self.dtype)

    self.xc_code_scf = copy(self.xc_code)
    self.niter_max_ev = kw['niter_max_ev'] if 'niter_max_ev' in kw else 15
    self.tol_ev = kw['tol_ev'] if 'tol_ev' in kw else 1e-6
    self.perform_gw = kw['perform_gw'] if 'perform_gw' in kw else False
    self.rescf = kw['rescf'] if 'rescf' in kw else False
    self.bsize = kw['bsize'] if 'bsize' in kw else min(40, self.norbs)
    self.tdscf = kw['tdscf'] if 'tdscf' in kw else None
    if sum(self.nelec) == 1: raise RuntimeError('Not implemented H, sorry :-) Look into scf/__init__.py for HF1e class...')
    
    if self.nspin==1: self.nocc_0t = nocc_0t = np.array([int((self.nelec+1)/2)])
    elif self.nspin==2: self.nocc_0t = nocc_0t = self.nelec
    else: raise RuntimeError('nspin>2?')

    if self.verbosity>0: print(__name__,'\t\t====> Number of occupied states = {}, states up to fermi level= {}, nspin = {}, and magnetization = {}'.format(nocc_0t,self.nfermi,self.nspin,self.spin))

    if 'nocc' in kw:
      s2nocc = [kw['nocc']] if type(kw['nocc'])==int else kw['nocc']
      self.nocc = array([min(i,j) for i,j in zip(s2nocc,nocc_0t)])
    else :
      self.nocc = array([min(6,j) for j in nocc_0t])
    
    if 'nvrt' in kw:
      s2nvrt = [kw['nvrt']] if type(kw['nvrt'])==int else kw['nvrt']
      self.nvrt = array([min(i,j) for i,j in zip(s2nvrt,self.norbs-nocc_0t)])
    else :
      self.nvrt = array([min(6,j) for j in self.norbs-nocc_0t])

    if self.verbosity>0: print(__name__,'\t\t====> Number of ocupied states are (nocc) = {}, Number of virtual states (nvrt) = {}'.format(self.nocc, self.nvrt))

    self.start_st,self.finish_st = self.nocc_0t-self.nocc, self.nocc_0t+self.nvrt

    self.nn = [range(self.start_st[s]+1, self.finish_st[s]) for s in range(self.nspin)] # list of states will be started from [1:]
    if self.verbosity>0: print(__name__,'\t\t====> Indices of states to be corrected = {}\n'.format(self.nn))

    if 'nocc_conv' in kw:
      s2nocc_conv = [kw['nocc_conv']] if type(kw['nocc_conv'])==int else kw['nocc_conv']
      self.nocc_conv = array([min(i,j) for i,j in zip(s2nocc_conv,nocc_0t)])
    else :
      self.nocc_conv = self.nocc

    if 'nvrt_conv' in kw:
      s2nvrt_conv = [kw['nvrt_conv']] if type(kw['nvrt_conv'])==int else kw['nvrt_conv']
      self.nvrt_conv = array([min(i,j) for i,j in zip(s2nvrt_conv,self.norbs-nocc_0t)])
    else :
      self.nvrt_conv = self.nvrt
    
    if self.rescf: self.kernel_scf() # here is rescf with HF functional tacitly assumed
        
    self.nff_ia = kw['nff_ia'] if 'nff_ia' in kw else 32
    self.tol_ia = kw['tol_ia'] if 'tol_ia' in kw else 1e-10
    (wmin_def,wmax_def,tmin_def,tmax_def) = self.get_wmin_wmax_tmax_ia_def(self.tol_ia)
    self.wmin_ia = kw['wmin_ia'] if 'wmin_ia' in kw else wmin_def
    self.wmax_ia = kw['wmax_ia'] if 'wmax_ia' in kw else wmax_def
    self.tmin_ia = kw['tmin_ia'] if 'tmin_ia' in kw else tmin_def
    self.tmax_ia = kw['tmax_ia'] if 'tmax_ia' in kw else tmax_def
    self.tt_ia,self.ww_ia = log_mesh(self.nff_ia, self.tmin_ia, self.tmax_ia, self.wmax_ia)
    #print('self.tmin_ia, self.tmax_ia, self.wmax_ia')
    #print(self.tmin_ia, self.tmax_ia, self.wmax_ia)
    #print(self.ww_ia[0], self.ww_ia[-1])

    self.dw_ia = self.ww_ia*(log(self.ww_ia[-1])-log(self.ww_ia[0]))/(len(self.ww_ia)-1)
    self.dw_excl = self.ww_ia[0]
    
    assert self.cc_da.shape[1]==self.nprod
    self.kernel_sq = self.hkernel_den
    #self.v_dab_ds = self.pb.get_dp_vertex_doubly_sparse(axis=2)

    self.x = require(self.mo_coeff[0,:,:,:,0], dtype=self.dtype, requirements='CW')

    if self.perform_gw: self.kernel_gw()
    
  def get_h0_vh_x_expval(self):
    if self.nspin==1:
      mat = self.get_hcore()+self.get_j()-0.5*self.get_k()
      mat1 = dot(self.mo_coeff[0,0,:,:,0], mat)
      expval = einsum('nb,nb->n', mat1, self.mo_coeff[0,0,:,:,0]).reshape((1,self.norbs))
    elif self.nspin==2:
      vh = self.get_j()
      mat = self.get_hcore()+vh[0]+vh[1]-self.get_k()
      expval = zeros((self.nspin, self.norbs))
      for s in range(self.nspin):
        mat1 = dot(self.mo_coeff[0,s,:,:,0], mat[s])
        expval[s] = einsum('nb,nb->n', mat1, self.mo_coeff[0,s,:,:,0])
    return expval
    
  def get_wmin_wmax_tmax_ia_def(self, tol):
    from numpy import log, exp, sqrt, where, amin, amax
    """ 
      This is a default choice of the wmin and wmax parameters for a log grid along 
      imaginary axis. The default choice is based on the eigenvalues. 
    """
    E = self.ksn2e[0,0,:]
    E_fermi = self.fermi_energy
    E_homo = amax(E[where(E<=E_fermi)])
    E_gap  = amin(E[where(E>E_fermi)]) - E_homo  
    E_maxdiff = amax(E) - amin(E)
    d = amin(abs(E_homo-E)[where(abs(E_homo-E)>1e-4)])
    wmin_def = sqrt(tol * (d**3) * (E_gap**3)/(d**2+E_gap**2))
    wmax_def = (E_maxdiff**2/tol)**(0.250)
    tmax_def = -log(tol)/ (E_gap)
    tmin_def = -100*log(1.0-tol)/E_maxdiff
    return wmin_def, wmax_def, tmin_def,tmax_def


  rf0_den = rf0_den
  rf0_cmplx_ref_blk = rf0_cmplx_ref_blk
  rf0_cmplx_ref = rf0_cmplx_ref
  rf0_cmplx_vertex_dp = rf0_cmplx_vertex_dp
  rf0_cmplx_vertex_ac = rf0_cmplx_vertex_ac
  rf0 = rf0_den
  
  rf = rf_den
  
  rf_pyscf = rf_den_pyscf

  def si_c(self, ww):
    from numpy.linalg import solve
    """ 
    This computes the correlation part of the screened interaction W_c
    by solving <self.nprod> linear equations (1-K chi0) W = K chi0 K
    scr_inter[w,p,q], where w in ww, p and q in 0..self.nprod 
    """
    rf0 = si0 = self.rf0(ww)
    for iw,w in enumerate(ww):
      k_c = dot(self.kernel_sq, rf0[iw,:,:])
      b = dot(k_c, self.kernel_sq)
      k_c = np.eye(self.nprod)-k_c
      si0[iw,:,:] = solve(k_c, b)

    return si0

  def si_c_via_diagrpa(self, ww):
    """ 
    This method computes the correlation part of the screened interaction W_c
    via the interacting response function. The interacting response function,
    in turn, is computed by diagonalizing RPA Hamiltonian.
    """
    rf = si0 = self.rf_pyscf(ww)
    for iw,r in enumerate(rf):
      si0[iw] = dot(self.kernel_sq, dot(r, self.kernel_sq))
    return si0

  def get_snmw2sf(self):
    """ 
    This computes a spectral function of the GW correction.
    sf[spin,n,m,w] = X^n V_mu X^m W_mu_nu X^n V_nu X^m,
    where n runs from s...f, m runs from 0...norbs, w runs from 0...nff_ia, spin=0...1 or 2.
    """
    wpq2si0 = self.si_c(ww = 1j*self.ww_ia).real
    v_pab = self.pb.get_ac_vertex_array()

    snmw2sf = []
    for s in range(self.nspin):
      nmw2sf = zeros((len(self.nn[s]), self.norbs, self.nff_ia), dtype=self.dtype)
      #nmw2sf = zeros((len(self.nn), self.norbs, self.nff_ia), dtype=self.dtype)
      xna = self.mo_coeff[0,s,self.nn[s],:,0]
      #xna = self.mo_coeff[0,s,self.nn,:,0]
      xmb = self.mo_coeff[0,s,:,:,0]
      nmp2xvx = einsum('na,pab,mb->nmp', xna, v_pab, xmb)
      for iw,si0 in enumerate(wpq2si0):
        nmw2sf[:,:,iw] = einsum('nmp,pq,nmq->nm', nmp2xvx, si0, nmp2xvx)
      snmw2sf.append(nmw2sf)
    return snmw2sf

  def gw_corr_int(self, sn2w, eps=None):
    """ This computes an integral part of the GW correction at energies sn2e[spin,len(self.nn)] """
    if not hasattr(self, 'snmw2sf'): self.snmw2sf = self.get_snmw2sf()
    sn2int = [np.zeros_like(n2w, dtype=self.dtype) for n2w in sn2w ]
    eps = self.dw_excl if eps is None else eps
    #print(__name__, 'self.dw_ia', self.dw_ia, sn2w)
    for s,ww in enumerate(sn2w):
      for n,w in enumerate(ww):
        #print(__name__, 's,n,w int corr', s,n,w)
        for m in range(self.norbs):
          if abs(w-self.ksn2e[0,s,m])<eps : continue
          state_corr = ((self.dw_ia*self.snmw2sf[s][n,m,:] / (w + 1j*self.ww_ia-self.ksn2e[0,s,m])).sum()/pi).real
          #print(n, m, -state_corr, w-self.ksn2e[0,s,m])
          sn2int[s][n] -= state_corr
    return sn2int

  def gw_corr_res(self, sn2w):
    """ This computes a residue part of the GW correction at energies sn2w[spin,len(self.nn)] """
    v_pab = self.pb.get_ac_vertex_array()
    sn2res = [np.zeros_like(n2w, dtype=self.dtype) for n2w in sn2w ]
    for s,ww in enumerate(sn2w):
      x = self.mo_coeff[0,s,:,:,0]
      for nl,(n,w) in enumerate(zip(self.nn[s],ww)):
        lsos = self.lsofs_inside_contour(self.ksn2e[0,s,:],w,self.dw_excl)
        zww = array([pole[0] for pole in lsos])
        #print(__name__, s,n,w, 'len lsos', len(lsos))
        si_ww = self.si_c(ww=zww)
        xv = dot(v_pab,x[n])
        for pole,si in zip(lsos, si_ww.real):
          xvx = dot(xv, x[pole[1]])
          contr = dot(xvx, dot(si, xvx))
          #print(pole[0], pole[2], contr)
          sn2res[s][nl] += pole[2]*contr
    return sn2res

  def lsofs_inside_contour(self, ee, w, eps):
    """ 
      Computes number of states the eigen energies of which are located inside an integration contour.
      The integration contour depends on w 
    """ 
    nGamma_pos = 0
    nGamma_neg = 0
    for i,e in enumerate(ee):
      zr = e - w
      zi = -np.sign(e-self.fermi_energy)
      if zr>=eps and zi>0 : nGamma_pos +=1
      if abs(zr)<eps and zi>0 : nGamma_pos +=1
    
      if zr<=-eps and zi<0 : nGamma_neg += 1
      if abs(zr)<eps and zi<0 : nGamma_neg += 1

    npol = max(nGamma_pos,nGamma_neg)
    if npol<1: return []

    i2zsc = []
    ipol = 0
    for i,e in enumerate(ee):
      zr = e - w
      zi = -eps*np.sign(e-self.fermi_energy)

      if zr>=eps and zi>0:
        ipol += 1
        if ipol>npol: raise RuntimeError('ipol>npol')
        i2zsc.append( [zr+1j*zi, i, -1.0] )
      elif zr<=-eps and zi<0:
        ipol += 1
        if ipol>npol: raise RuntimeError('ipol>npol')
        i2zsc.append( [zr+1j*zi, i, +1.0] )
      elif abs(zr)<eps and zi>0:
        ipol += 1
        if ipol>npol: raise RuntimeError('ipol>npol')
        i2zsc.append( [zr+1j*zi, i, -0.5] ) #[zr+1j*zi, i, -0.5]
      elif abs(zr)<eps and zi<0:
        ipol +=1
        if ipol>npol: raise RuntimeError('ipol>npol')
        i2zsc.append( [zr+1j*zi, i, +0.5] ) #[zr+1j*zi, i, +0.5]

    if ipol!=npol: raise RuntimeError('loop logics incompat???')
    return i2zsc
  
  def g0w0_eigvals(self):
    """ This computes the G0W0 corrections to the eigenvalues """
    sn2eval_gw = [np.copy(self.ksn2e[0,s,nn]) for s,nn in enumerate(self.nn) ]
    sn2eval_gw_prev = copy(sn2eval_gw)

    self.nn_conv = []  # self.nn_conv -- list of states to converge, spin-resolved.
    for nocc_0t,nocc_conv,nvrt_conv in zip(self.nocc_0t, self.nocc_conv, self.nvrt_conv):
      self.nn_conv.append( range(max(nocc_0t-nocc_conv,0), min(nocc_0t+nvrt_conv,self.norbs)))

    # iterations to converge the 
    if self.verbosity>0: print('-'*len(sn2eval_gw[0])*4,'|  G0W0 corrections of eigenvalues  |','-'*len(sn2eval_gw[0])*4)
    for i in range(self.niter_max_ev):
      sn2i = self.gw_corr_int(sn2eval_gw)
      sn2r = self.gw_corr_res(sn2eval_gw)
      
      sn2eval_gw = []
      for s,(evhf,n2i,n2r,nn) in enumerate(zip(self.h0_vh_x_expval,sn2i,sn2r,self.nn)):
        sn2eval_gw.append(evhf[nn]+n2i+n2r)
        
      sn2mismatch = zeros((self.nspin,self.norbs))
      for s, nn in enumerate(self.nn): sn2mismatch[s,nn] = sn2eval_gw[s][:]-sn2eval_gw_prev[s][:]
      sn2eval_gw_prev = copy(sn2eval_gw)
      err = 0.0
      for s,nn_conv in enumerate(self.nn_conv): err += abs(sn2mismatch[s,nn_conv]).sum()/len(nn_conv)

      if self.verbosity>0:
        np.set_printoptions(linewidth=1000, suppress=True, precision=5)
        print('Iteration #{}  Relative Error: {:.7f}'.format(i+1, err))
      if self.verbosity>1:
        #print(sn2mismatch)
        for s,n2ev in enumerate(sn2eval_gw):
          print('Spin{} {}'.format(s+1, n2ev[:]*HARTREE2EV)) #, sn2i[s][:]*HARTREE2EV, sn2r[s][:]*HARTREE2EV))
        
      if err<self.tol_ev : 
        if self.verbosity>0: print('-'*len(sn2eval_gw[0])*4,' |  Convergence has been reached  | ','-'*len(sn2eval_gw[0])*4,'\n')
        break
    return sn2eval_gw
    
  def report(self):
    """ Prints the energy levels """
    import re
    emfev = self.mo_energy[0].T * HARTREE2EV
    egwev = self.mo_energy_gw[0].T * HARTREE2EV
    file_name= re.sub('[^A-Za-z]', '', self.mol.atom)
    with open('report_'+file_name+'.out','w') as out_file:
        print('-'*30,'|G0W0 eigenvalues (eV)|','-'*30)
        out_file.write('-'*30+'|G0W0 eigenvalues (eV)|'+'-'*30+'\n')
        if self.nspin==1:
            print("\n   n  %14s %14s %7s " % ("E_mf", "E_gw", "occ") )
            for ie,(emf,egw,f) in enumerate(zip(emfev,egwev,self.mo_occ[0].T)):
                print("%5d  %14.7f %14.7f %7.2f " % (ie, emf[0], egw[0], f[0]) )
            print('Fermi energy        (eV):\t{:f}'.format(self.fermi_energy* HARTREE2EV))
            out_file.write('Fermi energy        (eV):\t{:f}\n'.format(self.fermi_energy* HARTREE2EV))            
            print('G0W0 HOMO energy    (eV):\t{:f}'.format(egwev[self.nfermi[0]-1,0]))
            out_file.write('G0W0 HOMO energy    (eV):\t{:f}\n'.format(egwev[self.nfermi[0]-1,0]))
            print('G0W0 LUMO energy    (eV):\t{:f}'.format(egwev[self.nfermi[0],0]))
            out_file.write('G0W0 LUMO energy    (eV):\t{:f}\n'.format(egwev[self.nfermi[0],0]))
            print('G0W0 HOMO-LUMO gap  (eV):\t{:f}'.format(egwev[self.nfermi[0],0]-egwev[self.nfermi[0]-1,0]))
            out_file.write('G0W0 HOMO-LUMO gap  (eV):\t{:f}\n'.format(egwev[self.nfermi[0],0]-egwev[self.nfermi[0]-1,0]))
        elif self.nspin==2:
            print("\n    n %14s %14s  %7s | %14s %14s  %7s" % ("E_mf_up", "E_gw_up", "occ_up", "E_mf_down", "E_gw_down", "occ_down"))
            out_file.write("\n    n %14s %14s  %7s | %14s %14s  %7s\n" % ("E_mf_up", "E_gw_up", "occ_up", "E_mf_down", "E_gw_down", "occ_down"))
            for ie,(emf,egw,f) in enumerate(zip(emfev,egwev,self.mo_occ[0].T)):
                print("%5d  %14.7f %14.7f %7.2f | %14.7f %14.7f %7.2f" % (ie+1, emf[0], egw[0], f[0],  emf[1], egw[1], f[1]) )
                out_file.write ("%5d  %14.7f %14.7f %7.2f | %14.7f %14.7f %7.2f\n" % (ie+1, emf[0], egw[0], f[0],  emf[1], egw[1], f[1]) )
            print('Fermi energy        (eV):\t{:f}'.format(self.fermi_energy* HARTREE2EV))
            out_file.write('Fermi energy        (eV):\t{:f}\n'.format(self.fermi_energy* HARTREE2EV))
            print('G0W0 HOMO energy    (eV):\t{:f}\t{:f}'.format(egwev[self.nfermi[0]-1,0],egwev[self.nfermi[1]-1,1]))
            out_file.write('G0W0 HOMO energy    (eV):\t{:f}\t{:f}\n'.format(egwev[self.nfermi[0]-1,0],egwev[self.nfermi[1]-1,1]))
            print('G0W0 LUMO energy    (eV):\t{:f}\t{:f}'.format(egwev[self.nfermi[0],0],egwev[self.nfermi[1],1]))
            out_file.write('G0W0 LUMO energy    (eV):\t{:f}\t{:f}\n'.format(egwev[self.nfermi[0],0],egwev[self.nfermi[1],1]))
            print('G0W0 HOMO-LUMO gap  (eV):\t{:f}\t{:f}'.format(egwev[self.nfermi[0],0]-egwev[self.nfermi[0]-1,0],egwev[self.nfermi[1],1]-egwev[self.nfermi[1]-1,1]))
            out_file.write('G0W0 HOMO-LUMO gap  (eV):\t{:f}\t{:f}\n'.format(egwev[self.nfermi[0],0]-egwev[self.nfermi[0]-1,0],egwev[self.nfermi[1],1]-egwev[self.nfermi[1]-1,1]))
        else:
            raise RuntimeError('not implemented...')
        t= int(timer() - starting_time)
        print('\nTotal running time is: {:02d}:{:02d}:{:02d}\nJOB DONE!'.format(t // 3600, (t % 3600 // 60), t % 60))
        out_file.write('\nTotal running time is: {:02d}:{:02d}:{:02d}\nJOB DONE!\n'.format(t // 3600, (t % 3600 // 60), t % 60))
        out_file.close
 
    
  def make_mo_g0w0(self):
    """ This creates the fields mo_energy_g0w0, and mo_coeff_g0w0 """

    self.h0_vh_x_expval = self.get_h0_vh_x_expval()
    if self.verbosity>1:
      print(__name__,'\t\t====> Expectation values of Hartree-Fock Hamiltonian (eV):\n no.\t  H_up\t\t  H_down')
      for i , (ab) in enumerate(zip(self.h0_vh_x_expval[0].T* HARTREE2EV,self.h0_vh_x_expval[1].T* HARTREE2EV)):
	      print(' {:d}\t{:f}\t{:f}'.format(i+1, ab[0],ab[1]))

    if not hasattr(self,'sn2eval_gw'): self.sn2eval_gw=self.g0w0_eigvals() # Comp. GW-corrections
    
    # Update mo_energy_gw, mo_coeff_gw after the computation is done
    self.mo_energy_gw = np.copy(self.mo_energy)
    self.mo_coeff_gw = np.copy(self.mo_coeff)
    #print(len(self.sn2eval_gw), type(self.sn2eval_gw))
    #print(self.nn, type(self.nn))
    #print(self.mo_energy_gw.shape, type(self.mo_energy_gw))
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
      if self.verbosity>0: print(__name__, '\t\t====> Spin {}: Corrected Molecular orbital indices in sorted order: {}'.format(str(s+1),np.argsort(self.mo_energy_gw[0,s,:])))
      argsrt = np.argsort(self.mo_energy_gw[0,s,:])
      self.mo_energy_gw[0,s,:] = np.sort(self.mo_energy_gw[0,s,:])
      for n,m in enumerate(argsrt): self.mo_coeff_gw[0,0,n] = self.mo_coeff[0,0,m]
 
    self.xc_code = 'GW'
    if self.verbosity>0:
      print(__name__,'\t\t====> Performed xc_code: {}\n '.format(self.xc_code))
      print('\nConverged GW-corrected eigenvalues:\n',self.mo_energy_gw)
        
  kernel_gw = make_mo_g0w0
