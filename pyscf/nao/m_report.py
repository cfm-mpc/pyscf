from __future__ import print_function, division
import numpy as np
from pyscf.data.nist import HARTREE2EV
import time

start_time = time.time()

def report_gw (self):
    """ Prints the energy levels of mean-field and G0W0"""
    import re
    emfev = self.mo_energy[0].T * HARTREE2EV
    egwev = self.mo_energy_gw[0].T * HARTREE2EV
    file_name= ''.join(self.get_symbols())
    # The output should be possible to write more concise...
    with open('report_'+file_name+'.out','w') as out_file:
        print('-'*30,'|G0W0 eigenvalues (eV)|','-'*30)
        out_file.write('-'*30+'|G0W0 eigenvalues (eV)|'+'-'*30+'\n')
        if self.nspin==1:
            out_file.write('Energy-sorted MO indices \t {}'.format(self.argsort[0]))
            if (np.allclose(self.argsort[0][:self.nfermi[0]],np.sort(self.argsort[0][:self.nfermi[0]]))==False):
                    print ("Warning: Swapping in orbital energies below Fermi has happened!")
                    out_file.write("\nWarning: Swapping in orbital energies below Fermi has happened!")
            print("\n   n  %14s %14s %7s " % ("E_mf", "E_gw", "occ") )
            out_file.write("\n   n  %14s %14s %7s \n" % ("E_mf", "E_gw", "occ") )
            for ie,(emf,egw,f) in enumerate(zip(emfev,egwev,self.mo_occ[0].T)):
                print("%5d  %14.7f %14.7f %7.2f " % (ie, emf[0], egw[0], f[0]) )
                out_file.write("%5d  %14.7f %14.7f %7.2f\n" % (ie, emf[0], egw[0], f[0]) )
            print('\nFermi energy        (eV):%16.7f'%(self.fermi_energy* HARTREE2EV))
            out_file.write('\nFermi energy        (eV):%16.7f\n'%(self.fermi_energy* HARTREE2EV))            
            print('G0W0 HOMO energy    (eV):%16.7f' % (egwev[self.nfermi[0]-1,0]))
            out_file.write('G0W0 HOMO energy    (eV):%16.7f\n'%(egwev[self.nfermi[0]-1,0]))
            print('G0W0 LUMO energy    (eV):%16.7f' % (egwev[self.nfermi[0],0]))
            out_file.write('G0W0 LUMO energy    (eV):%16.7f\n'%(egwev[self.nfermi[0],0]))
            print('G0W0 HOMO-LUMO gap  (eV):%16.7f' %(egwev[self.nfermi[0],0]-egwev[self.nfermi[0]-1,0]))
            out_file.write('G0W0 HOMO-LUMO gap  (eV):%16.7f\n'%(egwev[self.nfermi[0],0]-egwev[self.nfermi[0]-1,0]))
        elif self.nspin==2:
            for s in range(2):
                out_file.write('\nEnergy-sorted MO indices for spin {}\t {}'.format(str(s+1),self.argsort[s][max(self.nocc_0t[s]-10,0):min(self.nocc_0t[s]+10, self.norbs)]))
                if (np.allclose(self.argsort[s][:self.nfermi[s]],np.sort(self.argsort[s][:self.nfermi[s]]))==False):
                    print ("Warning: Swapping in orbital energies below Fermi has happened at spin {} channel!".format(s+1))
                    out_file.write("\nWarning: Swapping in orbital energies below Fermi has happened at spin {} channel!\n".format(s+1))         
            print("\n    n %14s %14s  %7s | %14s %14s  %7s" % ("E_mf_up", "E_gw_up", "occ_up", "E_mf_down", "E_gw_down", "occ_down"))
            out_file.write("\n    n %14s %14s  %7s | %14s %14s  %7s\n" % ("E_mf_up", "E_gw_up", "occ_up", "E_mf_down", "E_gw_down", "occ_down"))
            for ie,(emf,egw,f) in enumerate(zip(emfev,egwev,self.mo_occ[0].T)):
                print("%5d  %14.7f %14.7f %7.2f | %14.7f %14.7f %7.2f" % (ie, emf[0], egw[0], f[0],  emf[1], egw[1], f[1]) )
                out_file.write ("%5d  %14.7f %14.7f %7.2f | %14.7f %14.7f %7.2f\n" % (ie, emf[0], egw[0], f[0],  emf[1], egw[1], f[1]) )
            print('\nFermi energy        (eV):%16.7f'%(self.fermi_energy* HARTREE2EV))
            out_file.write('\nFermi energy        (eV):%16.7f\n'%(self.fermi_energy* HARTREE2EV))
            print('G0W0 HOMO energy    (eV):%16.7f %16.7f'%(egwev[self.nfermi[0]-1,0],egwev[self.nfermi[1]-1,1]))
            out_file.write('G0W0 HOMO energy    (eV):%16.7f %16.7f\n'%(egwev[self.nfermi[0]-1,0],egwev[self.nfermi[1]-1,1]))
            print('G0W0 LUMO energy    (eV):%16.7f %16.7f'%(egwev[self.nfermi[0],0],egwev[self.nfermi[1],1]))
            out_file.write('G0W0 LUMO energy    (eV):%16.7f %16.7f\n'%(egwev[self.nfermi[0],0],egwev[self.nfermi[1],1]))
            print('G0W0 HOMO-LUMO gap  (eV):%16.7f %16.7f'%(egwev[self.nfermi[0],0]-egwev[self.nfermi[0]-1,0],egwev[self.nfermi[1],1]-egwev[self.nfermi[1]-1,1]))
            out_file.write('G0W0 HOMO-LUMO gap  (eV):%16.7f %16.7f\n'%(egwev[self.nfermi[0],0]-egwev[self.nfermi[0]-1,0],egwev[self.nfermi[1],1]-egwev[self.nfermi[1]-1,1]))
        else:
            raise RuntimeError('not implemented...')
        print('G0W0 Total energy   (eV):%16.7f' %(self.etot_gw*HARTREE2EV))
        out_file.write('G0W0 Total energy   (eV):%16.7f\n'%(self.etot_gw*HARTREE2EV))
        elapsed_time = time.time() - start_time
        print('\nTotal running time is: {}\nJOB DONE! \t {}'.format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)),time.strftime("%c")))
        out_file.write('\nTotal running time is: {}\nJOB DONE! \t {}'.format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)),time.strftime("%c")))         
        out_file.close


def report_mfx(self):
    """
    This calculates the h_core, Hartree (K) and Exchange(J) expectation value.
    """
    #import re
    #import sys
    #file_name= ''.join(self.get_symbols())
    #sys.stdout=open('report_mf_'+file_name+'.out','w')

    dm1 = self.make_rdm1()
    H = self.get_hcore()
    ecore = (H*dm1[0,...,0]).sum()
    co = self.get_j()

    exp_h = np.zeros((self.nspin, self.norbs))
    exp_co = np.zeros((self.nspin, self.norbs))
    exp_x = np.zeros((self.nspin, self.norbs))
    if self.nspin==1:
        x = -0.5* self.get_k()
        mat_h = np.dot(self.mo_coeff[0,0,:,:,0], H)                   
        exp_h = np.einsum('nb,nb->n', mat_h, self.mo_coeff[0,0,:,:,0])
        mat_co = np.dot(self.mo_coeff[0,0,:,:,0], co)                   
        exp_co = np.einsum('nb,nb->n', mat_co, self.mo_coeff[0,0,:,:,0])
        mat_x = np.dot(self.mo_coeff[0,0,:,:,0], x)
        exp_x = np.einsum('nb,nb->n', mat_x, self.mo_coeff[0,0,:,:,0])
        print('='*12,'| the Exchange expectation value (eV) |','='*12)
        print('%3s  %12s  %12s  %12s  %12s  %3s'%('no.','<H_core>','<K>  ','<Sigma_x>','MF energy','occ'))
        for i, (a,b,c,d,e) in enumerate(zip(exp_h.T*HARTREE2EV,exp_co.T*HARTREE2EV, exp_x.T*HARTREE2EV, self.mo_energy.T*HARTREE2EV, self.mo_occ[0].T)):   
          if (i==self.nfermi[0]): print('-'*64)
          print(' %3d  %12.6f  %12.6f %12.6f %12.6f  %3d'%(i, a,b,c,d,e))
        Vha = 0.5*(co*dm1[0,...,0]).sum()
        EX = 0.5*(x*dm1[0,...,0]).sum()

    elif self.nspin==2:
        x = -self.get_k()
        cou = co[0]+co[1]
        Vha = 0.5*(cou*dm1[0,...,0]).sum()
        for s in range(self.nspin):
          mat_h = np.dot(self.mo_coeff[0,s,:,:,0], H)                   
          exp_h[s] = np.einsum('nb,nb->n', mat_h, self.mo_coeff[0,s,:,:,0])
          mat_co = np.dot(self.mo_coeff[0,s,:,:,0], cou)                   
          exp_co[s] = np.einsum('nb,nb->n', mat_co, self.mo_coeff[0,s,:,:,0])
          mat_x = np.dot(self.mo_coeff[0,s,:,:,0], x[s])
          exp_x[s] = np.einsum('nb,nb->n', mat_x, self.mo_coeff[0,s,:,:,0])
        print('='*42,'| the Exchange expectation value (eV) |','='*42)
        print('%3s  %12s  %12s  %12s  %12s  %3s |%12s  %12s  %12s  %12s  %3s '%('no.','<H_core>','<K>  ','<Sigma_x>','MF energy','occ','<H_core>','<K>   ','<Sigma_x>','MF energy','occ'))        
        for i , (a,b,c,d,e) in enumerate(zip(exp_h.T*HARTREE2EV,exp_co.T*HARTREE2EV, exp_x.T*HARTREE2EV, self.mo_energy.T*HARTREE2EV, self.mo_occ[0].T)):
          if (i==self.nfermi[0] or i==self.nfermi[1]): print('-'*125)
          print(' %3d  %12.6f  %12.6f %12.6f %12.6f  %3d  | %12.6f  %12.6f  %12.6f %12.6f  %3d'%(i, a[0],b[0],c[0],d[0],e[0],a[1],b[1],c[1],d[1],e[1]))
        EX = 0.5*(x*dm1[0,...,0]).sum()

    if hasattr(self, 'mf'): 
        print('\nmean-field Nucleus-Nucleus   (eV):%16.6f'%(self.energy_nuc()))
        print('mean-field Core energy       (eV):%16.6f'%(ecore))
        print('mean-field Exchange energy   (eV):%16.6f'%(EX))
        print('mean-field Hartree energy    (eV):%16.6f'%(Vha))
        print('mean-field Total energy      (eV):%16.6f'%(self.mf.e_tot))
        S = self.spin/2
        S0 = S*(S+1)
        SS = self.mf.spin_square()
        if ( SS[0]!= S ):
            print('<S^2> and  2S+1                  :%16.7f %16.7f'%(SS[0],SS[1]))
            print('Instead of                       :%16.7f %16.7f'%(S0, 2*S+1))
    #sys.stdout.close()




def exfock(self):
    """
    This calculates the Exchange expectation value, when:
    self.get_k() = Exchange operator/energy
    mat1 is product of this operator and molecular coefficients and it will be diagonalized in expval by einsum
    """
    if self.nspin==1:
      mat = -0.5*self.get_k()
      mat1 = np.dot(self.mo_coeff[0,0,:,:,0], mat)
      expval = np.einsum('nb,nb->n', mat1, self.mo_coeff[0,0,:,:,0]).reshape((1,self.norbs))
      print('---------| Expectationvalues of Exchange energy(eV) |---------\n %3s  %16s  %3s'%('no.','<Sigma_x> ','occ'))
      for i, (a,b) in enumerate(zip(expval.T*HARTREE2EV,self.mo_occ[0].T)):   #self.h0_vh_x_expval[0,:self.nfermi[0]+5] to limit the virual states
        if (i==self.nfermi[0]): print('-'*62)
        print (' %3d  %16.6f  %3d'%(i,a[0], b[0]))
    elif self.nspin==2:
      mat = -self.get_k()
      expval = np.zeros((self.nspin, self.norbs))
      for s in range(self.nspin):
        mat1 = np.dot(self.mo_coeff[0,s,:,:,0], mat[s])
        expval[s] = np.einsum('nb,nb->n', mat1, self.mo_coeff[0,s,:,:,0])
      print('-----------| the Exchange expectation value (eV) |-----------\n %3s  %16s  %3s  | %12s  %3s'%('no.','<Sigma_x>','occ','<Sigma_x>','occ'))        
      for i , (a,b) in enumerate(zip(expval.T* HARTREE2EV,self.mo_occ[0].T)):
        if (i==self.nfermi[0] or i==self.nfermi[1]): print('-'*60)
        print(' %3d  %16.6f  %3d  | %12.6f  %3d'%(i, a[0],b[0],a[1], b[1]))
    #return expval



#
# Example of reporting expectation values of mean-field calculations.
#
if __name__=='__main__':
    import numpy as np 
    from pyscf import gto, scf
    from pyscf.nao import gw as gw_c
    HARTREE2EV=27.2114
    mol = gto.M( verbose = 0, atom = 'O 0.0, 0.0, 0.622978 ; O 0.0, 0.0, -0.622978',basis = 'cc-pvdz', spin=2, charge=0)
    gto_mf = scf.UHF(mol)
    gto_mf.kernel()
    gw = gw_c(mf=gto_mf, gto=mol, verbosity=3, niter_max_ev=70, kmat_algo='sm0_sum')
    gw.report_mf()  #prints the energy levels of mean-field components
    gw.kernel_gw()
    gw.report()     #gives G0W0 spectra
