from __future__ import print_function, division
import unittest, numpy as np
from pyscf import gto, scf
from pyscf.nao import gw_iter
import numpy as np

class KnowValues(unittest.TestCase):

  def test_sf_iter(self):
    """ This compares matrix element of W calculated by G0W0 and G0W0_iter """

    mol = gto.M(atom='''O 0.0, 0.0, 0.622978 ; O 0.0, 0.0, -0.622978''', basis='ccpvdz',spin=2)
    mf = scf.UHF(mol)
    mf.kernel()

    gw = gw_iter(mf=mf, gto=mol, verbosity=1, niter_max_ev=1, nff_ia=5, nvrt=1,
                 nocc=1, krylov_solver="lgmres", krylov_options={"atol": 1e-04,
                                                                 "tol": 1.0e-5})

    gw_it = gw.get_snmw2sf_iter()
    gw_ref = gw.get_snmw2sf()
    self.assertTrue(np.allclose(gw_it, gw_ref, atol=1.0e-4))

    sn2eval_gw = [np.copy(gw.ksn2e[0,s,nn]) for s,nn in enumerate(gw.nn) ]
    sn2r_it  = gw.gw_corr_res_iter(sn2eval_gw)
    sn2r_ref = gw.gw_corr_res(sn2eval_gw)
    self.assertTrue(np.allclose(sn2r_it, sn2r_ref, atol=1.0e-4))
    
    
if __name__ == "__main__": unittest.main()
