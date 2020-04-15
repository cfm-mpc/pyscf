from __future__ import division
import numpy as np

def gw_xvx_dpcoo(self):

    xvx = []
    size = self.cc_da.shape[0]
    nfdp = self.pb.dpc2s[-1]

    for spin in range(self.nspin):

        vxdp = dpcoo_step1(self, size, nfdp, spin)
        vxdp = dpcoo_step2(vxdp, size, self.norbs)
        vxdp = dpcoo_step3(self, vxdp, size, spin)
        xvx.append(dpcoo_step4(self.cc_da, vxdp).todense())

    return xvx

def dpcoo_step1(self, size, nfdp, spin):
    import sparse

    xmb = sparse.COO.from_numpy(self.mo_coeff[0, spin, :, :, 0]).T
    v_pd = sparse.COO.from_scipy_sparse(self.v_dab.reshape(nfdp*self.norbs, self.norbs))

    return sparse.COO.dot(v_pd, xmb)

def dpcoo_step2(vxdp, size, norbs):

    return vxdp.reshape((size, norbs, \
            norbs)).transpose(axes=(1, 0, 2)).reshape((norbs, size*norbs))

def dpcoo_step3(self, vxdp, size, spin):
    import sparse

    xna = sparse.COO.from_numpy(self.mo_coeff[0, spin, self.nn[spin], :, 0])
    xxv2 = xna.dot(vxdp).reshape((len(self.nn[spin]), size, self.norbs))
    return xxv2.transpose(axes=(0, 2, 1))

def dpcoo_step4(cc_da, vxdp):
    import sparse
    return vxdp.dot(sparse.COO.from_scipy_sparse(cc_da))
