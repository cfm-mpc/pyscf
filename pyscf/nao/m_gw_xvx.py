from __future__ import division
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
import numba as nb

def gw_xvx_dpcoo(self):

    xvx = []
    size = self.cc_da.shape[0]
    nfdp = self.pb.dpc2s[-1]
    v_pd = self.v_dab.reshape(self.v_dab_csr.shape[0]*self.norbs, self.norbs).tocsr()

    for spin in range(self.nspin):

        xvx2 = csrmat_denmat_custom2(v_pd.indptr, v_pd.indices, v_pd.data,
                                     self.mo_coeff[0, spin, :, :, 0].T,
                                     self.mo_coeff[0, spin, self.nn[spin], :, 0],
                                     nfdp, self.norbs, len(self.nn[spin]))
        xvx2 = xvx2.reshape(len(self.nn[spin]), size, self.norbs)
        xvx2 = np.swapaxes(xvx2, 1, 2)

        xvx2_fin = np.zeros((xvx2.shape[0], xvx2.shape[1], self.cc_da.shape[1]),
                             dtype=xvx2.dtype)
        # Somehow dense.dot(sparse) uses a crazy amount of memory ...
        # xvx2 = xvx2.dot(self.cc_da)
        # better to do sparse.T.dot(dense.T).T
        for i in range(xvx2_fin.shape[0]):
            xvx2_fin[i, :, :] = self.cc_da_trans.dot(xvx2[i, :, :].T).T

        xvx.append(xvx2_fin)

    return xvx

@nb.jit(nopython=True)
def csrmat_denmat_custom2(indptr, indices, data, B, X, N1, N2, N3):
    """
    Perform in one shot the following operations (avoiding the use of temporary arrays):

    vxdp  = v_pd1.dot(xmb.T)
    vxdp  = vxdp.reshape(size,self.norbs, self.norbs)
    xvx2 = np.swapaxes(vxdp,0,1)
    xvx2 = xvx2.reshape(self.norbs,-1)
    xvx2 = xna.dot(xvx2)

    The array vxdp is pretty larger: (size*norbs, norbs) and pretty dense (0.1%)
    It is better to avoid its allocation for large systems.
    """

    D = np.zeros((N3, N1*N2), dtype=B.dtype)

    for jp in range(N2):
        Ctmp = np.zeros((N1*N2), dtype=B.dtype)
        for ip in range(N1):
            i = ip*N2 + jp
            for kp in range(N2):
                ipp = ip*N2 + kp

                for ind in range(indptr[i], indptr[i+1]):
                    k = indices[ind]
                    Ctmp[ipp] += data[ind]*B[k, kp]

        for ix in range(N3):
            for jx in range(N1*N2):
                D[ix, jx] += X[ix, jp]*Ctmp[jx]

    return D

def dpcoo_step1(self, size, nfdp, spin):
    import sparse

    xmb = csr_matrix(self.mo_coeff[0, spin, :, :, 0]).T
    v_pd = self.v_dab_csr.reshape(nfdp*self.norbs, self.norbs)
    vxdp = v_pd.dot(xmb).tocoo()
    return sparse.COO.from_scipy_sparse(vxdp)

def dpcoo_step2(vxdp, size, norbs):

    xxv2 = vxdp.reshape((size, norbs, norbs))
    xxv2 = xxv2.transpose(axes=(1, 0, 2))
    xxv2 = xxv2.reshape((norbs, size*norbs))
    return xxv2

def dpcoo_step3(self, vxdp, size, spin):
    import sparse

    xna = sparse.COO.from_numpy(self.mo_coeff[0, spin, self.nn[spin], :, 0])
    xxv2 = xna.dot(vxdp).reshape((len(self.nn[spin]), size, self.norbs))
    return xxv2.transpose(axes=(0, 2, 1))

def dpcoo_step4(cc_da, vxdp):
    import sparse
    return vxdp.dot(sparse.COO.from_scipy_sparse(cc_da))
