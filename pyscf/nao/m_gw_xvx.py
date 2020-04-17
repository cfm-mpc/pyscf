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

        # xvx2 = xvx2.dot(self.cc_da)
        # Somehow dense.dot(sparse) uses a crazy amount of memory ...
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
    vxdp = np.swapaxes(vxdp,0,1)
    vxdp = vxdp.reshape(self.norbs,-1)
    xvx2 = xna.dot(xvx2)

    The array vxdp is pretty large: (nfdp*norbs, norbs) and quite dense (0.1%)
    It is better to avoid its use for large systems, even in sparse format.

    Inputs:
        indptr: pointer index of v_dab in csr format
        indices: column indices of v_dab in csr format
        data: non zeros element of v_dab
        B: transpose of self.mo_coeff[0, spin, :, :, 0]
        X: self.mo_coeff[0, spin, self.nn[spin], :, 0]
        N1: nfdp
        N2: norbs
        N3: len(nn[spin])

    Outputs:
        D: xxv2 store in dense format
    """

    D = np.zeros((N3, N1*N2), dtype=B.dtype)

    # performs at the same time:
    # the matrix matrix product vxdp = v_pd1.dot(xmb.T), where v_pd1 is in CSR format
    # the reshape operations and sweepaxes on vxdp
    # finally the matrix matrix product xvx2 = xna.dot(xvx2)
    for jp in range(N2):

        # vxdp = v_pd1.dot(xmb.T)
        # store only one row of vxdp at a time
        Ctmp = np.zeros((N1*N2), dtype=B.dtype)
        for ip in range(N1):
            i = ip*N2 + jp
            for kp in range(N2):
                ipp = ip*N2 + kp

                for ind in range(indptr[i], indptr[i+1]):
                    k = indices[ind]
                    Ctmp[ipp] += data[ind]*B[k, kp]

        # xvx2 = xna.dot(xvx2)
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
