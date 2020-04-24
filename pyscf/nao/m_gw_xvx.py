from __future__ import division
import numpy as np
from timeit import default_timer as timer

def gw_xvx_ac_simple(self):
    """
    Simple metod to calculate basis product using atom-centered product
    basis: V_{mu}^{ab}

    This method is used as reference.
    direct multiplication with np and einsum
    """

    xvx = []
    v_pab = self.pb.get_ac_vertex_array(matformat="dense", dtype=self.dtype)

    for spin in range(self.nspin):

        #(nstat, norbs)
        xna = self.mo_coeff[0, spin, self.nn[spin], :, 0]

        #(norbs,norbs)
        xmb = self.mo_coeff[0, spin, :, :, 0]

        # einsum: direct multiplication
        xvx_ref  = np.einsum('na,pab,mb->nmp', xna, v_pab, xmb)

        # direct multiplication by using np.dot and swapping between axis
        xvx_ref2 = np.swapaxes(xna.dot(v_pab.dot(xmb.T)), 1, 2)

        xvx.append(xvx_ref)

    return xvx

def gw_xvx_ac(self):
    """
    metod to calculate basis product using atom-centered product
    basis: V_{mu}^{ab}
    """

    xvx = []
    v_pab = self.pb.get_ac_vertex_array(matformat="dense", dtype=self.dtype)

    # First step: convert into a 2D array of shape (Nprod*norbs, norbs)
    v_pab1= v_pab.reshape(self.nprod*self.norbs, self.norbs)
    for spin in range(self.nspin):

        #(nstat, norbs)
        xna = self.mo_coeff[0, spin, self.nn[spin], :, 0]

        #(norbs,norbs)
        xmb = self.mo_coeff[0, spin, :, :, 0]

        vx  = v_pab1.dot(xmb.T)
        # reshape into initial 3D shape
        vx  = vx.reshape(self.nprod, self.norbs, self.norbs)

        # Second step: sweep axes and reshape into 2D array
        xvx1 = np.swapaxes(vx, 0, 1)
        xvx1 = xvx1.reshape(self.norbs, -1)

        # Third step
        xvx1 = xna.dot(xvx1)
        xvx1 = xvx1.reshape(len(self.nn[spin]), self.nprod, self.norbs)
        xvx1 = np.swapaxes(xvx1, 1, 2)

        xvx.append(xvx1)

    return xvx

def gw_xvx_ac_blas(self):
    """
    Metod to calculate basis product using atom-centered product
    basis: V_{mu}^{ab}

    Use Blas to handle matrix-matrix multiplications
    """

    from pyscf.nao.m_rf0_den import calc_XVX

    xvx = []
    v = np.einsum('pab->apb', self.pb.get_ac_vertex_array())

    for spin in range(self.nspin):

        vx = v.dot(self.mo_coeff[0, spin, self.nn[spin], :, 0].T)
        xvx0 = calc_XVX(self.mo_coeff[0, spin, :, :, 0], vx)

        xvx.append(xvx0.T)

    return xvx

def gw_xvx_ac_sparse(self):
    """
    Metod to calculate basis product using atom-centered product
    basis: V_{mu}^{ab}

    Use a sparse version of the atom-centered product, allow calculations
    of larger systems, however, the computational cost to get the sparse
    version of the atom-centered product is very high (up to 30% of the full
    simulation time of a GW0 run).
    """

    from pyscf.nao.m_rf0_den import calc_XVX

    xvx = []

    if self.v_pab is None:
        t1 = timer()
        self.v_pab = self.pb.get_ac_vertex_array(matformat="sparse", dtype=self.dtype)
        t2 = timer()

        if self.verbosity>3:
            print("Get AC vertex timing: ", t2-t1)
            print("Vpab.shape: ", self.v_pab.shape)
            print("Vpab.nnz: ", self.v_pab.nnz)

    v = self.v_pab.transpose(axes=(1, 0, 2))
    for spin in range(self.nspin):

        vx = v.dot(self.mo_coeff[0, spin, self.nn[spin], :, 0].T)
        xvx0 = calc_XVX(self.mo_coeff[0, spin, :, :, 0], vx)

        xvx.append(xvx0.T)

    return xvx

def gw_xvx_dp(self):
    """
    Metod to calculate basis product using dominant product basis V_{mu}^{ab}
    """

    xvx = []
    size = self.cc_da.shape[0]

    # dominant product basis: V_{\widetilde{\mu}}^{ab}
    v_pd  = self.pb.get_dp_vertex_array()

    # atom_centered functional: C_{\widetilde{\mu}}^{\mu}
    # V_{\mu}^{ab} = V_{\widetilde{\mu}}^{ab} * C_{\widetilde{\mu}}^{\mu}
    c = self.pb.get_da2cc_den()

    # First step: transform v_pd in a 2D array
    v_pd1 = v_pd.reshape(v_pd.shape[0]*self.norbs, self.norbs)
    for spin in range(self.nspin):

        #(nstat, norbs)
        xna = self.mo_coeff[0, spin, self.nn[spin], :, 0]

        #(norbs,norbs)
        xmb = self.mo_coeff[0, spin, :, :, 0]

        vxdp  = v_pd1.dot(xmb.T)

        # Second step
        vxdp  = vxdp.reshape(size, self.norbs, self.norbs)
        xvx2 = np.swapaxes(vxdp, 0, 1)
        xvx2 = xvx2.reshape(self.norbs, -1)

        # Third step
        xvx2 = xna.dot(xvx2)
        xvx2 = xvx2.reshape(len(self.nn[spin]), size, self.norbs)
        xvx2 = np.swapaxes(xvx2, 1, 2)

        xvx2 = xvx2.dot(c)

        xvx.append(xvx2)

    return xvx

def gw_xvx_dp_sparse(self):
    """
    Method to calculate basis product using dominant product basis V_{mu}^{ab}

    Take advantages of the sparsity of the dominant product basis.
    Numba library must be installed
    """

    from pyscf.nao.m_gw_xvx_dp_sparse import gw_xvx_sparse_dp

    try:
        import numba as nb
    except:
        raise ValueError("Numba must be install to use gw_xvx dp_sparse method")

    if self.verbosity > 3:
        # dominant product basis: V_{\widetilde{\mu}}^{ab}
        print("V_dab.shape: ", self.v_dab.shape)
        print("V_dab.nnz: ", self.v_dab.nnz)

        # atom_centered functional: C_{\widetilde{\mu}}^{\mu}
        # V_{\mu}^{ab}= V_{\widetilde{\mu}}^{ab} * C_{\widetilde{\mu}}^{\mu}
        print("cc_da.shape: ", self.cc_da.shape)
        print("cc_da.nnz: ", self.cc_da.nnz)

    return gw_xvx_sparse_dp(self)
