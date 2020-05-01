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
    v = np.einsum('pab->apb', self.pb.get_ac_vertex_array(dtype=self.dtype))

    for spin in range(self.nspin):

        vx = v.dot(self.mo_coeff[0, spin, self.nn[spin], :, 0].T)
        xvx0 = calc_XVX(self.mo_coeff[0, spin, :, :, 0], vx, dtype=self.dtype)

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


    WARNING
    -------
    This method is experimental. For production run on large system, the
    dp_sparse method will be much more efficient in computational time and
    memory comsumption.
    """

    from pyscf.nao.m_rf0_den import calc_XVX

    try:
        import sparse
    except:
        mess = """
        Could not import the sparse package required by the ac_sparse method.
        See https://sparse.pydata.org/en/latest/
        """
        raise ImportError(mess)

    xvx = []

    if self.v_pab is None:
        t1 = timer()
        self.v_pab = self.pb.get_ac_vertex_array(matformat="sparse", dtype=self.dtype)
        t2 = timer()

        if self.verbosity>3:
            print("Get AC vertex timing: ", round(t2-t1,3))
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
    v_pd  = self.pb.get_dp_vertex_array(dtype=self.dtype)

    # atom_centered functional: C_{\widetilde{\mu}}^{\mu}
    # V_{\mu}^{ab} = V_{\widetilde{\mu}}^{ab} * C_{\widetilde{\mu}}^{\mu}
    c = self.pb.get_da2cc_den(dtype=self.dtype)

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

def gw_xvx_dp_ndcoo (self):

    from pyscf.nao import ndcoo
    xvx = []
    size = self.cc_da.shape[0]
    v_pd  = self.pb.get_dp_vertex_array(dtype=self.dtype)
    c = self.pb.get_da2cc_den(dtype=self.dtype)
    #First step
    data = v_pd.ravel() #v_pd.reshape(-1)
    #i0,i1,i2 = np.mgrid[0:v_pd.shape[0],0:v_pd.shape[1],0:v_pd.shape[2] ].reshape((3,data.size))   #fails in memory
    i0,i1,i2 = np.ogrid[0:v_pd.shape[0],0:v_pd.shape[1],0:v_pd.shape[2]]
    i00,i11,i22 = np.asarray(np.broadcast_arrays(i0,i1,i2)).reshape((3,data.size))

    nc = ndcoo((data, (i00, i11, i22)))
    m0 = nc.tocoo_pa_b('p,a,b->ap,b')

    for s in range(self.nspin):
        xna = self.mo_coeff[0,s,self.nn[s],:,0]
        xmb = self.mo_coeff[0,s,:,:,0]
        vx1 = m0*(xmb.T)
        #Second Step
        vx1 = vx1.reshape(size,self.norbs,self.norbs)   #shape (p,a,b)
        vx_ref = vx1.reshape(self.norbs,-1)             #shape (b,p*a)
        #data = vx1.ravel()
        #i00,i11,i22 = np.asarray(np.broadcast_arrays(i0,i1,i2)).reshape((3,data.size))
        #nc1 = ndcoo((data, (i00, i11, i22)))
        #m1 = nc1.tocoo_pa_b('p,a,b->ap,b')  
        #Third Step
        xvx3 = xna.dot(vx_ref)                               #xna(ns,a).V(a,p*b)=xvx(ns,p*b)
        xvx3 = xvx3.reshape(len(self.nn[s]),size,self.norbs) #xvx(ns,p,b)
        xvx3 = np.swapaxes(xvx3,1,2)                         #xvx(ns,b,p)
        xvx3 = xvx3.dot(c)                                #XVX=xvx.c
        xvx.append(xvx3)

    return xvx


if __name__=='__main__':
    from pyscf import gto, scf
    from pyscf.nao import gw_iter  
    from timeit import default_timer as timer 
    import numpy as np

    mol = gto.M(atom='''O 0.0, 0.0, 0.622978 ; O 0.0, 0.0, -0.622978''', basis='ccpvtz',spin=2)
    mf = scf.UHF(mol)
    mf.kernel()

    gw = gw_iter(mf=mf, gto=mol)
    
    timing = np.zeros(7)

    t1 = timer()
    ref = gw.gw_xvx(algo='simple')
    t2 = timer()
    timing[0] += round(t2-t1,3)
    
    t1 = timer()
    ac = gw.gw_xvx(algo='ac')
    t2 = timer()
    timing[1] += round(t2-t1,3)
    
    t1 = timer()
    ac_blas = gw.gw_xvx(algo='ac_blas')
    t2 = timer()
    timing[2] += round(t2-t1,3)
    
    t1 = timer()
    ac_sparse = gw.gw_xvx(algo='ac_sparse')
    t2 = timer()
    timing[3] += round(t2-t1,3)
    
    t1 = timer()
    dp= gw.gw_xvx(algo='dp')
    t2 = timer()
    timing[4] += round(t2-t1,3)
    
    t1 = timer()
    dp_coo= gw.gw_xvx(algo='dp_coo')
    t2 = timer()
    timing[5] += round(t2-t1,3)

    t1 = timer()
    dp_sparse = gw.gw_xvx(algo='dp_sparse')
    t2 = timer()
    timing[6] += round(t2-t1,3)

    for s in range(gw.nspin):
        print('Spin {}, atom-centered with ref                  : {}, timing {} sec'.format(s+1,
            np.allclose(ref[s], ac[s], atol=1e-15), timing[1]))
        print('Spin {}, atom-centered (BLAS) with ref           : {}, timing {} sec'.format(s+1,
            np.allclose(ref[s],ac_blas[s],atol=1e-15),timing[2]))
        print('Spin {}, sparse atom-centered with ref           : {}, timing {} sec'.format(s+1,
            np.allclose(ref[s], ac_sparse[s], atol=1e-15),timing[3]))
        print('Spin {}, dominant product with ref               : {}, timing {} sec'.format(s+1,
            np.allclose(ref[s], dp[s], atol=1e-15),timing[4]))
        print('Spin {}, sparse_dominant product-ndCoo with ref  : {}, timing {} sec'.format(s+1,
            np.allclose(ref[s], dp_coo[s], atol=1e-15),timing[5]))
        print('Spin {}, sparse_dominant product-numba with ref  : {}, timing {} sec'.format(s+1,
            np.allclose(ref[s], dp_sparse[s], atol=1e-15), timing[6]))
