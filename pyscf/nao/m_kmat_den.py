# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import print_function, division
import numpy as np
import scipy.linalg.blas as blas
from pyscf.nao.m_sparsetools import csr_matvec, csc_matvec

def kmat_den(mf, dm=None, algo=None, **kw):
    """
        Computes the matrix elements of Fock exchange operator
        Args:
        mf : (System Variables), this must have arrays of coordinates and species, etc
        Returns:
        matrix elements
    """

    pb, hk = mf.add_pb_hk(**kw)
    dm = mf.make_rdm1() if dm is None else dm

    n = mf.norbs
    if mf.nspin == 1:
        dm = dm.reshape((n,n))
    elif mf.nspin == 2:
        dm = dm.reshape((mf.nspin,n,n))
    else:
        print(nspin)
        raise RuntimeError('nspin>2?')


    algol = algo.lower() if algo is not None else 'sm0_sum'
    if mf.verbosity>1 and algol != 'sm0_sum':
        print(__name__, "\t====> Fock exchange algorithm '{}'.\f".format(algol))
    gen_spy_png = kw['gen_spy_png'] if 'gen_spy_png' in kw else False

    print(__name__, algol)
    if algol=='fci':

        mf.fci_den = abcd2v = mf.fci_den if hasattr(mf, 'fci_den') else pb.comp_fci_den(hk)
        kmat = np.einsum('abcd,...bc->...ad', abcd2v, dm)

    elif algol=='ac_vertex_fm':

        pab2v = pb.get_ac_vertex_array()
        pcd = np.einsum('pq,qcd->pcd', hk, pab2v)
        pac = np.einsum('pab,...bc->...pac', pab2v, dm)
        kmat = np.einsum('...pac,pcd->...ad', pac, pcd)

    elif algol=='dp_vertex_fm':

        dab2v = pb.get_dp_vertex_array()
        da2cc = pb.get_da2cc_den()
        dq2v  = np.einsum('dp,pq->dq', da2cc, hk)
        pcd   = np.einsum('dq,dab->qab', dq2v, dab2v)
        dac   = np.einsum('dab,...bc->...dac', dab2v, dm)
        pac   = np.einsum('...dac,dp->...pac', dac, da2cc)
        kmat  = np.einsum('...pac,pcd->...ad', pac, pcd)

    elif algol=='dp_vertex_loops_fm':

        dab2v = pb.get_dp_vertex_array()
        da2cc = pb.get_da2cc_den()
        dq2v  = einsum('dp,pq->dq', da2cc, hk)
        kmat  = np.zeros_like(dm)
        for d in range(n):
            pc2 = np.einsum('dq,da->qa', dq2v, dab2v[:,:,d])
            for a in range(n):
                dc  = np.einsum('db,...bc->...dc', dab2v[:,a,:], dm)
                pc1  = np.einsum('...dc,dp->...pc', dc, da2cc)
                kmat[...,a,d]  = np.einsum('...pc,pc->...', pc1, pc2)
            
    elif algol=='dp_vertex_loops_sm':
        """
        This algorithm uses some sparsity, but it has O(N^4) complexity
        because the pc1 and pc2 auxilaries are stored in the dense format.
        Moreover, pc1 and pc2 auxiliaries in atom-centered product basis generate
        rather dense matrices. Therefore, it is desirable to use dominant
        products to store/treat these auxiliaries.
        """

        dab2v = pb.get_dp_vertex_doubly_sparse(axis=1)
        da2cc = pb.get_da2cc_sparse().tocsr()
        qd2v  = (da2cc * hk).transpose()
        kmat  = np.zeros_like(dm)

        if len(dm.shape) == 3: # if spin index is present
          
            for d in range(n):
                #pc2 = einsum('dq,da->qa', dq2v, dab2v_den[:,:,d])
                #print(da2cc.shape, hk.shape, qd2v.shape, dab2v[d].shape)
                pc2 = (qd2v * dab2v[d])
                for s in range(mf.nspin):
                    for a in range(n):
                        #dc  = einsum('db,bc->dc', dab2v_den[:,a,:], dm[s])
                        dc  = (dab2v[a] * dm[s])
                        pc1 = (da2cc.T * dc)
                        kmat[s,a,d]  = (pc1*pc2).sum()
        elif len(dm.shape)==2: # if spin index is absent
            for d in range(n):
                pc2 = (qd2v * dab2v[d])
                for a in range(n):
                    dc  = (dab2v[a] * dm)
                    pc1 = (da2cc.T * dc)
                    kmat[a,d]  = (pc1*pc2).sum()
        else:
            print(dm.shape)
            raise RuntimeError('to impl dm.shape')

    elif algol == 'sm0_prd':
        import scipy.sparse as sparse
        if gen_spy_png: 
            import matplotlib.pyplot as plt
            plt.ioff()
          
        dab2v = pb.get_dp_vertex_doubly_sparse(axis=0)
        da2cc = pb.get_da2cc_sparse().tocsr()
        kmat  = np.zeros_like(dm)
        nnd,nnp = da2cc.shape

        if len(dm.shape)==3: # if spin index is present
            for s in range(mf.nspin):
                for mu,a_ap2v in enumerate(dab2v):
                    cc = da2cc[mu].toarray().reshape(nnp)
                    q2v = dot( cc, hk )
                    a_bp = sparse.csr_matrix(a_ap2v * dm[s])
                    for nu,bp_b2v in enumerate(dab2v):
                        q2cc = da2cc[nu].toarray().reshape(nnp)
                        v = (q2cc * q2v).sum()
                        ab2sigma = (a_bp * bp_b2v * v)
                        if ab2sigma.count_nonzero()>0:
                            kmat[s][ab2sigma.nonzero()] += ab2sigma.data
              
        elif len(dm.shape)==2: # if spin index is absent
            for mu,a_ap2v in enumerate(dab2v):
                cc = da2cc[mu].toarray().reshape(nnp)
                q2v = dot( cc, hk )
                a_bp = sparse.csr_matrix(a_ap2v * dm)

                if gen_spy_png:
                    plt.spy(a_bp.toarray())
                    fname = "spy-v-dm-{:06d}.png".format(mu); print(fname)
                    plt.savefig(fname, bbox_inches='tight'); plt.close()
                
                for nu,bp_b2v in enumerate(dab2v):
                    q2cc = da2cc[nu].toarray().reshape(nnp)
                    v = (q2cc * q2v).sum()
                    ab2sigma = (a_bp * bp_b2v * v)
                    if gen_spy_png:
                        plt.spy(ab2sigma.toarray())
                        fname = "spy-v-dm-v-{:06d}-{:06d}.png".format(mu,nu); print(fname)
                        plt.savefig(fname, bbox_inches='tight'); plt.close()

                    if ab2sigma.count_nonzero()>0 : kmat[ab2sigma.nonzero()] += ab2sigma.data
        else:
            print(dm.shape)
            raise RuntimeError('?dm.shape?')

    elif algol == 'sm0_sum':
        """ 
        This algorithm is using two sparse representations of the product 
        vertex V^ab_mu. The algorithm was not realized before and it seems
        to be superior to the algorithm sm0_prd (see above).
        """
        import scipy.sparse as sparse
        from timeit import default_timer as timer
        #t1 = timer()

        if hasattr(mf, 'dab2v'):
            dab2v = mf.dab2v
        else:
            mf.dab2v = dab2v =  pb.get_dp_vertex_doubly_sparse(axis=0)

        da2cc = mf.cc_da_csr
        kmat  = np.zeros_like(dm)
        (nnd,nnp),n = da2cc.shape,dm.shape[-1]

        if len(dm.shape) == 3:
            # if spin index is present

            for s in range(mf.nspin):
                for mu, a_ap2v in enumerate(dab2v):
                    cc = da2cc[mu].toarray().reshape(nnp)
                    q2v = np.dot( cc, hk )
                    nu2v = da2cc * q2v
                    a_bp2vd = sparse.csr_matrix(a_ap2v * dm[s])
                    bp_b2hv = sparse.csr_matrix((nu2v * mf.v_dab_csr).reshape((n,n)))
                    ab2vdhv = a_bp2vd * bp_b2hv
                    kmat[s][ab2vdhv.nonzero()] += ab2vdhv.data
            
        elif len(dm.shape) == 2:
            # if spin index is absent
            # [   14.59955484   314.09553769  1113.57001527   140.44369524   687.34180544   89.05969492    28.3846701 ]
            # C60: [ 15.85272067 288.39123102 153.34696097 578.67634735 128.22075884
            # 583.69870308 22.44895533]
          
            tt = np.zeros(8)
            ttt = np.zeros(7)

            for mu, a_ap2v in enumerate(dab2v):
                
                tt[0] = timer();
                # 1D; dense, could be sparse ... 
                cc = da2cc[mu].toarray().reshape(nnp)
                tt[1] = timer(); 

                # this could be done directly
                # Slower term for large systems
                # blas dgemv is much slower than np.dot
                # That is odd!!
                # q2v = blas.dgemv( 1.0, hk, cc, trans=1 )
                q2v = cc.dot(hk)
                tt[2] = timer();

                # slower term ..
                #nu2v = da2cc.dot(q2v)
                nu2v = csr_matvec(da2cc, q2v)
                tt[3] = timer();

                # a_ap2v: sparse
                a_bp2vd = sparse.coo_matrix(a_ap2v.dot(dm),
                                            dtype=a_ap2v.dtype).tocsr()
                tt[4] = timer();
                # bp_b2hv = sparse.csr_matrix((nu2v * dab2v_csr).reshape((n,n)))
                
                # slower term ..
                #bp_b2hv = dab2v_csr.T.dot(nu2v).reshape((n,n))
                # bp_b2hv = csc_matvec(dab2v_csr.T, nu2v).reshape((n,n))
                bp_b2hv = csr_matvec(mf.v_dab_trans, nu2v).reshape((n, n))
                tt[5] = timer();
                
                ab2vdhv = a_bp2vd.dot(bp_b2hv)
                tt[6] = timer()
                # kmat[ab2vdhv.nonzero()] += ab2vdhv.data

                kmat += ab2vdhv
                tt[7] = timer()
                ttt += tt[1:8]-tt[0:7]
                
            print(__name__, ttt)
              
        else:
            print(dm.shape)
            raise RuntimeError('?dm.shape?')

    else:
        
        print('algo=', algo)
        raise RuntimeError('unknown algorithm')

    return kmat
