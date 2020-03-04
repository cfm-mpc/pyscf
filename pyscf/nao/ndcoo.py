from __future__ import print_function, division
import numpy as np
from scipy.sparse import coo_matrix
from timeit import default_timer as timer
import sparse
import numba as nb

def combine_matrices(mat_list, shape):
    """
    Combine together a list of multidimensional sparse matrices
    """

    ndim = mat_list[0]["mat"].coords.shape[0]
    nnz = 0
    for mat in mat_list:
        nnz += mat["mat"].nnz

    coords = np.zeros((ndim, nnz), dtype=np.int32)
    data = np.zeros((nnz), dtype=mat_list[0]["mat"].dtype)
    st_nnz = 0
    for mat in mat_list:
        st_idx = mat["st_idx"]

        for idim, st in enumerate(mat["st_idx"]):
            coords[idim, st_nnz:st_nnz+mat["mat"].nnz] = \
                    mat["mat"].coords[idim, :] + st

        data[st_nnz:st_nnz+mat["mat"].nnz] = mat["mat"].data
        st_nnz += mat["mat"].nnz

    return sparse.COO(coords, data, shape=shape)

def merge_COO_matrix(mat1, mat2, st_idx_mat2):
    """
    Merge two COO matrix
    the coords of mat2 are shifted with st_idx_mat2

    if the index is already present in mat1, the value is replaced by the
    one in mat2
    """

    coords1 = mat1.coords
    coords2 = mat2.coords

    ndim = coords1.shape[0]
    for idim, st in enumerate(st_idx_mat2):
        coords2[idim, :] += st

        if coords2[idim, :].max() >= mat1.shape[idim]:
            mess = """
            index {} > {} for dim {}
            st_index = [{}, {}, {}]
            """.format(coords2[idim, :].max(), mat1.shape[idim], idim,
                    st_idx_mat2[0], st_idx_mat2[1], st_idx_mat2[2])
            raise ValueError(mess)

    nnz2add = count_nnz_toadd(coords1, coords2)
    nnz = mat1.nnz + nnz2add

    index2add = get_index_and_replace(coords1, mat1.data,
                                      coords2, mat2.data, nnz2add)

    if index2add.size < 1:
        return mat1
    
    coords, data = add_new_values(ndim, nnz, mat1.coords, mat1.data, mat2.coords,
                                  mat2.data, index2add)
    
    return sparse.COO(coords, data, shape=mat1.shape)

@nb.jit(nopython=True)
def add_new_values(ndim, nnz, coords1, data1, coords2, data2, index2add):

    nnz1 = data1.size
    coords = np.zeros((ndim, nnz), dtype=np.int32)
    data = np.zeros((nnz), dtype=data1.dtype)

    coords[:, 0:nnz1] = coords1
    data[0:nnz1] = data1

    innz = data1.size
    for idx in index2add:

        for idim in range(ndim):
            coords[idim, innz] = coords2[idim, idx]

        data[innz] = data2[idx]
        innz += 1
    
    return coords, data


@nb.jit(nopython=True)
def count_nnz_toadd(coords1, coords2):

    nnz = 0
    for i2, j2, k2 in zip(coords2[0, :], coords2[1, :], coords2[2, :]):

        toadd = True
        for i1, j1, k1 in zip(coords1[0, :], coords1[1, :], coords1[2, :]):

            # if data in mat1, replace value
            if i1 == i2 and j1 == j2 and k1 == k2:
                toadd = False
                break

        # if data not in mat 1, must be added
        if toadd:
            nnz += 1

    return nnz

@nb.jit(nopython=True)
def get_index_and_replace(coords1, data1, coords2, data2, nnz2add):

    index2add = np.zeros((nnz2add), dtype=np.int32)
    innz = 0
    index2 = 0
    for i2, j2, k2, val2 in zip(coords2[0, :], coords2[1, :], coords2[2, :], data2):

        toadd = True
        index1 = 0
        for i1, j1, k1 in zip(coords1[0, :], coords1[1, :], coords1[2, :]):

            # if data in mat1, replace value
            if i1 == i2 and j1 == j2 and k1 == k2:
                data1[index1] = val2
                toadd = False
                break

            index1 += 1

        # if data not in mat 1, must be added
        if toadd:
            index2add[innz] = index2
            innz += 1
        index2 += 1

    return index2add

#
#
#
class ndcoo():
  '''
  '''
  def __init__(self, inp, **kw):
    '''
    inp: (data, (i1,i2,i3)) when i is the indeces based on the shape of 3D matrix
    '''
    self.data = inp[0]
    self.ind = inp[1]
    self.shape = (self.ind[0].max()+1, self.ind[1].max()+1, self.ind[2].max()+1)
    self.ndim = len(self.ind)
    self.dtype = np.float64
  
  def tocoo_pa_b(self, descr):
    '''converts shape of sparse matrix () into (p*a, b)'''
    s = self.shape
    return coo_matrix((self.data, (self.ind[0]+self.ind[1]*s[0], self.ind[2])) )

  def tocoo_p_ab(self, descr):
    '''converts shape of sparse matrix () into (p, a*b)'''
    s = self.shape
    return coo_matrix((self.data, (self.ind[0], self.ind[2]+self.ind[1]*s[2])) )

  def tocoo_a_pb(self, descr):
    '''converts shape of sparse matrix () into (a, p*b)'''
    s = self.shape
    return coo_matrix((self.data, (self.ind[1], self.ind[0]+self.ind[2]*s[0])) )      

  def tocoo_b_pa(self, descr):
    '''converts shape of sparse matrix () into (b, p*a)'''
    s = self.shape
    return coo_matrix((self.data, (self.ind[2], self.ind[0]+self.ind[1]*s[0])) )  

if __name__=='__main__':
  import numpy as np
  ref = np.random.rand(70,12,35)
  #ref = np.array([[[1, 2], [0, 0], [0, 0], [3, 4]],[[0, 0], [1, 0], [5 ,6], [0, 1]], [[0, 9], [7, 8], [0, 0], [10, 0]]])
  print('ref shape: ====>\t',ref.shape)
  data = ref.reshape(-1)  #collect all data as 1D
  i0,i1,i2 = np.mgrid[0:ref.shape[0],0:ref.shape[1],0:ref.shape[2] ].reshape((3,data.size)) #provides mesh for i1,... which are shapes of given matrix
  
  #print(data.shape, i0.shape)
  #print(data)
  #print(i0,i1,i2)
  nc = ndcoo((data, (i0, i1, i2)))        #gives inputs to class ndcoo



  m0 = nc.tocoo_pa_b('p,a,b->ap,b')                 #change to favorable shape (ap,b) in COO format
  print('reshaped and sparse matrix m0(pa,b): ====>\t',m0.shape)             #(ap,b)            
  m0 = nc.tocoo_pa_b('p,a,b->ap,b').toarray().reshape((nc.shape[1], nc.shape[0], nc.shape[2]))
  m0 = np.swapaxes(m0,0,1)
  print('m0 reshaped to 3D array (a ,p ,b)',m0.shape) 
  print('comparison between ref and m0: ====>\t ',np.allclose(m0, ref)) #decompressed, reshaped and swapped matrix m0 should be equal to ref
                                          

  m1 = nc.tocoo_p_ab('p,a,b->p,ba')
  print('reshaped and sparse matrix m1(p,ba): ====>\t',m1.shape) 
  m1 = nc.tocoo_p_ab('p,a,b->p,ba').toarray().reshape((nc.shape[0], nc.shape[1], nc.shape[2]))
  print('m1 reshaped to 3D array ( p, a, b)',m1.shape)
  print('comparison between ref and m1: ====>\t ', np.allclose(m1, ref))     #compressed and reshaped matrix should be equal to referance array



  m2 = nc.tocoo_a_pb('p,a,b->a,pb')
  print('reshaped and sparse matrix m2(a,pb): ====>\t',m2.shape) 
  m2 = nc.tocoo_a_pb('p,a,b->a,pb').toarray().reshape((nc.shape[1], nc.shape[2], nc.shape[0]))
  m2 = np.swapaxes(m2.T,1,2)
  print('m2 reshaped to 3D array (p ,a , b)',m2.shape)
  print('comparison between ref and m2: ====>\t ', np.allclose(m2, ref))


  m3 = nc.tocoo_b_pa('p,a,b->b,pa')
  print('reshaped and sparse matrix m3(b,pa): ====>\t',m3.shape) 
  m3 = nc.tocoo_b_pa('p,a,b->b,pa').toarray().reshape((nc.shape[2], nc.shape[1], nc.shape[0]))
  m3 = m3.T
  print('m3 reshaped to 3D array (p ,a , b)',m3.shape)
  print('comparison between ref and m3: ====>\t ', np.allclose(m3, ref))


    # Constructing a matrix using ijv format
    #row  = np.array([0, 3, 1, 0])
    #col  = np.array([0, 3, 1, 2])
    #data = np.array([4, 5, 7, 9])
    #coo_matrix((data, (row, col)), shape=(4, 4)).toarray()
    #array([[4, 0, 9, 0],
    #       [0, 7, 0, 0],
    #       [0, 0, 0, 0],
    #       [0, 0, 0, 5]])
