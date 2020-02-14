/* Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
  
   Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
 
        http://www.apache.org/licenses/LICENSE-2.0
 
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
 */

#include<stdio.h>
#include<stdlib.h>
#include <omp.h>

/*
 * Count the number of non zeros element of the matrx C = AxB
 *   where A is a sparse matrix in CSR format
 *         B is a dense matrix
 *         C is gonna to be a sparse matrix
 *
 * Inputs:
 *   int nrow: number row A matrix
 *   int ncol2: number column of matrix B
 *   int *indptr: pinter index of matrix A
 *
 * Outputs:
 *   int nnz: number of non zeros element of the matrix C
 *
 */
extern "C" int count_nnz_spmat_denmat(int nrow, int ncol2, int *indptr)
{

  int nnz = 0;
  int i = 0, st = 0, fn = 0;

  for (i = 0; i < nrow; i++) {
  
    st = indptr[i];
    fn = indptr[i+1];

    if (st != fn) {
      nnz += ncol2;
    }
  }

  return nnz;
}
/*
 *  axpy operation
 *
 *    y = a*x + y
 *
 */
void saxpy(int size, float a, float *x, float *y)
{
  int i = 0;

  for (i = 0; i < size; i++) {
    y[i] += a*x[i];
  }

}

void daxpy(int size, double a, double *x, double *y)
{
  int i = 0;

  for (i = 0; i < size; i++) {
    y[i] += a*x[i];
  }

}

/*
 * Performs the inner multiplication in matrix matrix multiplication
 *
 */
void sinner_mult(int st, int fn, int *indices, float *data, int nrow_B, int ncol_B,
                 float *B, float *y)
{
  int ind;

  for (ind = st; ind < fn; ind++) {
    saxpy(ncol_B, data[ind], &B[indices[ind]*ncol_B], y);
  
  }

}

void dinner_mult(int st, int fn, int *indices, double *data, int nrow_B, int ncol_B,
                 double *B, double *y)
{
  int ind;

  for (ind = st; ind < fn; ind++) {
    daxpy(ncol_B, data[ind], &B[indices[ind]*ncol_B], y);
  
  }
}

/*
 *  Perfrom a matrix matrix multiplication
 *    C = AxB
 *
 *  where
 *    A is a CSR sparse matrix
 *    B is a dense matrix
 *    C is a CSR sparse matrix
 *
 */
extern "C" void scsr_spmat_denmat(int nrow_A, int ncol_A, int nnz_A, int *indptr_A,
    int *indices_A, float *data_A, int nrow_B, int ncol_B, float *B,
    int *indptr_C, int *indices_C, float *data_C)
{

  int i, idx, st, fn, innz_st, innz_fn;
  int col_index[ncol_B];

  for (i = 0; i < ncol_B; i++) {
    col_index[i] = i;
  }

  innz_st = 0;
  for(i = 0; i < nrow_A; i++){
  
    st = indptr_A[i];
    fn = indptr_A[i+1];

    if (st == fn) {
      indptr_C[i+1] = indptr_C[i];
      continue;
    }

    innz_fn = innz_st + ncol_B;

    sinner_mult(st, fn, indices_A, data_A, nrow_B, ncol_B, B, &data_C[innz_st]);

    // fill indices
    for (idx = 0; idx < ncol_B; idx++) {
      indices_C[innz_st+idx] = col_index[idx];
    }

    // get indptr
    indptr_C[i+1] = indptr_C[i] + ncol_B;
    innz_st += ncol_B;
  
  }
}
extern "C" void dcsr_spmat_denmat(int nrow_A, int ncol_A, int nnz_A, int *indptr_A,
    int *indices_A, double *data_A, int nrow_B, int ncol_B, double *B,
    int *indptr_C, int *indices_C, double *data_C)
{

  int i, idx, st, fn, innz_st, innz_fn;
  int col_index[ncol_B];

  // the indices for the row of C
  for (i = 0; i < ncol_B; i++) {
    col_index[i] = i;
  }

  innz_st = 0;
  for(i = 0; i < nrow_A; i++){
  
    st = indptr_A[i];
    fn = indptr_A[i+1];

    if (st == fn) {
      indptr_C[i+1] = indptr_C[i];
      continue;
    }

    innz_fn = innz_st + ncol_B;

    dinner_mult(st, fn, indices_A, data_A, nrow_B, ncol_B, B, &data_C[innz_st]);

    // fill indices
    for (idx = 0; idx < ncol_B; idx++) {
      indices_C[innz_st+idx] = col_index[idx];
    }

    // get indptr
    indptr_C[i+1] = indptr_C[i] + ncol_B;
    innz_st += ncol_B;
  
  }
}
