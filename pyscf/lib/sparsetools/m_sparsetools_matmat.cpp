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

//template <class I, class T>
extern "C" void dcsr_matmat(int nrow, int ncol, int nnz, int *Ap, int *Aj, 
    double *Ax, double *Xx, double *Yx)
{

  int i, jj;
  double sum = 0.0;

  # pragma omp parallel \
  shared (nrow, Yx, Ap, Ax, Xx, Aj) \
  private (i, jj, sum)
  {
    int nthreads = omp_get_num_threads();
    #pragma omp for schedule(dynamic, (nrow + 4*nthreads-1)/(4*nthreads))
    for(i = 0; i < nrow; i++){
      sum = Yx[i];
      for(jj = Ap[i]; jj < Ap[i+1]; jj++){
        sum += Ax[jj] * Xx[Aj[jj]];
      }
      Yx[i] = sum;
    }
  }
}
