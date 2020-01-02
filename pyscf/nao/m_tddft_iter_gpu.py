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
import warnings
import numpy as np
from ctypes import POINTER, c_double, c_int, c_int64, c_float, c_int, c_long
from pyscf.nao.m_gpu_base import gpu_initialize
import sys

try: # to import gpu library
  from pyscf.lib import misc
  libnao_gpu = misc.load_library("libnao_gpu")
  GPU_import = True
except:
  GPU_import = False

class tddft_iter_gpu_c(gpu_initialize):

    def __init__(self, GPU, X4, ksn2f, ksn2e, norbs, nfermi, nprod, vstart):
        """
          Input Parameters:
          -----------------
            GPU: variable to set up GPU calculations. It can take several forms
                * None : if GPU=None, no GPU will be use
                * True : if GPU is True, then calculation will be using GPUs with 
                      default setup
                * False : Identic to None
                * dictionary: a dictionary containing the different parameters
                    for the gpu setup, the keys are,
                      * use, booleean to know if wew will use GPU calculations
                      * device: integer to use a certain GPU if there is more than one
        """

        warnings.warn("GPU seems to give issue with the mat-mat products, results may not be trusted")
        gpu_initialize.__init__(self, GPU, norbs, nfermi, nprod, vstart)
        
        if self.GPU is not None:

              libnao_gpu.init_tddft_iter_gpu(
                          X4.ctypes.data_as(POINTER(c_float)), c_int(self.norbs),
                          ksn2e[0, 0, :].ctypes.data_as(POINTER(c_float)), 
                          ksn2f[0, 0, :].ctypes.data_as(POINTER(c_float)),
                          c_int(self.nfermi), c_int(self.nprod), c_int(self.vstart))

    def calc_nb2v_from_sab(self, reim):
        libnao_gpu.calc_nb2v_from_sab(c_int(reim))

    def calc_nm2v_real(self):
        libnao_gpu.get_nm2v_real()
    
    def calc_nm2v_imag(self):
        libnao_gpu.get_nm2v_imag()

    def calc_nb2v_from_nm2v_real(self):
        libnao_gpu.calc_nb2v_from_nm2v_real()

    def calc_nb2v_from_nm2v_imag(self):
        libnao_gpu.calc_nb2v_from_nm2v_imag()

    def calc_sab(self, reim):
        libnao_gpu.get_sab(c_int(reim))

    def div_eigenenergy_gpu(self, comega):
        libnao_gpu.div_eigenenergy_gpu(c_double(comega.real), c_double(comega.imag),
                self.block_size.ctypes.data_as(POINTER(c_int)),
                self.grid_size.ctypes.data_as(POINTER(c_int)))
