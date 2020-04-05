#An HDF5 file is a container for two kinds of objects: 
#  * datasets (array-like collections of data)
#  * groups (folder-like containers that hold datasets).
# Groups work like dictionaries, and datasets work like NumPy arrays

from __future__ import division
import numpy as np

def read_rst_h5py (value, filename=None):
    import h5py ,os
    if filename is None: 
        msg = "No file to open"
        return None, msg
  
    try:
        fl = h5py.File(filename, 'r')
    except:
        msg = "could not open file {}".format(filename)
        return None, msg

    #print("Keys: %s" % f.keys())
    a_group_key = list(fl.keys())
    if value in a_group_key: 
        # Get the data
        data = list(fl[value])
        msg = 'RESTART: matrix elements of {} was read from {}'.format(value, filename)
        return data, msg


def write_rst_h5py(data, value, filename = None):
    import h5py
    if filename is None: 
      filename= 'RESTART.hdf5'   

    hf = h5py.File(filename, 'a')
    try:
        hf.create_dataset(value, data=data)
        msg = 'WRITE: matrix elements of {} stored in {}'.format(value, filename)

    except:
        msg = "failed writting data to {}".format(filename)
        print(type(data))

    hf.close
    return msg



def write_rst_yaml (data , filename=None):
    import yaml
    if filename is None: filename= 'SCREENED_COULOMB.yaml'
    with open(filename, 'w+', encoding='utf8') as outfile:
        yaml.dump(data, outfile, default_flow_style=False, allow_unicode=True)
    msg = 'Full matrix elements of screened interactions stored in {}'.format(filename)
    return msg


def read_rst_yaml (filename=None):
    import yaml, os
    if filename is None: 
        path = os.getcwd()
        filename =find('*.yaml', path)
    with open(filename, 'r') as stream:
        try:
            data = yaml.load(stream)
            msg = 'RESTART: Full matrix elements of screened interactions (W_c) was read from {}'.format(filename)
            return data, msg
        except yaml.YAMLError as exc:
            return exc

def find (pattern, path):
    import os, fnmatch
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(name))
    if (len(result) == 0 ):
        msg = 'There is no file for restarting!'
        return msg
    elif (len(result) > 1 ):
        msg = 'Which {} file should be read! There are several: {}'.format(pattern,result)
        return msg
    else:
        return result[0]


if __name__=='__main__':
    from pyscf import gto, scf
    from pyscf.nao import gw as gw_c
    import numpy as np
    mol = gto.M( verbose = 0, atom = 'O 0.0, 0.0, 0.622978 ; O 0.0, 0.0, -0.622978',basis = 'cc-pvdz', spin=2, charge=0)
    gto_mf = scf.UHF(mol)
    gto_mf.kernel()
    gw = gw_c(mf=gto_mf, gto=mol)
    k = gw.get_k()
    w = gw.get_snmw2sf()
    #print(write_rst_yaml (w))
    #data, msg = read_rst_yaml(filename='SCREENED_COULOMB.yaml') 
    #print(msg)
    #print(np.allclose(w[0], data[0]))
    print(write_rst_h5py (value='screened_interactions',data=w))
    data1, msg1 = read_rst_h5py(value='screened_interactions',filename= 'RESTART.hdf5')
    print(write_rst_h5py (value='fock', data=k))
    data2, msg2 = read_rst_h5py(value='fock',filename= 'RESTART.hdf5') 
    print(msg1)
    print(np.allclose(w[0], data1[0]))
    print(msg2)
    print(np.allclose(k[0], data2[0]))
