import numpy as np
import sys
import math
import re

"""
If called from a G0W0 calculation as argument, this script will plot the quasi-particle energies as spectra by means of broadening.
"""

class GridSpec(object): #pylint:disable=R0903
    """a simple wrapper for a linear gridspec"""

    def __init__(self, minval, maxval, spacing):
        self._min = minval
        self._max = maxval
        self._space = spacing

    def __iter__(self):
        return self._griditerator()

    def _griditerator(self):
        """an iterator over the gridpoints contained in this GridSpec"""
        pos = self._min
        while pos <= self._max:
            yield pos
            pos += self._space

def read_qp_energies_pyscf(mf):
    """
    returns a list of the QP-energies from Pyscf-NAO
    """
    if hasattr(mf,'mo_energy_gw'):
        states=[]
        for s in range(mf.nspin):
            state = list(mf.mo_energy_gw[0][s]*HARTREE2EV)
            states.append(state)
        return sorted(states)
    else:
        msg = 'G0W0 calculation has not yet done!'
        print("\x1b[31mFATAL ERROR:\x1b[0m", msg, "\x1b[0m")
        sys.exit(1)


def make_spectrum(gridspec, states, broadening=0.04):
    """generate a spectrum by broadening the given states on the given grid"""
    x=[]
    y=[]
    with open('spectra.txt','w') as out_file:
        for point in gridspec:
            value = 0.0
            for peak in states:
                value += math.exp(-1.0/broadening * (point - peak)**2)
            print(point, value)
            out_file.write("%14.10f %14.10f \n" % (point, value))
            x.append(point)
            y.append(value)
    return x,y


def plot_spectra (x, y, states, msg=None):
    import matplotlib.pyplot as plt
    from matplotlib import rc
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.plot(x,y)
    plt.fill_between(x, 0, y, facecolor='c', alpha=0.2)
    for i in range(len(states)):
        plt.vlines(states[i], ymin=0, ymax=0.85, color='red',linewidth=1)
    plt.xlim((-20, 10))
    plt.title(r'$G_0W_0$@HF'+msg, fontsize=20)
    plt.xlabel('Energy (eV)', fontsize=15) 
    #plt.xlim((-10, 4))
    plt.yticks([])
    plt.tight_layout()
    #plt.savefig("spectra.svg", dpi=900)
    plt.show()


#
# Example of plotting the guassian spectra with broadening in eigenvalues.
#
if __name__=='__main__':
    from pyscf import gto, scf
    from pyscf.nao import gw as gw_c
    HARTREE2EV=27.2114
    mol = gto.M( verbose = 0, atom = 'O 0.0, 0.0, 0.622978 ; O 0.0, 0.0, -0.622978',basis = 'cc-pvdz', spin=2, charge=0)
    mf = scf.UHF(mol)
    mf.kernel()
    gw = gw_c(mf=mf, gto=mol, verbosity=3, niter_max_ev=1, kmat_algo='sm0_sum')
    gw.kernel_gw()     #gives G0W0 spectra

    #states= gw.mo_energy_gw[0, 0]*HARTREE2EV
    states=read_qp_energies_pyscf(gw)
    grid = GridSpec(-20, 10, 0.001)
    for i in range (len(states)):  
        x,y = make_spectrum(grid, states[i])
        msg= ' for spin {}'.format(i+1)
        plot_spectra (x,y, states[i], msg)
