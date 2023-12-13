import os, sys
import numpy as np

from wavefunction_analysis.plot import plt
from wavefunction_analysis.dynamics.molecular_dynamics import *
from wavefunction_analysis.utils import convert_units
from wavefunction_analysis.utils.sec_mole import read_symbols_coords

def plot_time_variables(md, ax, idx=0, idy=0):
    total_time = md.total_time
    nuclear_nsteps = md.nuclear_nsteps
    time_line = np.linspace(0, total_time, nuclear_nsteps) * FS

    dists = md.md_time_coordinates[:,1,-1] - md.md_time_coordinates[:,0,-1]
    energies = md.md_time_total_energies

    dists = np.array(dists) * BOHR
    energies = np.array(energies) - energies[0]

    #ax[0].plot(time_line, dists)
    #if idx == 0:
    #    ax[0].set_ylabel('He--H$^+$ Length ($\AA$)')

    ax.plot(time_line, energies)
    #if idy > 0:
    #    energies = md.md_time_total_energies2
    #    energies = np.array(energies) - energies[0]
    #    ax.plot(time_line, energies, label='real')
    #    ax.legend()

    if idy == 2:
        ax.set_xlabel('Time (fs)')
    if idx == 0:
        ax.set_ylabel('Energy (a.u.)')



if __name__ == '__main__':
    infile = sys.argv[1]
    molecule = infile.split('.')[0]
    symbols, coords = read_symbols_coords(infile)

    key = {}
    key['functional'] = 'hf'
    key['basis'] = sys.argv[2] #'3-21g' #'6-31+g*'#
    key['charge'] = 0
    #key['atmsym'] = [1, 2]
    #key['init_coords'] = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.929352]]
    #key['init_velocity'] = [[0.0, 0.0, 0.0008], [0.0, 0.0, -0.0008]]
    key['atmsym'] = symbols
    key['init_coords'] = coords
    #key['init_velocity'] = 'random'
    key['init_velocity'] = 'thermo_298'

    key['nuclear_dt'] = 10
    key['total_time'] = 8000
    key['nuclear_update_method'] = 'velocity_verlet'
    key['ortho_method'] = 'lowdin' # 'cholesky'

    methods = ['normal', 'extended_lag', 'extended_lag']
    times = [2, 4, 8, 16]
    fig, ax = plt.subplots(nrows=len(methods), ncols=len(times), figsize=(16,12), sharex=True, dpi=300)
    for j, method in enumerate(methods):
        key['ed_method'] = method
        if j==2: key['xl_nk'] = 9
        for i, t in enumerate(times):
            key['nuclear_dt'] = t

            md = MolecularDynamics(key)
            md.run_dynamics()
            plot_time_variables(md, ax[j,i], i, j)
            ax[0,i].title.set_text('dt=%2dau (%.3ffs)' % (t, convert_units(t, 'au', 'fs')))
    plt.tight_layout()
    #plt.savefig(molecule+'_extended_lag_md')
    plt.savefig(molecule+'_md_'+key['basis'])
