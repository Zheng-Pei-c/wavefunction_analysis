import os, sys
import numpy as np

from wavefunction_analysis.plot import plt
from wavefunction_analysis.dynamics.molecular_dynamics import *
from wavefunction_analysis.utils import convert_units

def plot_time_variables(md, ax, idx=0):
    total_time = md.total_time
    nuclear_nsteps = md.nuclear_nsteps
    time_line = np.linspace(0, total_time, nuclear_nsteps) * FS

    dists = md.md_time_coordinates[:,1,-1] - md.md_time_coordinates[:,0,-1]
    energies = md.md_time_total_energies

    dists = np.array(dists) * BOHR
    energies = np.array(energies) - energies[0]

    ax[0].plot(time_line, dists)
    if idx == 0:
        ax[0].set_ylabel('He--H$^+$ Length ($\AA$)')

    ax[1].plot(time_line, energies)
    ax[1].set_xlabel('Time (fs)')
    if idx == 0:
        ax[1].set_ylabel('Energy (a.u.)')



if __name__ == '__main__':
    key = {}
    key['functional'] = 'hf'
    #key['basis'] = '3-21g'
    #key['charge'] = 0
    #key['atmsym'] = [1, 1]
    #key['init_coords'] = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]]
    key['basis'] = 'sto-3g'
    key['charge'] = 1
    key['atmsym'] = [1, 2]
    key['init_coords'] = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.929352]]
    key['init_velocity'] = [[0.0, 0.0, 0.0008], [0.0, 0.0, -0.0008]]

    key['nuclear_dt'] = 10
    key['total_time'] = 4000
    key['nuclear_update_method'] = 'velocity_verlet'
    key['ortho_method'] = 'lowdin' # 'cholesky'

    #print('run extended_lag')

    #key['ed_method'] = 'extended_lag'

    times = [2, 4, 8, 16, 32]
    fig, ax = plt.subplots(nrows=2, ncols=len(times), figsize=(16,6), sharex=True)
    for i, t in enumerate(times):
        key['nuclear_dt'] = t

        md = MolecularDynamics(key)
        md.run_dynamics()
        plot_time_variables(md, ax[:,i], i)
        ax[0,i].title.set_text('dt=%2dau (%.3fs)' % (t, convert_units(t, 'au', 'fs')))
    plt.tight_layout()
    #plt.savefig('heh_extended_lag_md')
    plt.savefig('heh_md')
