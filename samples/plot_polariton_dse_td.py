import os, sys
import numpy as np

from wavefunction_analysis.utils import print_matrix
from wavefunction_analysis.utils.read_files import read_number
from wavefunction_analysis.utils import convert_units
from wavefunction_analysis.plot import plt, add_colorbar_map, gradient_color_line

if __name__ == '__main__':
    molecule = 'h2'
    if len(sys.argv) > 1: molecule = sys.argv[1]
    file_name = molecule+'.polariton.psi4.txt'
    fig_name = molecule+'_qed_dse_td.png'

    #fig = plt.figure(figsize=(4, 3), dpi=300)
    fig, axs = plt.subplots(3, 1, figsize=(5, 5), dpi=300,
            sharex=True, layout='constrained', height_ratios=[2, 2,1.5])

    coupling = read_number(file_name, 'polariton_tddft', n=0, dtype=str)
    for i, s in enumerate(coupling):
        coupling[i] = float(s.split('_')[-1].split('.out')[0]) * .005
    coupling = np.reshape(coupling, (5, -1))[0]

    dse  = read_number(file_name, 'total: ', 1, -1, float)
    ex_e = read_number(file_name, 'ex   ', 3, dtype=float)
    ex_e2 = read_number(file_name, 'ex   ', 4, dtype=float)
    ex_p = read_number(file_name, 'ex   ', 5, dtype=float)

    n = len(dse)
    dse = np.array(dse)
    ex_e = np.reshape(ex_e, (n, -1))
    ex_e2 = np.reshape(ex_e2, (n, -1))
    ex_p = np.reshape(ex_p, (n, -1))

    arg = np.argsort(coupling)
    coupling, dse, ex_e, ex_p = coupling[arg], dse[arg], ex_e[arg], ex_p[arg]
    ex_e2 = ex_e2[arg]

    dse = convert_units(dse, 'hartree', 'ev')
    ex_e = convert_units(ex_e, 'hartree', 'ev')
    ex_e2 = convert_units(ex_e2, 'hartree', 'ev')

    frequency = ex_e[0,0]
    ex_e -= frequency
    ex_p[0,0], ex_p[0,1] = .5, .5 # average photon for no-coupling states


    cm = add_colorbar_map(plt, axs[0], posx=.1)
    #axs[0].hlines(frequency, coupling[0], coupling[-1], color='red', ls='--')
    for i in range(ex_e.shape[1]):
        line_segments = gradient_color_line(axs[0], coupling, ex_e[:,i], ex_p[:,i], cm, label='S$_'+str(i+1)+'$')
    axs[0].grid(visible=True, axis='y', ls='--', lw=1, color='gray', alpha=.4)
    #axs[0].legend(loc='upper left', fontsize='small')
    #axs[0].set_xlim([np.min(coupling), np.max(coupling)])
    axs[0].set_ylim([np.min(ex_e)-.5, np.max(ex_e)+.5])
    axs[0].set_xlabel('coupling $\lambda$ (au)')
    axs[0].set_ylabel('excitation energy (eV)')

    cm = add_colorbar_map(plt, axs[1], posx=.1)
    for i in range(ex_e2.shape[1]):
        line_segments = gradient_color_line(axs[1], coupling, ex_e2[:,i], ex_p[:,i], cm, label='S$_'+str(i+1)+'$')
    axs[1].grid(visible=True, axis='y', ls='--', lw=1, color='gray', alpha=.4)
    axs[1].set_ylim([np.min(ex_e2)-.5, np.max(ex_e2)+.5])
    axs[1].set_xlabel('coupling $\lambda$ (au)')
    axs[1].set_ylabel('total energy (eV)')

    axs[2].plot(coupling, dse, color='black')
    axs[2].set_ylabel('gs dse (eV)')

    #plt.tight_layout()
    plt.savefig(fig_name)
