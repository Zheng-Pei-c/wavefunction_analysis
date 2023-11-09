import os, sys
import numpy as np

from wavefunction_analysis.utils import print_matrix
from wavefunction_analysis.utils.read_files import read_number
from wavefunction_analysis.utils import convert_units
from wavefunction_analysis.plot import plt, add_colorbar_map, gradient_color_line

if __name__ == '__main__':
    file_name = 'h2.polariton.psi4.txt'
    fig_name = 'h2_qed_dse_td.png'

    #fig = plt.figure(figsize=(4, 3), dpi=300)
    fig, axs = plt.subplots(2, 1, figsize=(5, 4), dpi=300,
            sharex=True, layout='constrained', height_ratios=[2,1.5])

    frequency = 0.42978
    frequency = convert_units(frequency, 'hartree', 'ev')

    dse  = read_number(file_name, 'total: ', 1, -1, float)
    ex_e = read_number(file_name, 'ex   ', 3, dtype=float)
    ex_p = read_number(file_name, 'ex   ', 5, dtype=float)

    n = len(dse)
    dse = np.array(dse)
    ex_e = np.reshape(ex_e, (n, -1))
    ex_p = np.reshape(ex_p, (n, -1))

    dse = convert_units(dse, 'hartree', 'ev')
    ex_e = convert_units(ex_e, 'hartree', 'ev')

    coupling = np.linspace(0.01, 0.1, 10)

    cm = add_colorbar_map(plt, axs[0], posx=.1)
    axs[0].hlines(frequency, coupling[0], coupling[-1], color='red', ls='--')
    for i in range(ex_e.shape[1]):
        line_segments = gradient_color_line(axs[0], coupling, ex_e[:,i], ex_p[:,i], cm, label='S$_'+str(i+1)+'$')
    axs[0].legend()
    axs[0].set_ylim([np.min(ex_e)-.5, np.max(ex_e)+.5])
    axs[0].set_xlabel('coupling $\lambda$ (au)')
    axs[0].set_ylabel('excitation energy (eV)')

    axs[1].plot(coupling, dse, color='black')
    axs[1].set_ylabel('gs dse (eV)')

    #plt.tight_layout()
    plt.savefig(fig_name)
