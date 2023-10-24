import os, sys
import numpy as np

from wavefunction_analysis.utils.read_files import read_matrix, read_number
from wavefunction_analysis.utils.print_matrix import print_matrix
from wavefunction_analysis.plot import plt, colors, ticker, mcolors


if __name__ == '__main__':
    infile = sys.argv[1]
    nstates = 21
    nwidth = 6
    fig_name = infile[:17]+'_iteration_conv'

    iteration = read_number(infile, 'zheng icyc in davidson:', 4)[-1]
    #print('iteration:', iteration)
    e_iter = read_matrix(infile, 1, nstates, 'zheng davidson e:', nwidth=nwidth)
    residual_iter = read_matrix(infile, 1, nstates, 'zheng dx_norm:', nwidth=nwidth)
    e_iter, residual_iter = e_iter[-iteration:], residual_iter[-iteration:]
    print(e_iter.shape, residual_iter.shape)
    #print_matrix('e_iter:', e_iter)

    fig = plt.figure(figsize=(12, 5), dpi=300, layout='constrained')

    x = range(e_iter.shape[0])

    ax = plt.subplot(1, 2, 1)
    for n in range(nstates):
        ax.plot(x, e_iter[:,n], label=str(n+1))
    ax.set_xlabel('iteration')
    ax.set_ylabel('energy (a.u.)')
    ax.legend()

    ax = plt.subplot(1, 2, 2)
    for n in range(nstates):
        ax.plot(x, residual_iter[:,n], label=str(n+1))
    ax.set_xlabel('iteration')
    ax.set_ylabel('residual')
    ax.legend()


    plt.tight_layout()
    plt.savefig(fig_name+'.png')
