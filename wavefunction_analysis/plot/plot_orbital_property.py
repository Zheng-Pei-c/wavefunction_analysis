import os, sys
import numpy as np

from properties.orbital_properties import cal_properties_from_scratch

from plot import plt, colors, ticker, mcolors
from plot import brokenaxes

def plot_orbital_properties(fig_name, orbital_properties):
    fig = plt.figure(figsize=(12, 3), dpi=300, layout='constrained')
    nrow, ncol = 1, 3
    gs = fig.add_gridspec(nrow, ncol)
    #subfig = fig.subfigures(nrow, ncol, wspace=3, hspace=1)

    keys = ['energy', 'time', 'length', 'velocity']

    energy = orbital_properties[keys[0]]
    xlims = ((energy[0]-5, energy[1]+5), (energy[2]-2, energy[-1]+2))

    for i in range(nrow*ncol):
        if i == 2:
            ylims = ((8*1e6, 1.6*1e7), (2*1e4, 7*1e5))
            ratio = [[1,2],[1,2]]
            pad = [.1,.1]
        else:
            ylims = [1]
            ratio = [1,2]
            pad = .1

        axs = brokenaxes(None, gs[i], xlims, ylims, ratio=ratio, pad=pad)

        for ax in axs:
            ax.plot(orbital_properties[keys[0]], orbital_properties[keys[i+1]],
                    marker='.', mfc='r', mec='r')
            if i==2:
                ax.ticklabel_format(axis='y', style='sci', scilimits=(4,1))

            ax.set_xlabel(keys[0]+'('+orbital_properties['units'][keys[0]]+')')
        axs[0].set_ylabel(keys[i+1]+'('+orbital_properties['units'][keys[i+1]]+')')
        if i==2:
            axs[2].set_ylabel(keys[i+1]+'('+orbital_properties['units'][keys[i+1]]+')')

#        subfig[i].supxlabel(keys[0]+'('+orbital_properties['units'][keys[0]]+')', y=0)
#        subfig[i].supylabel(keys[i+1]+'('+orbital_properties['units'][keys[i+1]]+')')

    plt.tight_layout()
    plt.savefig(fig_name)


if __name__ == '__main__':
    infile = sys.argv[1]
    orbital_properties = cal_properties_from_scratch(infile)

    fig_name = infile.split('.in')[0]
    plot_orbital_properties(fig_name, orbital_properties)
