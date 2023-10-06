import os, sys
import numpy as np

from wavefunction_analysis.plot import plt, colors, ticker, mcolors

def get_kwargs(marker):
    return dict(marker=marker, markersize=10,
                linestyle='none', color='k', mec='k', mew=1, clip_on=False)


def brokenaxes(fig, gs, xlims, ylims, ratio=[1.,1.], pad=.1):
    func = fig.add_gridspec if gs==None else gs.subgridspec
    nr, nc = len(ylims), len(xlims)

    if nr == 1:
        gs = func(nr, nc, width_ratios=ratio, wspace=pad)
    elif nc == 1:
        gs = func(nr, nc, height_ratios=ratio, hspace=pad)
    else:
        gs = func(nr, nc, width_ratios=ratio[0], height_ratios=ratio[1],
                              hspace=pad[0], wspace=pad[1])

    axs = gs.subplots()

    if nr == 1:
        for i in range(nc):
            axs[i].set_xlim(xlims[i])
            if i>0: axs[i].set_yticks([])
            #axs[i].tick_params(labelright=False)
        #axs[-1].yaxis.tick_right()
    elif nc == 1:
        for i in range(nr):
            axs[i].set_ylim(ylims[i])
            if i<nr-1: axs[i].set_xticks([])
    else:
        for i in range(nr):     # y
            for j in range(nc): # x
                axs[i,j].set_xlim(xlims[j])
                axs[i,j].set_ylim(ylims[i])
                if j>0: axs[i,j].set_yticks([])
                if i<nr-1: axs[i,j].set_xticks([])

    for ax in axs.flat:
        ss = ax.get_subplotspec()
        if nr > 1:
            ax.spines['top'].set_visible(ss.is_first_row())
            ax.spines['bottom'].set_visible(ss.is_last_row())
        if nc > 1:
            ax.spines['left'].set_visible(ss.is_first_col())
            ax.spines['right'].set_visible(ss.is_last_col())


    d = .5 # how big to make the diagonal lines in axes coordinates
    if nr == 1:
        marker=[(-d, -1), (d, 1)]
        kwargs = get_kwargs(marker)
        axs[0].plot([1, 1], [1, 0], transform=axs[0].transAxes, **kwargs)
        axs[1].plot([0, 0], [1, 0], transform=axs[1].transAxes, **kwargs)
    elif nc == 1:
        marker=[(-1, -d), (1, d)]
        kwargs = get_kwargs(marker)
        axs[0].plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
        axs[1].plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
    else:
        marker = [(-d, -1), (d, 1)]
        kwargs = get_kwargs(marker)
        axs[0,0].plot(1, 1, transform=axs[0,0].transAxes, **kwargs)
        axs[0,1].plot(0, 1, transform=axs[0,1].transAxes, **kwargs)
        axs[1,0].plot(1, 0, transform=axs[1,0].transAxes, **kwargs)
        axs[1,1].plot(0, 0, transform=axs[1,1].transAxes, **kwargs)

        marker=[(-1, -d), (1, d)]
        kwargs = get_kwargs(marker)
        axs[0,0].plot(0, 0, transform=axs[0,0].transAxes, **kwargs)
        axs[0,1].plot(1, 0, transform=axs[0,1].transAxes, **kwargs)
        axs[1,0].plot(0, 1, transform=axs[1,0].transAxes, **kwargs)
        axs[1,1].plot(1, 1, transform=axs[1,1].transAxes, **kwargs)

    return axs.flat


