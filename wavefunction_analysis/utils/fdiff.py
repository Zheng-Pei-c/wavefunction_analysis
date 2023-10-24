import os, sys
import numpy as np

from wavefunction_analysis.utils import print_matrix

class fdiff():
    def __init__(self, norder=2, step_size=1e-3, unit=1.):
        """
        central finite difference method
        norder 1,2,3...
        accuracy = 2 * norder
        """
        self.norder = norder
        self.step_size = step_size * unit

        if norder < 1: norder == 1
        if norder > 4: raise ValueError('fdiff order currently is in [1,4].')


    def get_x(self, x0, idx):
        x0 = np.array(x0)
        x = np.array([x0] * self.norder * 2)

        d = []
        for i in range(self.norder, 0, -1):
            d.append(-i*self.step_size)
        for i in range(1, self.norder+1):
            d.append(i*self.step_size)
        d = np.array(d)

        if len(idx) == 1:
            x[:, idx] = x[:, idx] + d
        elif len(idx) == 2:
            i, j = idx
            x[:, i, j] = x[:, i, j] + d

        #print_matrix('x:', x)
        return x


    def compute_fdiff(self, fx, unit=1.): # stencil
        coeff = 0
        if self.norder == 1:
            coeff = [-.5, .5]
        elif self.norder == 2:
            coeff = [-1./12., -2./3., 2./3., 1./12.]
        elif self.norder == 3:
            coeff = [-1./60., 3./20., -3./4., 3./4., -3./20., 1./60.]
        elif self.norder == 4:
            coeff = [1./280., -4./105., 1./5., -4./5., 4./5., -1./5., 4./105., -1./280.]
        coeff = np.array(coeff) / (self.step_size*unit)

        return np.einsum('i...,i->...', fx, coeff)


if __name__ == '__main__':
    from wavefunction_analysis.utils.unit_conversion import BOHR
    from wavefunction_analysis.utils.sec_mole import read_symbols_coords, write_mol_info_geometry, write_rem_info
    from wavefunction_analysis.utils.read_files import read_matrix

    infile = 'h2o.in'
    nbas = 13
    norder, step_size = 3, 1e-5
    symbols, coords = read_symbols_coords(infile)

    #jb = 'write'
    jb = 'cal'

    momentum_fd = []
    for n in range(len(symbols)):
        for x in range(3):
            fd = fdiff(norder, step_size)

            if jb == 'write':
                coords_new = fd.get_x(coords, [n, x])

                for d in range(len(coords_new)):
                    newfile = infile[:-3]+'_'+str(n+1)+'_'+str(x+1)+'_'+str(d+1)+ '.in'
                    write_mol_info_geometry(newfile, symbols=symbols, coords=coords_new[d])
                    write_rem_info(newfile, 'hf', '3-21g')

            elif jb == 'cal':
                momentum = []
                for d in range(norder*2):
                    newfile = infile[:-3]+'_'+str(n+1)+'_'+str(x+1)+'_'+str(d+1)+ '.out'
                    m = read_matrix(newfile, nbas, nbas, 'P momentum 3N', 4, 1)
                    momentum.append(m)
                momentum = fd.compute_fdiff(np.array(momentum), 1./BOHR)
                momentum_fd.append(momentum)


    if jb == 'cal':
        momentum_fd = np.reshape(momentum_fd, (len(symbols), 3, len(symbols), 3, nbas, nbas))
        for m in range(len(symbols)):
            for x in range(3):
                for n in range(len(symbols)):
                    for y in range(3):
                        print('m:', m+1, 'x:', x+1, 'n:', n+1, 'y:', y+1, end=' ')
                        print_matrix('fd:', momentum_fd[n,y,m,x], 6, 1)
