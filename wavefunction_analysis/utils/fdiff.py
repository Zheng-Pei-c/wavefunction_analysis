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
            coeff = [1./12., -2./3., 2./3., -1./12.]
        elif self.norder == 3:
            coeff = [-1./60., 3./20., -3./4., 3./4., -3./20., 1./60.]
        elif self.norder == 4:
            coeff = [1./280., -4./105., 1./5., -4./5., 4./5., -1./5., 4./105., -1./280.]
        coeff = np.array(coeff) / (self.step_size*unit)

        return np.einsum('i...,i->...', fx, coeff)


if __name__ == '__main__':
    from wavefunction_analysis.utils.unit_conversion import BOHR
    from wavefunction_analysis.utils.sec_mole import read_symbols_coords, write_mol_info_geometry, write_rem_info
    from wavefunction_analysis.utils.read_files import read_matrix, read_number

    infile = 'h2o.in'
    functional = 'hf'
    basis = '3-21g'

    norder, step_size = 2, 1e-4
    symbols, coords = read_symbols_coords(infile)
    natoms = len(symbols)

    #jb = 'write'
    jb = 'cal'
    suffixes = ['P ', 'angular '] # momentum, angular_momentum
    loop = 1 if jb == 'write' else len(suffixes)
    contract = True # contract hessian with density
    nwidth = 5

    for il in range(loop):
        suffix = suffixes[il]

        if jb == 'cal':
            out0 = infile[:-3] + '.out'
            nbas = read_number(out0, 'NBas: ', n=1)
            den = read_matrix(out0, nbas, nbas, 'scf density matrix arma', nwidth=-1, nskip=2)

            momentum_d1 = read_matrix(out0, nbas*nbas, natoms*3*natoms*3, suffix+'momentum derivative 3N arma', nwidth=-1, nskip=2)
            momentum_d1 = np.reshape(momentum_d1.T, (natoms, 3, natoms, 3, nbas, nbas))
            if suffix == 'angular':
                momentum_fd = np.einsum('ixjy...->xjy...', momentum_fd)
            print_matrix(suffix+'d1:', momentum_d1, nwidth, 1, trans=True)


        momentum_fd, momentum_fd2 = [], []
        for n in range(natoms):
            for x in range(3):
                fd = fdiff(norder, step_size)

                if jb == 'write':
                    coords_new = fd.get_x(coords, [n, x])

                    for d in range(len(coords_new)):
                        newfile = infile[:-3]+'_'+str(n+1)+'_'+str(x+1)+'_'+str(d+1)+ '.in'
                        write_mol_info_geometry(newfile, symbols=symbols, coords=coords_new[d])
                        write_rem_info(newfile, functional, basis)

                elif jb == 'cal':
                    momentum, momentum2 = [], []
                    for d in range(norder*2):
                        newfile = infile[:-3]+'_'+str(n+1)+'_'+str(x+1)+'_'+str(d+1)+ '.out'
                        m = read_matrix(newfile, nbas*nbas, natoms*3, suffix+'momentum 3N arma', nwidth=-1, nskip=2)
                        m2 = read_matrix(newfile, nbas*nbas, natoms*3*natoms*3, suffix+'momentum derivative 3N arma', nwidth=-1, nskip=2)
                        momentum.append(m.T)
                        momentum2.append(m2.T)
                    momentum = fd.compute_fdiff(np.array(momentum), 1./BOHR)
                    momentum2 = fd.compute_fdiff(np.array(momentum2), 1./BOHR)
                    momentum_fd.append(momentum)
                    momentum_fd2.append(momentum2)


        if jb == 'cal':
            momentum_fd = np.reshape(momentum_fd, (natoms, 3, natoms, 3, nbas, nbas))
            momentum_fd = np.einsum('ixjy...->jyix...', momentum_fd)
            if suffix == 'angular':
                momentum_fd = np.einsum('ixjy...->xjy...', momentum_fd)
            print_matrix(suffix+'fd:', momentum_fd, nwidth, 1, trans=True)
            print(suffix+'difference:', np.linalg.norm(momentum_fd+momentum_d1))

            if contract:
                momentum_fd2 = np.reshape(momentum_fd2, (natoms*3, natoms*3, natoms*3, nbas, nbas))
                momentum_fd2 = np.einsum('ijkpq,pq->jki', momentum_fd2, den)
                print_matrix(suffix+'fd2:', momentum_fd2, nwidth, 1)
            else:
                momentum_fd2 = np.reshape(momentum_fd2, (natoms*3, natoms*3, natoms*3, nbas, nbas))
                momentum_fd2 = np.einsum('ijkpq->jkipq', momentum_fd2)
                for i in range(natoms*3):
                    for j in range(natoms*3):
                        for k in range(natoms*3):
                            print('i:', i+1, 'j:', j+1, 'k:', k+1, end=' ')
                            print_matrix(suffix+'momentum derivative 2 3N:', momentum_fd2[i,j,k].T, nwidth, 1, trans=True)
