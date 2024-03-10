import sys
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

        d = []
        for i in range(self.norder, 0, -1):
            d.append(-i*self.step_size)
        for i in range(1, self.norder+1):
            d.append(i*self.step_size)
        self.d = np.array(d)


    def get_d(self, ndim, idx):
        d = np.zeros((self.norder * 2, ndim))
        d[:, idx] += self.d
        return d


    def get_x(self, x0, idx):
        x0 = np.array(x0)
        x = np.array([x0] * self.norder * 2)

        if len(idx) == 1:
            x[:, idx] = x[:, idx] + self.d
        elif len(idx) == 2:
            i, j = idx
            x[:, i, j] = x[:, i, j] + self.d

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
    from wavefunction_analysis.utils import read_matrix, read_number
    from wavefunction_analysis.utils.unit_conversion import BOHR
    from wavefunction_analysis.utils.sec_mole import read_symbols_coords, write_mol_info_geometry, write_rem_info

    infile = sys.argv[1]
    jb = 'write'
    if len(sys.argv) > 2: jb = sys.argv[2] # 'write' # cal

    functional = read_number(infile, 'method ', n=1, dtype=str)
    basis      = read_number(infile, 'basis ', n=1, dtype=str)

    norder, step_size = 2, 1e-4
    symbols, coords = read_symbols_coords(infile)
    natoms = len(symbols)

    suffixes = ['P ', 'angular '] # momentum, angular_momentum
    fix2s    = ['', 'N']
    loop = 1 if jb == 'write' else len(suffixes)
    loop2 = 1 if jb == 'write' else len(fix2s)
    contract = False #True # contract hessian with density
    nwidth = 5

    p_momentum_3n_d1 = None

    for il in range(loop):
        suffix = suffixes[il]

        if jb == 'cal':
            out0 = infile[:-3] + '.out'
            nbas = read_number(out0, 'NBas: ', n=1)
            den = read_matrix(out0, nbas, nbas, 'scf density matrix arma', nwidth=-1, nskip=2)
            print('den-den.T:', np.linalg.norm(den-den.T))

            momentum_3 = read_matrix(out0, nbas*nbas, 3, suffix+'momentum 3 arma', nwidth=-1, nskip=2).T.reshape(3, nbas, nbas)
            momentum_3n = read_matrix(out0, nbas*nbas, natoms*3, suffix+'momentum 3N arma', nwidth=-1, nskip=2).T.reshape(natoms, 3, nbas, nbas)
            print(suffix+'momentum difference:', np.linalg.norm(momentum_3 - np.einsum('nx...->x...', momentum_3n)))
            if suffix == 'angular ':
                momentum_ref = read_matrix(out0, nbas*nbas, 3, 'angular momentum ref', nwidth=-1, nskip=2).T.reshape(3, nbas, nbas)
                m_diff = momentum_3 - momentum_ref
                print('angular difference ref:', np.linalg.norm(m_diff))
            if suffix == 'P ':
                momentum_ref = read_matrix(out0, nbas*nbas, natoms*3, 'SRx momentum 3N ref', nwidth=-1, nskip=2).T.reshape(natoms, 3, nbas, nbas)
                m_diff = momentum_3n - momentum_ref
                print('P difference ref:', np.linalg.norm(m_diff))
            #    print_matrix(suffix+'momentum 3:', momentum_3, nwidth, nind=1)
            #    #print_matrix(suffix+'momentum_3n:', momentum_3n, nwidth, 1)

            momentum_3_d1 = read_matrix(out0, nbas*nbas, natoms*3*3, suffix+'momentum derivative 3 arma', nwidth=-1, nskip=2).T.reshape(3, natoms, 3, nbas, nbas)
            momentum_3n_d1 = read_matrix(out0, nbas*nbas, natoms*3*natoms*3, suffix+'momentum derivative 3N arma', nwidth=-1, nskip=2).T.reshape(natoms, 3, natoms, 3, nbas, nbas)
            print(suffix+'momentum derivative difference:', np.linalg.norm(momentum_3_d1 - np.einsum('nx...->x...', momentum_3n_d1)))
            #if suffix == 'P ':
            #    p_momentum_3n_d1 = np.copy(momentum_3n_d1)
            #    #print_matrix(suffix+'momentum 3_d1:', momentum_3_d1, nwidth, nind=1)
            #    print_matrix(suffix+'momentum_3n_d1:', momentum_3n_d1, nwidth, 1)

            if contract:
                momentum_3_d2 = read_matrix(out0, natoms*3*natoms*3, 3, suffix+'momentum derivative 2 3 arma', nwidth=-1, nskip=2).T.reshape(3, natoms*3, natoms*3)
                momentum_3n_d2 = read_matrix(out0, natoms*3*natoms*3, natoms*3, suffix+'momentum derivative 2 3N arma', nwidth=-1, nskip=2).T.reshape(natoms, 3, natoms*3, natoms*3)
            else:
                momentum_3_d2 = read_matrix(out0, nbas*nbas*natoms*3*natoms*3, 3, suffix+'momentum derivative 2 3 arma', nwidth=-1, nskip=2).T.reshape(3, natoms*3, natoms*3, nbas, nbas)
                momentum_3n_d2 = read_matrix(out0, nbas*nbas*natoms*3*natoms*3, natoms*3, suffix+'momentum derivative 2 3N arma', nwidth=-1, nskip=2).T.reshape(natoms, 3, natoms*3, natoms*3, nbas, nbas)
            print(suffix+'momentum derivative 2 difference:', np.linalg.norm(momentum_3_d2 - np.einsum('nx...->x...', momentum_3n_d2)))


        momentum_3_fd, momentum_3n_fd, momentum_3_fd2, momentum_3n_fd2 = [], [], [], []
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
                    momentum, momentum2, momentum3, momentum4 = [], [], [], []
                    for d in range(norder*2):
                        newfile = infile[:-3]+'_'+str(n+1)+'_'+str(x+1)+'_'+str(d+1)+ '.out'
                        m = read_matrix(newfile, nbas*nbas, 3, suffix+'momentum 3 arma', nwidth=-1, nskip=2)
                        m2 = read_matrix(newfile, nbas*nbas, 3*natoms*3, suffix+'momentum derivative 3 arma', nwidth=-1, nskip=2)
                        m3 = read_matrix(newfile, nbas*nbas, natoms*3, suffix+'momentum 3N arma', nwidth=-1, nskip=2)
                        m4 = read_matrix(newfile, nbas*nbas, natoms*3*natoms*3, suffix+'momentum derivative 3N arma', nwidth=-1, nskip=2)
                        momentum.append(m.T)
                        momentum2.append(m2.T)
                        momentum3.append(m3.T)
                        momentum4.append(m4.T)
                    momentum = fd.compute_fdiff(np.array(momentum), 1./BOHR)
                    momentum2 = fd.compute_fdiff(np.array(momentum2), 1./BOHR)
                    momentum3 = fd.compute_fdiff(np.array(momentum3), 1./BOHR)
                    momentum4 = fd.compute_fdiff(np.array(momentum4), 1./BOHR)
                    momentum_3_fd.append(momentum)
                    momentum_3_fd2.append(momentum2)
                    momentum_3n_fd.append(momentum3)
                    momentum_3n_fd2.append(momentum4)


        if jb == 'cal':
            momentum_3_fd = np.reshape(momentum_3_fd, (natoms, 3, 3, nbas, nbas))
            momentum_3_fd = np.einsum('ixy...->yix...', momentum_3_fd)
            m_diff = momentum_3_fd-momentum_3_d1
            print(suffix+'3 derivative difference:', np.linalg.norm(m_diff))
            #if suffix == 'angular ':
            #    print_matrix(suffix+'3 derivative difference:', m_diff, nwidth, 1)
            #    print_matrix(suffix+'momentum_3_d1:', momentum_3_d1, nwidth, 1)
            #    print_matrix(suffix+'momentum_3_fd:', momentum_3_fd, nwidth, 1)
            momentum_3n_fd = np.reshape(momentum_3n_fd, (natoms, 3, natoms, 3, nbas, nbas))
            momentum_3n_fd = np.einsum('ixjy...->jyix...', momentum_3n_fd)
            m_diff = momentum_3n_fd-momentum_3n_d1
            print(suffix+'3N derivative difference:', np.linalg.norm(m_diff))
            #if suffix == 'angular ':
            #    print_matrix(suffix+'3N derivative difference:', m_diff, nwidth, 1)
            #    print_matrix(suffix+'momentum_3n_d1:', momentum_3n_d1, nwidth, 1)
            #    print_matrix(suffix+'momentum_3n_fd:', momentum_3n_fd, nwidth, 1)

            momentum_3_fd2 = np.reshape(momentum_3_fd2, (natoms*3, 3, natoms*3, nbas, nbas))
            momentum_3n_fd2 = np.reshape(momentum_3n_fd2, (natoms*3, natoms*3, natoms*3, nbas, nbas))
            if contract:
                momentum_3_fd2 = np.einsum('ijkpq,pq->jki', momentum_3_fd2, den)
                momentum_3n_fd2 = np.einsum('ijkpq,pq->jki', momentum_3n_fd2, den)
                momentum_3n_d2 = np.reshape(momentum_3n_d2, (natoms*3, natoms*3, natoms*3))
                print_matrix(suffix+' fd2:', momentum_3n_fd2, nwidth, 1)
                print_matrix(suffix+' d2:', momentum_3n_d2, nwidth, 1)
            else:
                momentum_3_fd2 = np.einsum('ijkpq->jkipq', momentum_3_fd2)
                momentum_3n_fd2 = np.einsum('ijkpq->jkipq', momentum_3n_fd2)
                momentum_3n_d2 = np.reshape(momentum_3n_d2, (natoms*3, natoms*3, natoms*3, nbas, nbas))

            m_diff = momentum_3_fd2 - momentum_3_d2
            print(suffix+'3 derivative 2 difference:', np.linalg.norm(m_diff))
            if suffix == 'P ':
                print_matrix(suffix+'3 derivative 2 difference:', m_diff, nwidth, 1)
                print_matrix(suffix+'momentum_3_d2', momentum_3_d2, nwidth, 1)
                print_matrix(suffix+'momentum_3_fd2', momentum_3_fd2, nwidth, 1)

                #momentum_3_d2 = np.reshape(momentum_3_d2, (3, natoms, 3, natoms, 3, nbas, nbas))
                #p_momentum_3n_d1 = np.einsum('ixjypq->ixjyqp', p_momentum_3n_d1) - p_momentum_3n_d1
                #for x in range(3):
                #    y, z = (x+1) % 3, (x+2) % 3
                #    momentum_3_d2[x,:,x,:,y] += p_momentum_3n_d1[:,z,:,x]
                #    momentum_3_d2[x,:,x,:,z] -= p_momentum_3n_d1[:,y,:,x]

                #    momentum_3_d2[x,:,y,:,x] += p_momentum_3n_d1[:,z,:,x]
                #    momentum_3_d2[x,:,z,:,x] -= p_momentum_3n_d1[:,y,:,x]

                #    momentum_3_d2[x,:,y,:,y] += 2.*p_momentum_3n_d1[:,z,:,y]
                #    momentum_3_d2[x,:,z,:,z] -= 2.*p_momentum_3n_d1[:,y,:,z]

                #    momentum_3_d2[x,:,y,:,z] += p_momentum_3n_d1[:,z,:,z]
                #    momentum_3_d2[x,:,y,:,z] -= p_momentum_3n_d1[:,y,:,y]
                #
                #    momentum_3_d2[x,:,z,:,y] += p_momentum_3n_d1[:,z,:,z]
                #    momentum_3_d2[x,:,z,:,y] -= p_momentum_3n_d1[:,y,:,y]
                #momentum_3_d2 = np.reshape(momentum_3_d2, (3, natoms*3, natoms*3, nbas, nbas))
                #print_matrix(suffix+'momentum_3_d2 correct', momentum_3_d2, nwidth, 1)
                #m_diff = momentum_3_fd2 - momentum_3_d2
                #print_matrix(suffix+'3 derivative 2 difference correct:', m_diff, nwidth, 1)
                #print(suffix+'3 derivative 2 difference correct:', np.linalg.norm(m_diff))

            m_diff = momentum_3n_fd2 - momentum_3n_d2
            print(suffix+'3N derivative 2 difference:', np.linalg.norm(m_diff))
            #if suffix == 'P ':
            #    print_matrix(suffix+'3N derivative 2 difference:', m_diff, nwidth, 1)
            #    print_matrix(suffix+'momentum_3n_d2', momentum_3n_d2, nwidth, 1)
            #    print_matrix(suffix+'momentum_3n_fd2', momentum_3n_fd2, nwidth, 1)
