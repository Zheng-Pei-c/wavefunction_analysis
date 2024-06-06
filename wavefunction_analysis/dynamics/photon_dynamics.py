import numpy as np
from scipy.linalg import expm

from wavefunction_analysis.utils import print_matrix, convert_units
from wavefunction_analysis.utils import put_keys_kwargs_to_object
from wavefunction_analysis.dynamics import harmonic_oscillator

def get_trans_amplitude(ntot, coupling=1., energy=None, vector=False):
    trans = np.sqrt(np.arange(1, ntot))*coupling # does not include the end value

    if vector:
        return trans

    trans = np.diagflat(trans, 1)
    trans += trans.T

    if energy:
        np.fill_diagonal(trans, np.arange(0, ntot)*energy)

    return trans


class PhotonDynamicsStep():
    def __init__(self, key, **kwargs):
        key.setdefault('frequency', 0.05)
        key.setdefault('freq_unit', 'hartree')
        key.setdefault('c_lambda', np.array([0.,0.,0.1]))
        key.setdefault('init_number', [1,1,1])
        key.setdefault('basis_size', 10)

        put_keys_kwargs_to_object(self, key, **kwargs)

        if self.freq_unit != 'hartree' or self.freq_unit != 'eh':
            self.frequency = convert_units(self.frequency, self.freq_unit, 'hartree')
            self.freq_unit = 'hartree'

        if isinstance(self.frequency, float):
            self.frequency = np.array([self.frequency])
            self.c_lambda = np.array([self.c_lambda])
            self.init_number = np.array([self.init_number])
            self.basis_size = np.array([self.basis_size])

        self.scaled_freq = np.sqrt(self.frequency/2.)

        self.nmode = len(self.frequency)

        self._trans, self.density = [None]*self.nmode, [None]*self.nmode
        for i in range(self.nmode):
            self._trans[i] = get_trans_amplitude(self.basis_size[i], vector=True)
            self.density[i] = self.get_initial_density(self.basis_size[i], self.init_number[i])


    def get_initial_density(self, ntot, ns):
        if isinstance(ns, int): ns = [ns]

        init_density = np.zeros((len(ns),ntot,ntot))
        #init_density[n,n] = n
        for x in range(len(ns)):
            n = int(ns[x])
            for i in range(n+1):
                init_density[x,i,i] = 1./(n+1)
        #init_density = np.ones((ntot,ntot))/ntot
        return init_density


    def update_density(self, molecular_dipole, dt, half=1):
        if half == 2:
            return

        coupling = np.einsum('i,ix,x->ix', self.scaled_freq, self.c_lambda, molecular_dipole)

        trans_coeff, energy = np.zeros((self.nmode, 3)), np.zeros((self.nmode, 3))
        for i in range(self.nmode):
            ntot = self.basis_size[i]
            _trans = self._trans[i]

            for x in range(np.argwhere(coupling[i]>1e-8)[0]): # spatial directions
                trans = get_trans_amplitude(ntot, coupling[i,x], self.frequency[i])

                #e, v = np.linalg.eigh(trans)
                #print_matrix('eigenvalues:', e)
                #transp = np.einsum('pi,i,qi->pq', v, np.exp(1j*e*dt), v)
                transp = expm(1j*trans*dt)
                #transm = np.einsum('pi,i,qi->pq', v, np.exp(-1j*e*dt), v)
                transm = transp.conjugate()
                self.density[i,x] = np.einsum('ij,jk,kl->il', transp, self.density[i,x], transm)#.real
                #print_matrix('diagonal of density:', self.density[i])

                trans_coeff[i,x] = np.dot(_trans, self.density[i,x].diag(1)+self.density[i,x].diag(-1))
                energy[i,x] = self.frequency[i] * np.dot(range(ntot), np.diag(self.density[i,x]))

        self.energy = np.sum(energy)

        kwargs = {}
        kwargs['trans_coeff'] = np.einsum('i,ix,ix->x', self.scaled_freq, trans_coeff, self.c_lambda)

        return kwargs


class PhotonDynamicsStep2(harmonic_oscillator):
    def convert_parameter_units(self, unit_dict):
        self.n_site = 3 #xyz
        self.frequency = convert_units(self.frequency, self.freq_unit, 'hartree')

        if isinstance(self.frequency, float):
            self.frequency = np.array([self.frequency])
            self.c_lambda = np.array([self.c_lambda])

        self.mass = np.ones(len(self.frequency))


    def update_density(self, molecular_dipole, dt, half=1):
        force = -np.einsum('i,ix,x->ix', self.frequency, self.c_lambda, molecular_dipole)
        if self.n_site == 1:
            force = np.sum(force, axis=1).reshape(-1, 1)

        self.update_coordinate_velocity(force, half)

        kwargs = {}
        kwargs['trans_coeff'] = np.einsum('i,ix,ix->x', self.frequency, self.coordinate, self.c_lambda)

        return kwargs


if __name__ == '__main__':
    dt = 25 #au
    nsteps = 10
    c_lambda = np.zeros(3)
    key = {}
    key['c_lambda'] = c_lambda

    photon = PhotonDynamicsStep(**key)

    energy = []
    for i in range(nsteps):
        kwargs = photon.update_density(np.zeros(3), dt)
        energy.append(kwargs['energy'])
    energy = np.array(energy)

    print_matrix('energy:', energy, 5)
