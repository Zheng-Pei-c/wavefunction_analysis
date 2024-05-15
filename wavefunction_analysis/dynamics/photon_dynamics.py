import numpy as np
from scipy.linalg import expm

from wavefunction_analysis.utils import print_matrix, convert_units
from wavefunction_analysis.dynamics import harmonic_oscillator

def get_trans_amplitude(ntot, scaling=1., energy=None):
    trans = np.zeros((ntot, ntot))
    for j in range(ntot-1):
        trans[j,j+1] = trans[j+1,j] = np.sqrt(j+1)*scaling

    if energy:
        for j in range(ntot):
            trans[j,j] = j*energy

    return trans


class PhotonDynamicsStep():
    def __init__(self, **kwargs):
        kwargs.setdefault('frequency', 0.05)
        kwargs.setdefault('freq_unit', 'hartree')
        kwargs.setdefault('c_lambda', np.array([0.,0.,0.1]))
        kwargs.setdefault('number', 1)
        kwargs.setdefault('basis_size', 10)

        for name, value in kwargs.items(): # put all the variables in the class
            setattr(self, name, value)

        if self.freq_unit != 'hartree' or self.freq_unit != 'eh':
            self.frequency = convert_units(self.frequency, self.freq_unit, 'hartree')
            self.freq_unit = 'hartree'

        if isinstance(self.frequency, float):
            self.frequency = np.array([self.frequency])
            self.c_lambda = np.array([self.c_lambda])
            self.number = np.array([self.number])
            self.basis_size = np.array([self.basis_size])

        self.nmode = len(self.frequency)

        self.density = [None]*self.nmode
        for i in range(self.nmode):
            self.density[i] = self.get_initial_density(self.number[i], self.basis_size[i])


    def get_initial_density(self, n, ntot):
        init_density = np.zeros((ntot,ntot))
        init_density[n,n] = n
        #init_density = np.ones((ntot,ntot))/ntot
        return init_density


    def update_density(self, molecular_dipole, dt):
        coupling = np.einsum('i,ix,x->i', np.sqrt(self.frequency/2.), self.c_lambda, molecular_dipole)
        print('coupling:', coupling)

        trans_coeff, energy = np.zeros(self.nmode), np.zeros(self.nmode)
        for i in range(self.nmode):
            ntot = self.basis_size[i]
            trans = get_trans_amplitude(ntot, coupling[i], self.frequency[i])

            #e, v = np.linalg.eigh(trans)
            #print_matrix('eigenvalues:', e)
            #transp = np.einsum('pi,i,qi->pq', v, np.exp(1j*e*dt), v)
            transp = expm(1j*trans*dt)
            #transm = np.einsum('pi,i,qi->pq', v, np.exp(-1j*e*dt), v)
            transm = transp.conjugate()
            self.density[i] = np.einsum('ij,jk,kl->il', transp, self.density[i], transm)#.real
            print_matrix('diagonal of density:', self.density[i][:5,:5])

            trans = get_trans_amplitude(ntot)
            trans_coeff[i] = np.einsum('ij,ji->', trans, self.density[i])
            energy[i] = self.frequency[i] * np.dot(range(ntot), np.diag(self.density[i]))

        kwargs = {}
        kwargs['trans_coeff'] = trans_coeff
        kwargs['photon_energy'] = np.sum(energy)

        print('trans:', trans_coeff)
        print('photon_energy:', np.sum(energy))
        return kwargs


class PhotonDynamicsStep2(harmonic_oscillator):
    def convert_parameter_units(self, unit_dict):
        self.n_site = 1 #xyz
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

        trans_coeff = np.sum(self.coordinate, axis=1) #2. * np.dot(self.frequency, self.coordinate)
        energy = self.energy

        kwargs = {}
        kwargs['trans_coeff'] = trans_coeff
        kwargs['photon_energy'] = energy
        kwargs['frequency'] = 8.*self.frequency**2
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
