import numpy as np

from wavefunction_analysis.utils import print_matrix, convert_units

def get_photon_trans_amplitude(ntot, scaling=1., energy=None):
    trans = np.zeros((ntot, ntot))
    for j in range(ntot-1):
        trans[j,j+1] = trans[j+1,j] = np.sqrt(j+1)*scaling

    if energy:
        for j in range(ntot):
            trans[j,j] = j*energy

    return trans


class PhotonDynamicsStep():
    def __init__(self, key):
        key.setdefault('photon_frequency', 0.05)
        key.setdefault('freq_unit', 'hartree')
        key.setdefault('coupling_strength', np.array([0.,0.,0.1]))
        key.setdefault('photon_number', 1)
        key.setdefault('photon_basis_size', 10)

        for name, value in key.items(): # put all the variables in the class
            setattr(self, name, value)

        if self.freq_unit != 'hartree' or self.freq_unit != 'eh':
            self.photon_frequency = convert_units(self.photon_frequency, self.freq_unit, 'hartree')
            self.freq_unit = 'hartree'

        if isinstance(self.photon_frequency, float):
            self.photon_frequency = np.array([self.photon_frequency])
            self.coupling_strength = np.array([self.coupling_strength])
            self.photon_number = np.array([self.photon_number])
            self.photon_basis_size = np.array([self.photon_basis_size])

        self.nmode = len(self.photon_frequency)

        self.photon_density = [None]*self.nmode
        for i in range(self.nmode):
            self.photon_density[i] = self.get_initial_photon_density(self.photon_number[i], self.photon_basis_size[i])


    def get_initial_photon_density(self, n, ntot):
        init_density = np.zeros((ntot,ntot))
        init_density[n,n] = n
        return init_density


    def update_photon_density(self, molecular_dipole, dt):
        coupling = np.einsum('i,ix,x->i', np.sqrt(self.photon_frequency/2.), self.coupling_strength, molecular_dipole)

        photon_trans, energy = np.zeros(self.nmode), np.zeros(self.nmode)
        for i in range(self.nmode):
            ntot = self.photon_basis_size[i]
            trans = get_photon_trans_amplitude(ntot, coupling[i], self.photon_frequency[i])

            e, v = np.linalg.eigh(trans)
            print_matrix('eigenvalues:', e)
            transp = np.einsum('pi,i,qi->pq', v, np.exp(1j*e*dt), v)
            #transm = np.einsum('pi,i,qi->pq', v, np.exp(-1j*e*dt), v)
            transm = transp.conjugate()
            self.photon_density[i] = np.einsum('ij,jk,kl->il', transp, self.photon_density[i], transm).real
            print_matrix('diagonal of photon_density:', np.diag(self.photon_density[i]))

            photon_trans[i] = np.einsum('ij,ji->', trans, self.photon_density[i])
            energy[i] = self.photon_frequency[i] * np.dot(range(ntot), np.diag(self.photon_density[i]))

        kwargs = {}
        kwargs['trans_coeff'] = photon_trans
        kwargs['photon_energy'] = np.sum(energy)

        #print('photon_trans:', photon_trans)
        #print('photon_energy:', np.sum(energy))
        return kwargs
