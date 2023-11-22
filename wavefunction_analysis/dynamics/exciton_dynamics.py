import os, sys
import warnings
import numpy as np
from pyscf.data.nist import HARTREE2J, HARTREE2EV, BOLTZMANN, PROTON_MASS_AU, BOHR, PLANCK, E_CHARGE
from pyscf import lib

from wavefunction_analysis.utils import print_matrix
from wavefunction_analysis.utils.sec_rem import put_kwargs_keys_to_object, put_kwargs_to_keys


AU2FS = 2.e5 * 8.854187817e-12 * PLANCK * BOHR / E_CHARGE**2

def get_boltzmann_beta(temperature):
    return HARTREE2J / (BOLTZMANN * temperature)


class OscillatorDynamicsStep():
    def __init__(self, key={}, **kwargs):
        """
        required input:
            nuclear_mass in amu
            nuclear_omega in meV
        input with default values:
            temperature as 298 K
            nuclear_update_method as velocity_verlet
            nuclear_distribution as thermo
        """
        self.debug = 2
        self.temperature = 298 # K

        # phonon dynamics time step
        self.nuclear_dt = 1 # au

        # nuclear coordinates and velocities update method
        self.nuclear_update_method = 'euler' #'velocity_verlet'
        # initial distribution method for oscillator coordinates and velocities
        self.nuclear_distribution = 'thermo'

        # vibrational mode number of each site
        self.n_mode = 0
        # oscillator mass in amu for vibrational mode
        self.nuclear_mass = 0
        # vibrational frequency in meV
        self.nuclear_omega = 0

        put_kwargs_keys_to_object(self, key, **kwargs)

        self.check_sanity()
        # convert the input parameters into atomic unit for the calculations
        self.convert_parameter_units()


    def check_sanity(self):
        """
        check the input parameters
        """
        if self.n_mode == 0:
            raise ValueError('n_mode in %s should be larger than 0' % self.__class__.__name__)
        assert self.nuclear_mass.size  == self.nuclear_omega.size == self.n_mode


    def convert_parameter_units(self):
        # Boltzmann coefficient is 1/(k_B * T)
        self.beta_b = get_boltzmann_beta(self.temperature)
        self.nuclear_mass *= PROTON_MASS_AU
        self.nuclear_omega /= (HARTREE2EV*1000)
        self.nuclear_omega2 = self.nuclear_omega**2 # for computational efficiency

        if self.debug > 0:
            #print('KT:', 1./self.beta_b, 1./self.beta_b*HARTREE2EV*1000)
            print_matrix('oscillator mass (au):', self.nuclear_mass)
            print_matrix('oscillator omega (au):', self.nuclear_omega)


    def get_initial_coordinates_velocities(self, n_site, distribution=None):
        """
        generate the initial oscillator coordinates and velocities
        n_site: number of total molecular sites
        """
        if distribution is None: distribution = self.nuclear_distribution

        self.nuclear_coordinates = np.zeros((n_site, self.n_mode))
        self.nuclear_velocities  = np.zeros((n_site, self.n_mode))

        def get_gaussian_distribution(variance, size, mean=0):
            rng = np.random.default_rng()
            return rng.normal(loc=mean, scale=np.sqrt(variance), size=size)

        # Boltzmann thermol distribution follows gaussian function
        if distribution == 'thermo':
            K = np.einsum('i,i->i', self.nuclear_mass, self.nuclear_omega2)
            variance = 1. / (self.beta_b * K)
            if self.debug > 0:
                print_matrix('force constant:', K)
                print_matrix('coordinate variance:', variance)

            for i in range(self.n_mode):
                self.nuclear_coordinates[:,i] = get_gaussian_distribution(variance[i], n_site)

            variance = 1. / (self.beta_b * self.nuclear_mass)
            if self.debug > 0:
                print_matrix('velocity variance:', variance)

            for i in range(self.n_mode):
                self.nuclear_velocities[:,i] = get_gaussian_distribution(variance[i], n_site)

        if self.debug > 1:
            print_matrix('initial coordinates:', self.nuclear_coordinates, 10)
            print_matrix('initial velocities:', self.nuclear_velocities, 10)

        nuclear_energy = self.get_nuclear_energy()


    def update_nuclear_coords_velocity(self, nuclear_force, nuclear_coordinates=None):
        if nuclear_coordinates is None: nuclear_coordinates = self.nuclear_coordinates

        # add oscillator force first
        nuclear_force -= np.einsum('m,m,nm->nm', self.nuclear_mass, self.nuclear_omega2, nuclear_coordinates)

        if self.nuclear_update_method == 'euler':
            self.euler_step(nuclear_force)
        elif self.nuclear_update_method == 'leapfrog':
            self.leapfrog_step(nuclear_force)
        elif self.nuclear_update_method == 'velocity_verlet':
            self.velocity_verlet_step(nuclear_force, 1)
            # we will finish the last falf after electronic step


    def euler_step(self, nuclear_force):
        self.nuclear_velocities += self.nuclear_dt * np.einsum('ni,i->ni', nuclear_force, 1./self.nuclear_mass)
        self.nuclear_coordinates += self.nuclear_dt * self.nuclear_velocities
        self.get_nuclear_energy(self.nuclear_velocities)


    def leapfrog_step(self, nuclear_force):
        old_nuclear_velocities = np.copy(self.nuclear_velocities)
        self.nuclear_velocities += self.nuclear_dt * np.einsum('ni,i->ni', nuclear_force, 1./self.nuclear_mass)
        self.nuclear_coordinates += self.nuclear_dt * self.nuclear_velocities

        average_nuclear_velocities = 0.5 * (old_nuclear_velocities + self.nuclear_velocities)
        self.get_nuclear_energy(average_nuclear_velocities)


    def velocity_verlet_step(self, nuclear_force, half):
        self.nuclear_velocities += 0.5 * self.nuclear_dt * np.einsum('ni,i->ni', nuclear_force, 1./self.nuclear_mass)
        if half == 1:
            self.nuclear_coordinates += self.nuclear_dt * self.nuclear_velocities
        if half == 2:
            self.get_nuclear_energy(self.nuclear_velocities)


    def get_nuclear_energy(self, velocities=None, coordinates=None, mass=None, omega2=None):
        if velocities is None: velocities = self.nuclear_velocities
        if coordinates is None: coordinates = self.nuclear_coordinates
        if mass is None: mass = self.nuclear_mass
        if omega2 is None: omega2 = self.nuclear_omega2

        v2 = np.einsum('ni,ni->i', velocities, velocities)
        self.nuclear_kinetic = 0.5 * np.einsum('i,i', mass, v2)
        self.nuclear_temperature = self.nuclear_kinetic * 2 / (velocities.size)
        v2 = np.einsum('ni,ni->i', coordinates, coordinates)
        self.nuclear_potential = 0.5 * np.einsum('i,i,i->', mass, omega2, v2)
        self.nuclear_energy = self.nuclear_kinetic + self.nuclear_potential
        #print('nuclear energy (au):', self.nuclear_energy, self.nuclear_kinetic, self.nuclear_potential)

        return self.nuclear_energy


    def get_phonon_hamiltonian(self, velocities=None, coordinates=None, mass=None, omega2=None):
        if velocities is None: velocities = self.nuclear_velocities
        if coordinates is None: coordinates = self.nuclear_coordinates
        if mass is None: mass = self.nuclear_mass
        if omega2 is None: omega2 = self.nuclear_omega2

        mass2 = np.copy(mass) *.5
        self.phonon_hamiltonian = np.einsum('i,ni,ni->n', mass2, velocities, velocities)
        v2 = np.einsum('i,i->i', mass2, omega2)
        self.phonon_hamiltonian += np.einsum('i,ni,ni->n', v2, coordinates, coordinates)

        return self.phonon_hamiltonian


class ExcitonDynamicsStep():
    def __init__(self, key={}, **kwargs):
        """
        required input:
            nstate is a number
            n_site as a number or 1d, 2d, or 3d int array
            distance as a number or 1d, 2d, or 3d float array in Angstrom
            energy as an array in meV
            coupling_g has dimension (n_mode, nstate) in meV / AA
            coupling_j in meV
            coupling_a in meV
        """
        self.debug = 1
        self.temperature = 298 # K

        # exciton dynamics time step
        self.exciton_dt = 1 # au
        # 3D molecular site number
        self.n_site = 0
        # vibrational mode number of each site
        self.n_mode = 0
        # exciton number of each site
        self.nstate = 0
        # intermolecular distance in (x,y,z) directions
        self.distance = 0 # Angstrom

        # exciton energy of each site
        self.energy = 0
        # three couplings
        self.coupling_g = 0
        self.coupling_j = 0
        self.coupling_a = 0

        put_kwargs_keys_to_object(self, key, **kwargs)

        self.check_sanity()
        # convert the input parameters into atomic unit for the calculations
        self.convert_parameter_units()
        self.process_parameters()


    def check_sanity(self):
        """
        check the input parameters
        """
        if self.n_mode == 0:
            raise ValueError('n_mode in %s should be larger than 0' % self.__class__.__name__)


    def convert_parameter_units(self):
        self.beta_b = get_boltzmann_beta(self.temperature)
        self.n_site_tot = np.prod(self.n_site)
        self.distance /= BOHR
        self.length = np.linspace(0, self.distance*self.n_site_tot, self.n_site_tot)
        self.length -= np.average(self.length)

        self.energy /= (HARTREE2EV*1000)
        self.coupling_g /= (HARTREE2EV*1000/BOHR)
        self.coupling_j /= (HARTREE2EV*1000)
        self.coupling_a /= (HARTREE2EV*1000/BOHR)

        self.ntype = self.coupling_j.shape[0] # number of different dimers

        if self.debug > 0:
            print_matrix('on-site exciton energy (au):', self.energy, 10)
            print_matrix('on-site exciton-phonon coupling (au):', self.coupling_g, 10)
            print_matrix('off-site exciton-exciton coupling (au):', self.coupling_j.reshape(-1, self.nstate**2), 10)
            print_matrix('off-site exciton-phonon-exciton coupling (au):', self.coupling_a.reshape(-1, self.nstate**2), 10)


    def process_parameters(self):
        #n_site_tot, n_mode, nstate = self.n_site_tot, self.n_mode, self.nstate
        # every site has same exciton energies and same on-site couplings
        #self.energy = np.tile(self.energy, (n_site_tot, 1))
        #self.coupling_g = np.tile(self.self.coupling_g, (n_site_tot, 1, 1))


        # off-site coupling_j and coupling_a have different values for different dimer types
        #n = n_site_tot // self.ntype
        #self.coupling_j = np.tile(self.coupling_j, (n, 1, 1))
        #self.coupling_a = np.tile(self.coupling_a, (n, 1, 1, 1))

        #if self.debug > 0:
        #    print_matrix('coupling_j:', self.coupling_j, 10)
        pass


    def get_exciton_hamiltonian0(self):
        self.exciton_hamiltonian0 = np.zeos((self.n_site_tot, self.nstate, self.n_site_tot, self.nstate))

        for i in range(self.n_site_tot-1):
            k = i % self.ntype
            hamiltonian[i,:,i+1] = self.coupling_j[k]
            hamiltonian[i+1,:,i] = hamiltonian[i,:,i+1].transpose()
        self.exciton_hamiltonian0.reshape(self.n_site_tot*self.nstate, -1)

        np.fill_diagonal(self.exciton_hamiltonian0, np.tile(self.energy, (n_site_tot, 1)).ravel())


    def get_exciton_hamiltonian1(self, nuclear_coordinates):
        diagonal = np.einsum('mi,nm->ni', self.coupling_g, nuclear_coordinates)

        hamiltonian = np.zeros((self.n_site_tot, self.nstate, self.n_site_tot, self.nstate))

        nuclear_coordinates1 = np.copy(nuclear_coordinates)
        nuclear_coordinates1[:-1] -= nuclear_coordinates[1:]

        #coupling = np.zeros((self.n_site_tot, self.nstate, self.nstate))
        #coupling = np.einsum('kmij,nkm->nkij', self.coupling_a, nuclear_coordinates1.reshape(-1, self.ntype, self.n_mode))
        #coupling = coupling.reshape(-1, self.nstate, self.nstate)
        for i in range(self.n_site_tot-1):
            k = i % self.ntype
            coupling = np.einsum('mij,m->ij', self.coupling_a[k], nuclear_coordinates1[i])
            hamiltonian[i,:,i+1] = coupling[i]
            hamiltonian[i+1,:,i] = hamiltonian[i,:,i+1].transpose()

        np.fill_diagonal(hamiltonian, diagonal.ravel())
        return hamiltonian


    def get_exciton_hamiltonian2(self, nuclear_coordinates):
        self.exciton_hamiltonian = self.get_exciton_hamiltonian1(nuclear_coordinates)
        self.exciton_hamiltonian += self.exciton_hamiltonian0
        return self.exciton_hamiltonian


    def get_exciton_diagonal(self, nuclear_coordinates):
        diagonal = np.einsum('mi,nm->ni', self.coupling_g, nuclear_coordinates)
        diagonal += np.tile(self.energy, (self.n_site_tot, 1))
        #diagonal += np.tile(phonon_hamiltonian, (self.nstate, 1)).T
        #return np.ravel(self.energy + diagonal)
        return diagonal.ravel()


    def get_exciton_couplings(self, nuclear_coordinates):
        hamiltonian = np.zeros((self.n_site_tot, self.nstate, self.n_site_tot, self.nstate))

        nuclear_coordinates1 = np.copy(nuclear_coordinates)
        nuclear_coordinates1[:-1] -= nuclear_coordinates[1:]
        #coupling = np.einsum('kmij,nkm->nkij', self.coupling_a, nuclear_coordinates1.reshape(-1, self.ntype, self.n_mode))
        #coupling = coupling.reshape(-1, self.nstate, self.nstate)

        for i in range(self.n_site_tot-1):
            k = i % self.ntype
            coupling = np.einsum('mij,m->ij', self.coupling_a[k], nuclear_coordinates1[i])
            hamiltonian[i,:,i+1] = self.coupling_j[k] + coupling
            #hamiltonian[i,:,i+1] = self.coupling_j[i] + coupling[i]
            hamiltonian[i+1,:,i] = hamiltonian[i,:,i+1].transpose()

        return np.reshape(hamiltonian, (self.n_site_tot*self.nstate, -1))


    def get_exciton_hamiltonian(self, nuclear_coordinates):
        self.exciton_hamiltonian = self.get_exciton_couplings(nuclear_coordinates)

        diagonal = self.get_exciton_diagonal(nuclear_coordinates)
        np.fill_diagonal(self.exciton_hamiltonian, diagonal)
        return self.exciton_hamiltonian


    def get_initial_coefficients(self, nuclear_coordinates):
        H = self.get_exciton_hamiltonian(nuclear_coordinates)
        #print_matrix('H:', H, 10)
        w, v = np.linalg.eigh(H)
        # the eigenvalues from eig function are not sorted
        # but it doesn't matter since we will get the largest weight vector later
        #arg = np.argsort(w)
        #w, v = w[arg], v[arg]

        weight = np.exp(- self.beta_b * w)
        probility = weight / np.sum(weight)

        #print_matrix('initial eigenvalues:', w, 10)
        #print_matrix('initial eigenvectors:', v, 10)
        arg = np.where(probility > .1)
        print('initial probility:', arg, probility[arg])
        self.coefficients = np.copy(v[arg[0]][0])
        #print_matrix('initial coefficients:', self.coefficients, 10)

        c2 = np.einsum('i,i->i', self.coefficients.conj(), self.coefficients)
        return np.reshape(c2, (self.n_site_tot, -1))


    def update_coefficients(self, nuclear_coordinates, nuclear_coordinates1=None, dt=None):
        if dt is None: dt = self.exciton_dt

        #hamiltonian = self.get_exciton_hamiltonian(nuclear_coordinates1)
        #delta = (1j * dt) * np.einsum('ij,j->i', hamiltonian, self.coefficients)

        #hamiltonian = self.get_exciton_hamiltonian(nuclear_coordinates)
        #delta2 = (1j * dt * dt *.5) * np.einsum('ij,j->i', hamiltonian, self.coefficients_dot)

        #self.coefficients -= (delta + delta2)
        #self.coefficients_dot = -1j * np.einsum('ij,j->i', hamiltonian, self.coefficients)

        hamiltonian = self.get_exciton_hamiltonian(nuclear_coordinates)
        #exp_h = np.exp((-1j * dt) * hamiltonian)
        w, v = np.linalg.eigh(hamiltonian)
        exp_h = np.exp((-1j * dt) * w)
        exp_h = np.einsum('ji,i,ki->jk', v, exp_h, v) # v is in Fortran-order
        self.coefficients = np.einsum('ij,j->i', exp_h, self.coefficients)

        c2 = np.einsum('i,i->i', self.coefficients.conj(), self.coefficients)
        return np.reshape(c2, (self.n_site_tot, -1)).real


    def cal_energy(self, hamiltonian=None, coefficients=None):
        if hamiltonian is None: hamiltonian = self.exciton_hamiltonian
        if coefficients is None: coefficients = self.coefficients

        energy = np.einsum('i,ij,j->', coefficients.conj(), hamiltonian, coefficients)
        #print('exciton energy (au):', energy.real, energy.imag)
        return energy.real


    def cal_force(self, nuclear_coordinates, coefficients=None):
        if coefficients is None: coefficients = self.coefficients
        coefficients = np.reshape(coefficients, (self.n_site_tot, -1))

        force = np.zeros((self.n_site_tot, self.n_mode))

        for i in range(self.n_site_tot-1):
            k = i % self.ntype
            c2 = np.einsum('i,j->ij', coefficients[i].conj(), coefficients[i+1])
            force[i] = -2.* np.einsum('mij,ij->m', self.coupling_a[k], c2.real)

        return force


    def cal_r_correlation(self, coefficients=None):
        if coefficients is None: coefficients = self.coefficients

        coefficients = np.reshape(coefficients, (self.n_site_tot, -1))
        c2 = np.einsum('ni,nj->nij', coefficients.conj(), coefficients)
        correlation = np.einsum('n,nij->', self.length**2, c2)
        correlation -= np.einsum('n,nij->', self.length, c2)**2
        #print('correlation: %8.6f %10.8f' % correlation.real, correlation.imag)
        return correlation.real



class Dynamics():
    def __init__(self, key, **kwargs):
        self.total_time = 1

        put_kwargs_to_keys(key, **kwargs)
        # only take the total_time here
        if 'total_time' in key.keys():
            self.total_time = key.pop('total_time')
        print('dynamics run %d steps in %.3f fs.' %(self.total_time, float(self.total_time*AU2FS)))

        self.ndstep = OscillatorDynamicsStep(key)
        self.edstep = ExcitonDynamicsStep(key)

        self.md_time_total_energies = np.zeros(self.total_time)
        self.correlation = np.zeros(self.total_time)
        self.c2 = []


    def kernel(self):
        self.ndstep.get_initial_coordinates_velocities(self.edstep.n_site_tot)
        coords = self.ndstep.nuclear_coordinates # equal sign used here as a pointer
        c2 = self.edstep.get_initial_coefficients(coords)
        self.c2.append(c2.ravel())

        self.md_time_total_energies[0] = self.edstep.cal_energy() + self.ndstep.nuclear_energy
        electronic_force = self.edstep.cal_force(coords)

        for ti in range(1, self.total_time):
            self.ndstep.update_nuclear_coords_velocity(electronic_force)
            c2 = self.edstep.update_coefficients(coords)
            self.c2.append(c2.ravel())#np.max(c2))
            self.correlation[ti] = self.edstep.cal_r_correlation()

            # velocity_verlet is cumbersome
            if self.ndstep.nuclear_update_method == 'velocity_verlet':
                self.ndstep.velocity_verlet_step(electronic_force, 2)

            self.md_time_total_energies[ti] = self.edstep.cal_energy() + self.ndstep.nuclear_energy
            electronic_force = self.edstep.cal_force(coords)

        #print_matrix('ground-state energy (eV):', self.energy*HARTREE2EV, 10)
        #print_matrix('coefficient weights:', np.array(self.c2), 10)
        #print_matrix('correlation:', self.correlation, 10)


    def plot_time_variables(self, fig_name=None):
        import matplotlib.pyplot as plt

        #fig_name = 'site21_md'
        dpi = 300 if fig_name else 100
        #fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(11,6), sharex=True)
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(11,6), sharex=False, dpi=dpi)

        time_line = np.linspace(0, self.total_time, self.total_time) * AU2FS

        variable = self.md_time_total_energies
        variable -= variable[0]
        ax[0].plot(time_line, variable)
        ax[0].set_ylabel('Energy (a.u.)')
        ax[0].set_xlabel('Time (fs)')

        n = len(self.c2)
        d = int(n/8)
        variable = self.c2[::d]
        line = range(1, len(variable[0])+1)
        for i in range(len(variable)):
            ax[1].plot(line, variable[i], label='%.1f fs' % float(d*i*AU2FS))

        ax[1].set_ylabel('Coefficients')
        ax[1].set_xlabel('Site (State) No.')
        ax[1].legend()

        plt.tight_layout()
        if fig_name:
            plt.savefig(fig_name)
        else:
            plt.show()



if __name__ == '__main__':
    total_time = 400
    key = {}

    n_mode = 6
    key['n_mode'] = n_mode
    key['nuclear_mass']  = [6., 6., 754., 754., 754., 754.] # amu
    key['nuclear_omega'] = [144., 148., 5., 5., 5., 5.] # meV

    n_site = np.array([11, 1, 1])
    distance = 8.64 #[8.64, 8.64, 8.64] # Angstrom
    nstate = 2
    key['n_site'] = n_site
    key['distance'] = distance
    key['nstate'] = nstate

    key['energy'] = [0., 10.] # meV

    coupling_g = np.zeros((n_mode, nstate))
    coupling_g[0,0] = 1821. # meV/AA
    coupling_g[1,1] = 2231. # meV/AA
    key['coupling_g'] = coupling_g

    # x, y, z axis.
    # (1,1,1) is center O, (0,1,1) and (2,1,1) is the left and right points on x-axis
    coupling_j = np.zeros((2, nstate, nstate))
    coupling_j[0,0,0] = -39. # meV # A dimer x to x+1
    coupling_j[0,0,1] = -13. # meV
    coupling_j[0,1,0] = -13. # meV
    coupling_j[0,1,1] = -24. # meV
    coupling_j[1,0,0] = -18. # meV # B dimer x to x-1
    coupling_j[1,0,1] = 18.  # meV
    coupling_j[1,1,0] = 18.  # meV
    coupling_j[1,1,1] = 13.  # meV
    key['coupling_j'] = coupling_j

    coupling_a = np.zeros((2, n_mode, nstate, nstate))
    coupling_a[0,2,0,0] = 71. # meV/AA # A dimer
    coupling_a[0,3,0,1] = 36. # meV/AA
    coupling_a[0,4,1,0] = 41. # meV/AA
    coupling_a[0,5,1,1] = 23. # meV/AA
    coupling_a[1,2,0,0] = 29. # meV/AA # B dimer
    coupling_a[1,3,0,1] = 28. # meV/AA
    coupling_a[1,4,1,0] = 29. # meV/AA
    coupling_a[1,5,1,1] = 37. # meV/AA
    key['coupling_a'] = coupling_a

    obj = Dynamics(key, total_time=total_time)
    obj.kernel()
    obj.plot_time_variables()
