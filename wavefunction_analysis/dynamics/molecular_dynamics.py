import sys
import numpy as np

from wavefunction_analysis.dynamics import ElectronicDynamicsStep, GrassmannElectronicDynamicsStep, CurvyElectronicDynamicsStep, ExtendedLagElectronicDynamicsStep, PhotonDynamicsStep
from wavefunction_analysis.utils import print_matrix, convert_units
from wavefunction_analysis.utils.sec_mole import get_molecular_center, get_moment_of_inertia
from wavefunction_analysis.plot import plt


#AU_TIME_IN_SEC = 2.0 * 6.6260693e-34 * 8.854187817e-12 * 5.291772108e-11 \
#                    / 1.60217653e-19 / 1.60217653e-19
kT_AU_to_Kelvin = 0.25 * 9.1093826e-31 * (1.60217653e-19*1.60217653e-19 * 8.854187817e-12 * 6.6260693e-34)**2 / 1.3806505e-23

#FS = 1.0e15
BOHR = 0.52917721067121
ELECTRON_MASS_IN_AMU = 5.4857990945e-04 # from qchem

def get_boltzmann_beta(temperature):
    from pyscf.data.nist import HARTREE2J, BOLTZMANN
    return HARTREE2J / (BOLTZMANN * temperature)


def angular_property(mass, coords, props):
    # props is for example force or velocity
    inertia = get_moment_of_inertia(mass, coords)

    # total angular property I^-1 * L
    U, s, Vt = np.linalg.svd(inertia)
    idx = np.where(s[np.abs(s)>1e-10])[0]
    s, U, Vt = 1./s[idx], U[:,idx], Vt[:,idx]
    ang_prop = np.einsum('ij,j,kj,k->i', Vt, s, U, props)

    return ang_prop


def remove_trans_rotat_velocity(velocity, mass, coords):
    # remove translation (ie. center of mass velocity)
    p_com = np.einsum('i,ix->x', mass, velocity) / len(mass)
    velocity -= np.einsum('i,x->ix', 1./mass, p_com)

    # move coords to com
    com = get_molecular_center(mass, coords)
    coords -= com

    # remove rotation
    ang_mom = np.einsum('i,ix->x', mass, np.cross(coords, velocity)) # r x v
    ang_vec = angular_property(mass, coords, ang_mom)

    velocity -= np.cross(ang_vec, coords)
    return velocity


def remove_trans_rotat_force(force, mass, coords):
    # remove translation (ie. center of mass force)
    f_com = np.sum(force, axis=0) / np.sum(mass)
    force -= np.einsum('i,x->ix', mass, f_com)

    # move coords to com
    com = get_molecular_center(mass, coords)
    coords -= com

    # remove rotation
    torque = np.sum(np.cross(coords, force), axis=0) # r x f
    ang_f = angular_property(mass, coords, torque)

    force -= np.einsum('i,ix->ix', mass, np.cross(ang_f[None,:], coords))
    return force


class NuclearDynamicsStep():
    """
    update_method has options
    euler, leapforg, velocity_verlet
    """

    def __init__(self, atmsym, init_coords,
            nuclear_dt, nuclear_update_method, nuclear_save_nframe,
            init_velocity=None, init_kick=None):

        #key.setdefault('temperature', 298) # system temperature
        #key.setdefault('nuclear_dt', 10)
        #key.setdefault('nuclear_update_method', 'velocity_verlet')
        #key.setdefault('nuclear_nuclear_save_nframe', 0)
        #key.setdefault('init_velocity', None)
        #key.setdefault('init_kick', None)

        #for name, value in key.items(): # put all the variables in the class
        #    setattr(self, name, value)

        self.nuclear_dt = nuclear_dt
        self.nuclear_update_method = nuclear_update_method

        self.atmsym = atmsym
        self.natoms = len(atmsym)

        self.nuclear_mass = np.zeros(self.natoms)
        # use pyscf's
        from pyscf.data import elements
        for i in range(self.natoms):
            self.nuclear_mass[i] = elements.MASSES[elements.charge(self.atmsym[i])] / ELECTRON_MASS_IN_AMU
        #print_matrix('nuclear_mass:\n', self.nuclear_mass)

        self.nuclear_save_nframe = nuclear_save_nframe

        self.nuclear_coordinate = np.reshape(np.copy(init_coords), (self.natoms, 3)) # copy!
        # assume init_coords in AA and change it to A.U.
        self.nuclear_coordinate /= BOHR

        if isinstance(init_velocity, list):
            init_velocity = np.reshape(init_velocity, (self.natoms, 3))
        elif init_velocity == 'random':
            init_velocity = self.init_velocity_random(2e-3)
        elif 'thermo' in init_velocity:
            init_velocity = self.init_velocity_thermo(float(init_velocity.split('_')[1]))

        if isinstance(init_velocity, np.ndarray):
            self.project_velocity(init_velocity)
            self.get_nuclear_kinetic_energy(self.nuclear_velocity)
        else: # no initial velocity
            self.nuclear_velocity = np.zeros((self.natoms, 3))
            self.nuclear_kinetic = 0.
            self.nuclear_temperature = 0.

        if init_kick:
            init_kick = np.reshape(init_kick, (self.natoms, 3))
            self.project_force(init_kick)
        else: # no initial force
            self.nuclear_force = np.zeros((self.natoms, 3))


    def init_velocity_random(self, etrans, sigma=1e-4, scale=.1, seed=12345):
        """
        random kinetic energy for atoms at three directions
        """
        size = 3* self.natoms

        #etrans = convert_units(etrans*scale, 'eh', 'kcal')
        mean = etrans / float(size)

        rng = np.random.default_rng(seed)
        # mean is the center
        # sigma is the standard deviation whose square is variance
        ek = rng.normal(loc=mean, scale=sigma, size=size)
        ek = np.abs(ek) * etrans / np.sum(ek) # scale by the generated kinetic energy

        sign = rng.random((self.natoms, 3))
        sign = np.where(sign>.5, 1, -1)
        velocity = 2.* np.einsum('ix,i->ix', ek.reshape(self.natoms, 3), 1./self.nuclear_mass)
        velocity = np.einsum('ix,ix->ix', sign, np.sqrt(velocity))
        return velocity


    def init_velocity_thermo(self, temp, seed=12345):
        """
        random velocity following Boltzmann distribution
        """
        beta_b = get_boltzmann_beta(temp)
        sigma = np.sqrt(1./beta_b/self.nuclear_mass)

        velocity = np.zeros((self.natoms, 3))

        rng = np.random.default_rng(seed)
        for i in range(self.natoms):
            velocity[i] = rng.normal(loc=0., scale=sigma[i], size=3)

        return velocity


    def update_nuclear_coords_velocity(self, nuclear_force):
        if self.nuclear_update_method == 'euler':
            self.euler_step(nuclear_force)
        elif self.nuclear_update_method == 'leapfrog':
            self.leapfrog_step(nuclear_force)
        elif self.nuclear_update_method == 'velocity_verlet':
            self.velocity_verlet_step(nuclear_force, 1)
            # we will finish the last falf after electronic step


    def update_nuclear_coords_velocity2(self, nuclear_force):
        # velocity_verlet is cumbersome
        if self.nuclear_update_method == 'velocity_verlet':
            self.velocity_verlet_step(nuclear_force, 2)

        self.project_velocity(self.nuclear_velocity)
        nuclear_force = self.project_force(nuclear_force)

        return nuclear_force


    def euler_step(self, nuclear_force):
        self.nuclear_velocity += self.nuclear_dt * np.einsum('ix,i->ix', nuclear_force, 1./self.nuclear_mass)
        self.nuclear_coordinate += self.nuclear_dt * self.nuclear_velocity
        self.get_nuclear_kinetic_energy(self.nuclear_velocity)


    def leapfrog_step(self, nuclear_force):
        old_nuclear_velocity = np.copy(self.nuclear_velocity)
        self.nuclear_velocity += self.nuclear_dt * np.einsum('ix,i->ix', nuclear_force, 1./self.nuclear_mass)
        self.nuclear_coordinate += self.nuclear_dt * self.nuclear_velocity

        average_nuclear_velocity = .5 * (old_nuclear_velocity + self.nuclear_velocity)
        self.get_nuclear_kinetic_energy(average_nuclear_velocity)


    def velocity_verlet_step(self, nuclear_force, half):
        self.nuclear_velocity += .5 * self.nuclear_dt * np.einsum('ix,i->ix', nuclear_force, 1./self.nuclear_mass)
        if half == 1:
            self.nuclear_coordinate += self.nuclear_dt * self.nuclear_velocity
        if half == 2:
            self.get_nuclear_kinetic_energy(self.nuclear_velocity)


    def get_nuclear_kinetic_energy(self, velocity, mass=None):
        if mass is None: mass = self.nuclear_mass

        v2 = np.einsum('ix,ix->i', velocity, velocity)
        self.nuclear_kinetic = .5* np.einsum('i,i', mass, v2)
        self.nuclear_temperature = self.nuclear_kinetic * 2. / (velocity.size)


    def project_velocity(self, velocity, mass=None, coords=None):
        if mass is None: mass = self.nuclear_mass
        if coords is None: coords = self.nuclear_coordinate

        self.nuclear_velocity = remove_trans_rotat_velocity(velocity, mass, coords)


    def project_force(self, nuclear_force, mass=None, coords=None):
        if mass is None: mass = self.nuclear_mass
        if coords is None: coords = self.nuclear_coordinate

        self.nuclear_force = remove_trans_rotat_force(nuclear_force, mass, coords)
        return self.nuclear_force



class MolecularDynamics():
    def __init__(self, key):
        atmsym                = key.get('atmsym', None)
        if atmsym == None:
            raise AttributeError('no molecule symbols given')
        init_coords           = key.get('init_coords', None)
        if init_coords is None:
            raise ValueError('no initial molecule coordinate given')

        self.ed_method        = key.get('ed_method', 'normal')
        self.ph_method        = key.get('ph_method', None)

        self.total_time       = key.get('total_time', 4000) # au
        self.nuclear_dt       = key.get('nuclear_dt', 10) # au
        nuclear_update_method = key.get('nuclear_update_method', 'velocity_verlet')
        nuclear_save_nframe   = key.get('nuclear_save_nframe', 0)
        init_velocity         = key.get('init_velocity', None)
        init_kick             = key.get('init_kick', None)


        self.electronic_dt    = key.get('electronic_dt', 0)

        self.nuclear_nsteps = int(self.total_time/self.nuclear_dt) + 1
        print('running molecular dynamics in\n%4d steps, total time %6.3f fs\n' % (self.nuclear_nsteps, convert_units(self.total_time, 'au', 'fs')))

        self.ndstep = NuclearDynamicsStep(atmsym, init_coords, self.nuclear_dt,
                                          nuclear_update_method, nuclear_save_nframe,
                                          init_velocity, init_kick)

        self.edstep = self.set_electronic_step(key, self.ed_method)
        self.phstep = self.set_photon_step(key, self.ph_method)

        self.md_time_total_energies = np.zeros(self.nuclear_nsteps)
        self.md_time_coordinate = np.zeros((self.nuclear_nsteps, self.ndstep.natoms, 3))
        self.md_time_dipoles = np.zeros((self.nuclear_nsteps, 3))


    def set_electronic_step(self, key, ed_method='normal'):
        if self.ed_method == 'extended_lag':
            return ExtendedLagElectronicDynamicsStep(key)
        elif self.ed_method == 'curvy':
            key['electronic_dt'] = nuclear_dt
            key['electronic_update_method'] = key.get('nuclear_update_method', 'velocity_verlet')
            return CurvyElectronicDynamicsStep(key)
        elif self.ed_method == 'grassmann':
            return GrassmannElectronicDynamicsStep(key)
        else:
            return ElectronicDynamicsStep(key)


    def set_photon_step(self, key, ph_method=None):
        if ph_method == 'quantum':
            return PhotonDynamicsStep(key)
        else:
            return None


    def run_dynamics(self):
        print('current time:%7.3f fs' % 0.0)
        coords = self.ndstep.nuclear_coordinate # equal sign used here as a pointer
        # coords will change when self.ndstep.nuclear_coordinate changes!

        kwargs = {}
        if self.phstep:
            kwargs['c_lambda'] = self.phstep.coupling_strength

        et, electronic_force = self.edstep.init_electronic_density_static(coords, **kwargs)
        electronic_force = self.ndstep.project_force(electronic_force)

        if self.phstep:
            kwargs['frequency'] = self.phstep.photon_frequency
            self.md_time_dipoles[0] = self.edstep.mf.dip_moment(unit='au')
            kwargs.update(self.phstep.update_photon_density(self.md_time_dipoles[0], self.ndstep.nuclear_dt))
        photon_energy = kwargs.get('photon_energy', 0.)

        self.md_time_coordinate[0] = coords
        self.md_time_total_energies[0] = et + self.ndstep.nuclear_kinetic + photon_energy

        print('temperature: %4.2f K' % float(self.ndstep.nuclear_temperature * kT_AU_to_Kelvin))
        print('potential energy: %15.10f  kinetic energy: %15.10f  total energy: %15.10f' % (et, self.ndstep.nuclear_kinetic, self.md_time_total_energies[0]))
        print_matrix('force:\n', self.edstep.electronic_force)
        print_matrix('velocity:\n', self.ndstep.nuclear_velocity)
        print_matrix('current nuclear coordinate:\n', coords*BOHR)


        # loop times
        for ti in range(1, self.nuclear_nsteps):
            self.ndstep.update_nuclear_coords_velocity(electronic_force)
            et, electronic_force = self.edstep.update_electronic_density_static(coords, **kwargs)

            if self.phstep:
                self.md_time_dipoles[ti] = self.edstep.mf.dip_moment(unit='au')
                kwargs.update(self.phstep.update_photon_density(self.md_time_dipoles[ti], self.ndstep.nuclear_dt))
            photon_energy = kwargs.get('photon_energy', 0.)

            print('current time:%7.3f fs' % convert_units(ti*self.nuclear_dt, 'au', 'fs'))
            #coords = self.ndstep.nuclear_coordinate # dont need to reassign
            print_matrix('current nuclear coordinate:\n', coords*BOHR)

            # velocity_verlet is cumbersome
            electronic_force = self.ndstep.update_nuclear_coords_velocity2(electronic_force)

            if self.ed_method == 'curvy':
                et, electronic_force = self.edstep.update_electronic_density_static2(coords, **kwargs)

            self.md_time_coordinate[ti] = coords
            self.md_time_total_energies[ti] = et + self.ndstep.nuclear_kinetic + photon_energy #+ self.edstep.electronic_kinetic

            print('temperature: %4.2f K' % float(self.ndstep.nuclear_temperature * kT_AU_to_Kelvin))
            print('potential energy: %15.10f  kinetic energy: %15.10f  total energy %15.10f' % (et, self.ndstep.nuclear_kinetic, self.md_time_total_energies[ti]))
            print_matrix('force:\n', self.edstep.electronic_force)
            print_matrix('velocity:\n', self.ndstep.nuclear_velocity)


    def plot_time_variables(self, fig_name=None):
        time_line = np.linspace(0, convert_units(self.total_time, 'au', 'fs'), self.nuclear_nsteps)
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(11,6), sharex=True)

        coords = self.md_time_coordinate * BOHR
        energy = convert_units(self.md_time_total_energies, 'hartree', 'kcal')

        ax[0].plot(time_line, coords[:,1,-1]-coords[:,0,-1])
        ax[0].set_ylabel('He--H$^+$ ($\\AA$)')
        ax[1].plot(time_line, energy-energy[0])
        ax[1].set_xlabel('Time (fs)')
        ax[1].set_ylabel('$\\Delta$E (kcal/mol)')

        plt.tight_layout()
        if fig_name:
            plt.savefig(fig_name)
        else:
            plt.show()


def plot_time_variables(total_time, nuclear_nsteps, dists, energies):
    time_line = np.linspace(0, convert_units(total_time, 'au', 'fs'), nuclear_nsteps)
    method = ['BO', 'XL-3', 'XL-6', 'XL-9', 'Curvy']

    dists = np.array(dists) * BOHR
    energies = np.array(energies)
    energies -= energies[0,0]

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(11,6), sharex=True)
    for i in range(dists.shape[0]):
        ax[0].plot(time_line, dists[i], label=method[i])
        ax[0].set_ylabel('He--H$^+$ Length ($\\AA$)')
        ax[0].legend()

    for i in range(energies.shape[0]):
        ax[1].plot(time_line, energies[i], label=method[i])
        ax[1].set_xlabel('Time (fs)')
        ax[1].set_ylabel('Energy (a.u.)')
        ax[1].legend()

    plt.tight_layout()
    plt.savefig('dynamics')



if __name__ == '__main__':
    mdtype = int(sys.argv[1])

    key = {}
    key['functional'] = 'hf'
    #key['basis'] = '3-21g'
    #key['charge'] = 0
    #key['atmsym'] = [1, 1]
    #key['init_coords'] = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]]
    key['basis'] = 'sto-3g'
    key['charge'] = 1
    key['atmsym'] = ['H', 'He']
    key['init_coords'] = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.929352]]
    key['init_velocity'] = [[0.0, 0.0, 0.0008], [0.0, 0.0, -0.0008]]

    key['nuclear_dt'] = 10
    #key['total_time'] = 120
    key['total_time'] = 4000
    key['nuclear_update_method'] = 'velocity_verlet'
    key['ortho_method'] = 'lowdin' # 'cholesky'

    if mdtype == 0:
        md = MolecularDynamics(key)
        md.run_dynamics()
        md.plot_time_variables(fig_name='normal_time_coords')

    elif mdtype == 1:
        print('run extended_lag')
        key['ed_method'] = 'extended_lag'
        md = MolecularDynamics(key)
        md.run_dynamics()
        md.plot_time_variables(fig_name='extended_lag_time_coords')

    elif mdtype == 2:
        print('run curvy')
        key['ed_method'] = 'curvy'
        key['ortho_method'] = 'cholesky'
        md = MolecularDynamics(key)
        md.run_dynamics()
        md.plot_time_variables(fig_name='curvy_time_coords')

    elif mdtype == 3:
        key['ed_method'] = 'grassmann'
        md = MolecularDynamics(key)
        md.run_dynamics()
        md.plot_time_variables(fig_name='grassmann_time_coords')

    elif mdtype == 4:
        dists = []
        energies = []

        md = MolecularDynamics(key)
        md.run_dynamics()

        dists.append( md.md_time_coordinate[:,1,-1] - md.md_time_coordinate[:,0,-1])
        energies.append( md.md_time_total_energies)

        key['ed_method'] = 'extended_lag'
        for xl_nk in [3, 6, 9]:

            key['xl_nk'] = xl_nk
            md = MolecularDynamics(key)
            md.run_dynamics()

            dists.append( md.md_time_coordinate[:,1,-1] - md.md_time_coordinate[:,0,-1])
            energies.append( md.md_time_total_energies)
            energies.append( md.md_time_total_energies2)


#        key['ed_method'] = 'curvy'
#        key['ortho_method'] = 'cholesky'
#        md = MolecularDynamics(key)
#        md.run_dynamics()
#        dists.append( md.md_time_coordinate[:,1,-1] - md.md_time_coordinate[:,0,-1])
#        energies.append( md.md_time_total_energies)

        key['ed_method'] = 'grassmann'
        key['ortho_method'] = 'lowdin'
        md = MolecularDynamics(key)
        md.run_dynamics()
        dists.append( md.md_time_coordinate[:,1,-1] - md.md_time_coordinate[:,0,-1])
        energies.append( md.md_time_total_energies)


        np.savetxt('bond.txt', np.array(dists))
        np.savetxt('energy.txt', np.array(energies))
        plot_time_variables(md.total_time, md.nuclear_nsteps, dists, energies)

    elif mdtype == 5:
        dists = np.loadtxt('bond.txt')
        energies = np.loadtxt('energy.txt')
        total_time, nuclear_dt = key['total_time'], key['nuclear_dt']
        plot_time_variables(total_time, int(total_time/nuclear_dt) + 1, dists, energies)


