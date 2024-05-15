import sys
import numpy as np

from wavefunction_analysis.dynamics import (
        ElectronicDynamicsStep,
        GrassmannElectronicDynamicsStep,
        CurvyElectronicDynamicsStep,
        ExtendedLagElectronicDynamicsStep,
        ExcitonDynamicsStep,
        PhotonDynamicsStep,
        PhotonDynamicsStep2,
        NuclearDynamicsStep,
        OscillatorDynamicsStep,
        )
from wavefunction_analysis.utils import print_matrix, convert_units
from wavefunction_analysis.plot import plt

kT_AU_to_Kelvin = 0.25 * 9.1093826e-31 * (1.60217653e-19**2 / 8.854187817e-12 / 6.6260693e-34)**2 / 1.3806505e-23


class MolecularDynamics():
    def __init__(self, key, ed_key={}, ph_key={}, **kwargs):
        self.ed_method  = key.pop('ed_method', 'normal')
        self.ph_method  = key.pop('ph_method', None)

        self.total_time = key.pop('total_time', 4000) # au
        self.dt         = key.get('dt', 10) # au

        self.nsteps = int(self.total_time/self.dt) + 1
        print('running molecular dynamics in %5d steps and total time %6.3f fs\n' % (self.nsteps, convert_units(self.total_time, 'au', 'fs')))

        self.ndstep = self.set_nuclear_step(key)
        self.edstep = self.set_electronic_step(ed_key, atmsym=key['atmsym'], **kwargs)
        self.phstep = self.set_photon_step(ph_key, dt=self.dt)

        self.md_time_total_energy = np.zeros(self.nsteps)
        self.md_time_coordinate = np.zeros((self.nsteps, self.ndstep.natoms, 3))
        self.md_time_dipoles = np.zeros((self.nsteps, 3))


    def set_nuclear_step(self, key):
        if self.ed_method == 'exciton':
            return OscillatorDynamicsStep(key)
        else:
            return NuclearDynamicsStep(key)


    def set_electronic_step(self, key, **kwargs):
        if self.ed_method == 'extended_lag':
            return ExtendedLagElectronicDynamicsStep(key, **kwargs)
        elif self.ed_method == 'curvy':
            return CurvyElectronicDynamicsStep(key, **kwargs)
        elif self.ed_method == 'grassmann':
            return GrassmannElectronicDynamicsStep(key, **kwargs)
        elif self.ed_method == 'normal':
            return ElectronicDynamicsStep(key, **kwargs)
        elif self.ed_method == 'exciton':
            return ExcitonDynamicsStep(key, **kwargs)


    def set_photon_step(self, key, **kwargs):
        if self.ph_method == 'quantum':
            return PhotonDynamicsStep(key, **kwargs)
        if self.ph_method == 'quantum2':
            return PhotonDynamicsStep2(key, **kwargs)
        else:
            return None


    def run_dynamics(self):
        coords = self.ndstep.coordinate # equal sign used here as a pointer!
        # coords will change when self.ndstep.coordinate changes!

        kwargs = {}
        if self.phstep:
            kwargs['c_lambda'] = self.phstep.c_lambda

        et, force = self.edstep.init_electronic_density_static(coords, **kwargs)
        force = self.ndstep.project_force(force)
        dipole = self.edstep.mf.dip_moment(unit='au')

        if self.phstep:
            print('photon frequency (au) is:', self.phstep.frequency)
            kwargs['frequency'] = self.phstep.frequency
            kwargs.update(self.phstep.update_density(dipole, self.ndstep.dt))
        photon_energy = kwargs.get('photon_energy', 0.)
        etot = et + self.ndstep.kinetic + np.sum(photon_energy)

        self.md_time_coordinate[0] = coords
        self.md_time_dipoles[0] = dipole
        self.md_time_total_energy[0] = etot

        print('current time:%7.3f fs' % 0.0, end='  ')
        print('temperature: %4.2f K' % float(self.ndstep.temperature * kT_AU_to_Kelvin))
        print('total energy (au): %15.10f  potential: %15.10f  kinetic: %15.10f' % (etot, et, self.ndstep.kinetic), end='  ')
        if self.phstep: print('photon: %15.10f %15.10f' % (photon_energy[0], photon_energy[1]))
        else: print('')
        print_matrix('force:\n', self.ndstep.force)
        print_matrix('velocity:\n', self.ndstep.velocity)
        print_matrix('nuclear coordinate:\n', convert_units(coords, 'bohr', 'angstrom'))


        # loop times
        for ti in range(1, self.nsteps):
            self.ndstep.update_coordinate_velocity(force, 1)
            et, force = self.edstep.update_electronic_density_static(coords, **kwargs)
            dipole = self.edstep.mf.dip_moment(unit='au')

            if self.phstep:
                kwargs.update(self.phstep.update_density(self.md_time_dipoles[ti-1], self.ndstep.dt, 1))

            #coords = self.ndstep.coordinate # dont need to reassign!

            if self.ed_method == 'curvy':
                et, force = self.edstep.update_electronic_density_static2(coords, **kwargs)

            # velocity_verlet is cumbersome
            force = self.ndstep.update_coordinate_velocity(force, 2)

            if self.phstep:
                kwargs.update(self.phstep.update_density(self.md_time_dipoles[ti-1], self.ndstep.dt, 2))
            photon_energy = kwargs.get('photon_energy', 0.)
            etot = et + self.ndstep.kinetic + np.sum(photon_energy)

            self.md_time_coordinate[ti] = coords
            self.md_time_dipoles[ti] = dipole
            self.md_time_total_energy[ti] = etot

            print('current time:%7.3f fs' % convert_units(ti*self.dt, 'au', 'fs'), end='  ')
            print('temperature: %4.2f K' % float(self.ndstep.temperature * kT_AU_to_Kelvin))
            print('total energy (au): %15.10f  potential: %15.10f  kinetic: %15.10f' % (etot, et, self.ndstep.kinetic), end='  ')
            #if self.phstep: print('photon: %15.10f' % photon_energy)
            if self.phstep: print('photon: %15.10f %15.10f' % (photon_energy[0], photon_energy[1]))
            else: print('')
            print_matrix('force:\n', self.ndstep.force)
            print_matrix('velocity:\n', self.ndstep.velocity)
            print_matrix('nuclear coordinate:\n', convert_units(coords, 'bohr', 'angstrom'))


    def plot_time_variables(self, fig_name=None):
        time_line = np.linspace(0, convert_units(self.total_time, 'au', 'fs'), self.nsteps)
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(11,6), sharex=True)

        coords = convert_units(self.md_time_coordinate, 'bohr', 'angstrom')
        energy = convert_units(self.md_time_total_energy, 'hartree', 'kcal')

        ax[0].plot(time_line, np.linalg.norm(coords[:,1]-coords[:,0], axis=1))
        ax[0].set_ylabel('He--H$^+$ ($\\AA$)')
        ax[1].plot(time_line, energy-energy[0])
        ax[1].set_xlabel('Time (fs)')
        ax[1].set_ylabel('$\\Delta$E (kcal/mol)')

        plt.tight_layout()
        if fig_name:
            plt.savefig(fig_name)
        else:
            plt.show()


def plot_time_variables(total_time, nsteps, dists, energies):
    time_line = np.linspace(0, convert_units(total_time, 'au', 'fs'), nsteps)
    method = ['BO', 'XL-3', 'XL-6', 'XL-9', 'Curvy']

    dists = convert_units(np.array(dists), 'bohr', 'angstrom')
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
    key['atmsym'] = ['H', 'He']
    key['coordinate'] = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.929352]]
    key['velocity'] = [[0.0, 0.0, 0.0008], [0.0, 0.0, -0.0008]]

    key['dt'] = 10
    #key['total_time'] = 120
    key['total_time'] = 4000
    key['update_method'] = 'velocity_verlet'

    ed_key = {}
    ed_key['functional'] = 'hf'
    #ed_key['basis'] = '3-21g'
    #ed_key['charge'] = 0
    ed_key['basis'] = 'sto-3g'
    ed_key['charge'] = 1
    ed_key['ortho_method'] = 'lowdin' # 'cholesky'

    if mdtype == 0:
        md = MolecularDynamics(key, ed_key)
        md.run_dynamics()
#        md.plot_time_variables(fig_name='normal_time_coords')

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
        energies.append( md.md_time_total_energy)

        key['ed_method'] = 'extended_lag'
        for xl_nk in [3, 6, 9]:

            key['xl_nk'] = xl_nk
            md = MolecularDynamics(key)
            md.run_dynamics()

            dists.append( md.md_time_coordinate[:,1,-1] - md.md_time_coordinate[:,0,-1])
            energies.append( md.md_time_total_energy)
            energies.append( md.md_time_total_energy2)


#        key['ed_method'] = 'curvy'
#        key['ortho_method'] = 'cholesky'
#        md = MolecularDynamics(key)
#        md.run_dynamics()
#        dists.append( md.md_time_coordinate[:,1,-1] - md.md_time_coordinate[:,0,-1])
#        energies.append( md.md_time_total_energy)

        key['ed_method'] = 'grassmann'
        key['ortho_method'] = 'lowdin'
        md = MolecularDynamics(key)
        md.run_dynamics()
        dists.append( md.md_time_coordinate[:,1,-1] - md.md_time_coordinate[:,0,-1])
        energies.append( md.md_time_total_energy)


        np.savetxt('bond.txt', np.array(dists))
        np.savetxt('energy.txt', np.array(energies))
        plot_time_variables(md.total_time, md.nsteps, dists, energies)

    elif mdtype == 5:
        dists = np.loadtxt('bond.txt')
        energies = np.loadtxt('energy.txt')
        total_time, dt = key['total_time'], key['dt']
        plot_time_variables(total_time, int(total_time/dt) + 1, dists, energies)


