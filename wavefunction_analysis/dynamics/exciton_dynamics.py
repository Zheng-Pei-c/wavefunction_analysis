import sys
import warnings
from pyscf.data.nist import HARTREE2J, HARTREE2EV, BOLTZMANN, AMU2AU, BOHR, PLANCK, E_CHARGE

from wavefunction_analysis import np, itertools
from wavefunction_analysis.utils import print_matrix
from wavefunction_analysis.utils import put_keys_kwargs_to_object, put_kwargs_to_keys
from wavefunction_analysis.dynamics import harmonic_oscillator
from wavefunction_analysis.dynamics.dimers_in_crystal import add_molecules_cell, read_unit_cell_info

from wavefunction_analysis.transport.read_parameters import read_energy_coupling


AU2FS = 2.e5 * 8.854187817e-12 * PLANCK * BOHR / E_CHARGE**2

def get_boltzmann_beta(temperature):
    return HARTREE2J / (BOLTZMANN * temperature)


"""
refer to Alessandro Troisi and Giorgio Orlandi PRL 2006 10.1103/PhysRevLett.96.086601
         Fornari, et. al. JPCC 2016 10.1021/acs.jpcc.6b01298
"""
class OscillatorDynamicsStep(harmonic_oscillator):
    def convert_parameter_units(self, unit_dict):
        """
        required input:
            mass in amu
            frequency in meV
        """
        # Boltzmann coefficient is 1/(k_B * T)
        self.beta_b = get_boltzmann_beta(self.init_temp)
        self.mass *= AMU2AU
        self.frequency /= (HARTREE2EV*1000)
        self.omega2 = self.frequency**2 # for computational efficiency

        if self.debug > 0:
            #print('KT:', 1./self.beta_b, 1./self.beta_b*HARTREE2EV*1000)
            print_matrix('oscillator mass (au):', self.mass)
            print_matrix('oscillator omega (au):', self.frequency)


    def get_phonon_hamiltonian(self, velocity=None, coordinate=None, mass=None, omega2=None):
        # get phonon energy on each site
        if velocity is None: velocity = self.velocity
        if coordinate is None: coordinate = self.coordinate
        if mass is None: mass = self.mass
        if omega2 is None: omega2 = self.omega2

        # the first index of velocity and coordinate is mass related
        mass2 = np.copy(mass) *.5
        self.hamiltonian = np.einsum('i,in,in->n', mass2, velocity, velocity)
        v2 = np.einsum('i,i->i', mass2, omega2)
        self.hamiltonian += np.einsum('i,in,in->n', v2, coordinate, coordinate)

        return self.hamiltonian



class ExcitonDynamicsStep():
    def __init__(self, key={}, **kwargs):
        """
        required input:
            nstate is a number
            n_site as a number or 1d, 2d, or 3d int array
            distance as a number or 1d, 2d, or 3d float array in Angstrom
            energy as an array in meV
            coupling_g on site coupling between vibrational modes has dimension (n_mode, nstate) in meV / AA
            coupling_j neighboring site coupling between excitons in meV
            coupling_a neighboring site coupling between vibrational modes in meV
            assume coupling_j and coupling_a have same first dimension
            dimer_label provide the neighboring dimer indices
        """
        self.debug = 1
        self.temperature = 298 # K

        # exciton dynamics time step
        self.exciton_dt = 1 # au
        # 3D molecular unit cell number
        self.n_site = 0
        # number of molecules in a unit cell
        self.n_mol = 0
        # vibrational mode number of each site
        self.n_mode = 0
        # exciton number of each site
        self.nstate = 0

        #TODO: move n_mol to unit_cell dict
        # unit cell informations
        # including intermolecular distance: abc
        # angle, element, and coordinate: scale
        self.unit_cell = {'abc': 0, 'angle': 0, 'element': 0, 'scale': 0, 'n_site': [5,5,5]}

        # exciton energy of each site
        self.energy = 0
        # three couplings
        self.coupling_g = 0 # (nmode, nstate)
        self.coupling_j = 0 # (ndimer, nstate, nstate)
        self.coupling_a = 0 # (ndimer, nmode, nstate, nstate)

        self.dimer_label = None

        put_keys_kwargs_to_object(self, key, **kwargs)

        if getattr(self, 'cif'):
            self.get_unit_cell_info(self.cif)

        self.check_sanity()
        # convert the input parameters into atomic unit for the calculations
        self.convert_parameter_units(getattr(self, 'unit_dict', None))
        self.process_parameters()


    def get_unit_cell_info(self, cif):
        abc, angle, element, scale = read_unit_cell_info(cif)
        self.unit_cell.update({'abc': abc, 'angle': angle, 'element': element, 'scale': scale})


    def check_sanity(self):
        """
        check the input parameters
        """
        if self.n_mode == 0:
            raise ValueError('n_mode in %s should be larger than 0' % self.__class__.__name__)


    def convert_parameter_units(self, unit_dict):
        self.beta_b = get_boltzmann_beta(self.temperature)
        self.n_site_tot = np.prod(self.n_site) * self.n_mol

        self.energy /= (HARTREE2EV*1000)
        self.coupling_g /= (HARTREE2EV*1000/BOHR)
        self.coupling_j /= (HARTREE2EV*1000)
        self.coupling_a /= (HARTREE2EV*1000/BOHR)

        # number of different dimers
        # used for both coupling_j and coupling_a
        self.ntype = self.coupling_j.shape[0]

        if self.debug > 0:
            print_matrix('on-site exciton energy (au):', self.energy, 10)
            print_matrix('on-site exciton-phonon coupling (au):', self.coupling_g, 10)
            print_matrix('off-site exciton-exciton coupling (au):', self.coupling_j.reshape(-1, self.nstate**2), 10)
            print_matrix('off-site exciton-phonon-exciton coupling (au):', self.coupling_a.reshape(-1, self.nstate**2), 10)


    def process_parameters(self):
        self.length = np.linspace(0, self.unit_cell['abc']*self.n_site_tot, self.n_site_tot)
        self.length -= np.average(self.length) # move center

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


    def get_exciton_hamiltonian0(self):
        n_site_tot, nstate = self.n_site_tot, self.nstate
        hamiltonian = np.zeros((n_site_tot, nstate, n_site_tot, nstate))

        for i in range(n_site_tot-1):
            k = i % self.ntype
            hamiltonian[i,:,i+1] = self.coupling_j[k]
            hamiltonian[i+1,:,i] = hamiltonian[i,:,i+1].transpose()

        hamiltonian = hamiltonian.reshape(n_site_tot*nstate, -1)
        np.fill_diagonal(hamiltonian, np.tile(self.energy, (n_site_tot, 1)).ravel())

        self.hamiltonian0 = hamiltonian
        return self.hamiltonian0


    def get_exciton_hamiltonian1(self, coordinate):
        n_site_tot, nstate = self.n_site_tot, self.nstate
        diagonal = np.einsum('mi,mn->ni', self.coupling_g, coordinate)

        hamiltonian = np.zeros((n_site_tot, nstate, n_site_tot, nstate))

        coordinate1 = np.copy(coordinate)
        coordinate1[:,:-1] -= coordinate[:,1:]

        for i in range(n_site_tot-1):
            k = i % self.ntype
            coupling = np.einsum('mij,m->ij', self.coupling_a[k], coordinate1[:,i])
            hamiltonian[i,:,i+1] = coupling
            hamiltonian[i+1,:,i] = hamiltonian[i,:,i+1].transpose()

        hamiltonian = hamiltonian.reshape(n_site_tot*nstate, -1)
        np.fill_diagonal(hamiltonian, diagonal.ravel())
        return hamiltonian


    def get_exciton_hamiltonian2(self, coordinate):
        self.hamiltonian = self.get_exciton_hamiltonian1(coordinate)
        self.hamiltonian += self.hamiltonian0
        return self.hamiltonian


    def get_exciton_diagonal(self, coordinate):
        diagonal = np.einsum('mi,mn->ni', self.coupling_g, coordinate)
        diagonal += np.tile(self.energy, (self.n_site_tot, 1))
        return diagonal.ravel()


    def get_exciton_couplings(self, coordinate):
        n_site_tot, nstate = self.n_site_tot, self.nstate
        hamiltonian = np.zeros((n_site_tot, nstate, n_site_tot, nstate))

        coordinate1 = np.copy(coordinate)
        coordinate1[:,:-1] -= coordinate[:,1:]

        for i in range(n_site_tot-1):
            k = i % self.ntype
            coupling = np.einsum('mij,m->ij', self.coupling_a[k], coordinate1[:,i])
            hamiltonian[i,:,i+1] = self.coupling_j[k] + coupling
            hamiltonian[i+1,:,i] = hamiltonian[i,:,i+1].transpose()

        return np.reshape(hamiltonian, (n_site_tot*nstate, -1))


    def get_exciton_hamiltonian(self, coordinate):
        self.hamiltonian = self.get_exciton_couplings(coordinate)
        print_matrix('exciton_hamiltonian:', self.hamiltonian)

        diagonal = self.get_exciton_diagonal(coordinate)
        np.fill_diagonal(self.hamiltonian, diagonal)
        return self.hamiltonian


    def get_initial_coefficients(self, coordinate):
        H = self.get_exciton_hamiltonian(coordinate)
        if self.debug > 5: # check if the two approach give same hamiltonian
            self.get_exciton_hamiltonian0()
            H_t = self.get_exciton_hamiltonian2(coordinate) - H
            print_matrix('hamiltonian difference: '+str(np.count_nonzero(H_t)), H_t, 10)

        ##print_matrix('H:', H, 10)
        #w, v = np.linalg.eigh(H)
        ## the eigenvalues from eig function are not sorted
        ## but it doesn't matter since we will get the largest weight vector later
        ##arg = np.argsort(w)
        ##w, v = w[arg], v[arg]

        #weight = np.exp(- self.beta_b * w)
        #probility = weight / np.sum(weight)

        ##print_matrix('initial eigenvalues:', w, 10)
        ##print_matrix('initial eigenvectors:', v, 10)
        #arg = np.where(probility > .1)
        #print('initial probility:', arg, probility[arg])
        #self.coefficients = np.copy(v[arg[0]][0])
        self.coefficients = np.zeros(self.n_site_tot*self.nstate)
        self.coefficients[int(self.n_site_tot*self.nstate//4)] = 1.
        print_matrix('initial coefficients:', self.coefficients, 10)

        c2 = np.einsum('i,i->i', self.coefficients.conj(), self.coefficients)
        return c2 #np.reshape(c2, ((self.n_site_tot, -1)))


    def update_coefficients(self, coordinate, coordinate1=None, dt=None):
        if dt is None: dt = self.exciton_dt

        #hamiltonian = self.get_exciton_hamiltonian(coordinate1)
        #delta = (1j * dt) * np.einsum('ij,j->i', hamiltonian, self.coefficients)

        #hamiltonian = self.get_exciton_hamiltonian(coordinate)
        #delta2 = (1j * dt * dt *.5) * np.einsum('ij,j->i', hamiltonian, self.coefficients_dot)

        #self.coefficients -= (delta + delta2)
        #self.coefficients_dot = -1j * np.einsum('ij,j->i', hamiltonian, self.coefficients)

        hamiltonian = self.get_exciton_hamiltonian(coordinate)
        if self.debug > 5: # check if the two approach give same hamiltonian
            hamiltonian_t = self.get_exciton_hamiltonian2(coordinate) - hamiltonian
            print_matrix('hamiltonian difference: '+str(np.count_nonzero(hamiltonian_t)), hamiltonian_t, 10)

        #exp_h = np.exp((-1j * dt) * hamiltonian)
        w, v = np.linalg.eigh(hamiltonian)
        exp_h = np.exp((-1j * dt) * w)
        exp_h = np.einsum('ji,i,ki->jk', v, exp_h, v) # v is in Fortran-order
        self.coefficients = np.einsum('ij,j->i', exp_h, self.coefficients)

        c2 = np.einsum('i,i->i', self.coefficients.conj(), self.coefficients)
        return c2.real #np.reshape(c2, ((self.n_site_tot, -1))).real


    def cal_energy(self, hamiltonian=None, coefficients=None):
        if hamiltonian is None: hamiltonian = self.hamiltonian
        if coefficients is None: coefficients = self.coefficients

        energy = np.einsum('i,ij,j->', coefficients.conj(), hamiltonian, coefficients)
        #print('exciton energy (au):', energy.real, energy.imag)
        return energy.real


    def cal_force(self, coordinate, coefficients=None):
        if coefficients is None: coefficients = self.coefficients
        coefficients = np.reshape(coefficients, (self.n_site_tot, -1))

        #force = np.zeros((self.n_site_tot, self.n_mode))

        c2 = np.einsum('ni,ni->ni', coefficients.conj(), coefficients)
        force = - np.einsum('mi,ni->mn', self.coupling_g, c2.real)
        for i in range(self.n_site_tot-1):
            k = i % self.ntype
            c2 = np.einsum('i,j->ij', coefficients[i].conj(), coefficients[i+1])
            c2 += c2.conj().T
            g = np.einsum('mij,ij->m', self.coupling_a[k], c2.real)

            force[:,i] -= g
            force[:,i+1] += g # shift site and change sign

        return force


    def cal_r_correlation(self, coefficients=None, c2=None):
        if c2 is None:
            if coefficients is None: coefficients = self.coefficients
            c2 = np.einsum('i,i->i', coefficients.conj(), coefficients)

        c2 = np.reshape(c2, (self.n_site_tot, -1))
        correlation = np.einsum('n,ni->', self.length**2, c2)
        correlation -= np.einsum('n,ni->', self.length, c2)**2
        #print('correlation: %8.6f %10.8f' % correlation.real, correlation.imag)
        return correlation


    def cal_ipr_value(self, coefficients=None, c2=None):
        if c2 is None:
            if coefficients is None: coefficients = self.coefficients
            c2 = np.einsum('i,i->i', coefficients.conj(), coefficients)

        # inverse participation ratio (ipr)
        ipr = 1./ np.einsum('i,i->', c2, c2)
        #print('ipr: %8.6f %10.8f' % ipr.real, ipr.imag)
        return ipr


    def analyze_wf_property(self, coefficients=None, c2=None):
        if c2 is None:
            if coefficients is None: coefficients = self.coefficients
            c2 = np.einsum('i,i->i', coefficients.conj(), coefficients)

        correlation = self.cal_r_correlation(c2=c2)
        ipr = self.cal_ipr_value(c2=c2)

        return correlation, ipr



class ExcitonDynamicsStep3D(ExcitonDynamicsStep):
    # x, y, z axis.
    # (1,1,1) is center O, (0,1,1) and (2,1,1) is the left and right points on x-axis
    def process_parameters(self):
        abc, angle, element, scale = self.unit_cell['abc'], self.unit_cell['angle'], self.unit_cell['element'], self.unit_cell['scale']
        elements_all, coordinates, centers_all, site_label = add_molecules_cell(self.unit_cell['n_site'], abc, angle, element, scale)

        n_tot = np.prod(self.unit_cell['n_site'])*self.n_mol
        i = int(n_tot//2) # center site
        print('center site i:', i, site_label[i])
        distances = []
        for j in range(n_tot):
            distances.append(np.linalg.norm(centers_all[i]-centers_all[j]))
        distances = np.array(distances)

        npairs = 6
        order = distances.argsort()[:npairs+1]

        print('site_label:')
        for k in range(npairs+1):
            print('%3d: %10s %12.5f' % (order[k]+1, site_label[order[k]], distances[order[k]]))

        mol = 'H2OBPc'
        energy, coupling = [], []
        for k in order[1:npairs+1]:
            outfile = '../pbe0/'+mol+'-'+str(i+1)+'-'+str(k+1)+'-dimer'+'_%4.2f-dc.out' % distances[k]
            e, c = read_energy_coupling(outfile)
            energy.append(e)
            coupling.append(c)

        self.energy = np.array(energy[0])
        self.coupling_j = np.array(coupling)
        self.coupling_g = np.zeros_like(self.coupling_g)
        self.coupling_a = np.zeros_like(self.coupling_a)

        # given neighboring pairs
        index = [list(map(int, site_label[order[k]].split(','))) for k in range(len(order))]
        #print('index:\n', index)

        neighbor_index = [None]*self.n_mol
        i = index[0][3]
        neighbor_index[i] = np.array(index[1:]) # remove the center index

        for j in range(1, len(index)):
            a, b, c, d = index[j]
            index[j] = [-a, -b, -c, abs(d-1)]
        #neighbor_index[abs(i-1)] = np.array(index[1:])
        neighbor_index[abs(i-1)] = np.zeros_like(index[1:]) #TODO: is this right???

        self.neighbor_index = np.array(neighbor_index)
        self.ntype = self.neighbor_index.shape[1]


    def get_exciton_couplings(self, coordinate):
        # nt is number of molecules in a unit cell
        # ns is number of states per molecule
        (nx, ny, nz), nt, ns = self.n_site, self.n_mol, self.nstate
        hamiltonian = np.zeros((nx, ny, nz, nt, ns, nx, ny, nz, nt, ns))

        coordinate1 = np.copy(coordinate)
        coordinate1[:-1] -= coordinate[1:]
        #coupling = np.einsum('kmij,nkm->nkij', self.coupling_a, coordinate1.reshape(-1, self.ntype, self.n_mode))
        #coupling = coupling.reshape(-1, self.nstate, self.nstate)

        for icount, (i, j, k, l) in enumerate(itertools.product(range(1, nx-1), range(1, ny-1), range(1, nz-1), range(nt))):
            coupling = np.einsum('tmij,m->tij', self.coupling_a, coordinate1[icount])
            print('coupling:', coupling)
            for x, (a, b, c, d) in enumerate(self.neighbor_index[l]):
                hamiltonian[i,j,k,l,:,i+a,j+b,k+c,d] = self.coupling_j[x] + coupling
                hamiltonian[i+a,j+b,k+c,d,i,j,k,l] = hamiltonian[i,j,k,0,:,i+a,j+b,k+c,d].transpose()

        return np.reshape(hamiltonian, (self.n_site_tot*self.nstate, -1))


    def cal_force(self, coordinate, coefficients=None):
        if coefficients is None: coefficients = self.coefficients

        nx, ny, nz, nt = self.n_site, self.n_mol
        coefficients = np.reshape(coefficients, (nx, ny, nz, nt, -1)) # last dimension is nstate
        force = np.zeros((nx, ny, nz, nt, self.n_mode))

        for (i, j, k, l) in itertools.product(range(1, nx-1), range(1, ny-1), range(1, nz-1), range(nt)):
            c2 = np.zeros(self.ntype)
            for x, (a, b, c, d) in enumerate(self.neighbor_index[l]):
                c2[x] = np.einsum('i,j->ij', coefficients[i,j,k,l].conj(), coefficients[i+a,j+b,k+c,d])
            force[i,j,k,l] = -2.* np.einsum('tmij,tij->m', self.coupling_a, c2.real)

        return np.ravel(force)



class Dynamics():
    def __init__(self, key, **kwargs):
        self.total_time = 1

        put_kwargs_to_keys(key, **kwargs)
        # only take the total_time here
        if 'total_time' in key.keys():
            self.total_time = key.pop('total_time')
        print('dynamics run %d steps in %.3f fs.' %(self.total_time, float(self.total_time*AU2FS)))

        if 'dimer_label' in key.keys():
            self.edstep = ExcitonDynamicsStep3D(key)
        else: # normal 1d
            self.edstep = ExcitonDynamicsStep(key)

        key['n_site'] = self.edstep.n_site_tot
        self.ndstep = OscillatorDynamicsStep(key)

        # variables needed
        self.total_energy = np.zeros(self.total_time)
        self.correlation = np.zeros(self.total_time)
        self.ipr = np.zeros(self.total_time)
        self.c2 = []


    def kernel(self):
        # assign local variables for class attributes
        ndstep, edstep = self.ndstep, self.edstep

        coords = ndstep.coordinate # equal sign used here as a pointer
        c2 = edstep.get_initial_coefficients(coords)
        correlation, ipr = edstep.analyze_wf_property(c2=c2)

        self.c2.append(c2)
        self.correlation[0] = correlation
        self.ipr[0] = ipr

        self.total_energy[0] = edstep.cal_energy() + ndstep.energy
        force = edstep.cal_force(coords)

        for ti in range(1, self.total_time):
            ndstep.update_coordinate_velocity(force)
            c2 = edstep.update_coefficients(coords)
            correlation, ipr = edstep.analyze_wf_property(c2=c2)

            self.c2.append(c2)
            self.correlation[ti] = correlation
            self.ipr[ti] = ipr

            # velocity_verlet is cumbersome
            force = ndstep.update_coordinate_velocity(force, 2)

            self.total_energy[ti] = edstep.cal_energy() + ndstep.energy
            force = edstep.cal_force(coords)

        #print_matrix('ground-state energy (eV):', self.total_energy*HARTREE2EV, 10)
        #print_matrix('coefficient weights:', np.array(self.c2), 10)
        #print_matrix('correlation:', self.correlation, 10)
        #print_matrix('ipr:', self.ipr, 6)


    def plot_time_variables(self, fig_name=None):
        import matplotlib.pyplot as plt

        dpi = 300 if fig_name else 100
        fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(11,6), sharex=True, dpi=dpi)

        time_line = np.linspace(0, self.total_time, self.total_time) * AU2FS *1e-3

        ax[0].plot(time_line, self.total_energy)
        ax[0].set_ylabel('Energy (a.u.)')

        #n = len(self.c2)
        #d = int(n/8)
        #variable = self.c2[::d]
        #line = range(1, len(variable[0])+1)
        #for i in range(len(variable)):
        #    ax[1].plot(line, variable[i], label='%.1f fs' % float(d*i*AU2FS))

        #ax[1].set_ylabel('Coefficients')
        #ax[1].set_xlabel('Site (State) No.')
        #ax[1].legend()

        ax[1].plot(time_line, self.correlation)
        ax[1].set_ylabel('correlation ($\\AA^2$)')

        ax[2].plot(time_line, self.ipr)
        ax[2].set_ylabel('IPR')
        ax[2].set_xlabel('Time (ps)')

        plt.tight_layout()
        if fig_name:
            plt.savefig(fig_name)
        else:
            plt.show()



if __name__ == '__main__':
    total_time = 60000
    key = {}
    key['debug'] = 2
    key['random_seed'] = 1385448536
    key['dt'] = 10
    key['exciton_dt'] = 10
    #key['update_method'] = 'velocity_verlet'

    """
    Fornari, et. al. JPCC 2016 10.1021/acs.jpcc.6b01298 parameters testing
    for H2-OBPc molecular crystal
    """
    n_mode = 6
    key['n_mode'] = n_mode
    key['mass']  = [6., 6., 754., 754., 754., 754.] # amu
    key['frequency'] = [144., 148., 5., 5., 5., 5.] # meV

    n_site = np.array([21, 1, 1])
    distance = 8.64 #[8.64, 8.64, 8.64] # Angstrom
    nstate = 2
    n_mol = 2

    key['n_site'] = n_site
    key['distance'] = distance
    key['nstate'] = nstate
    key['n_mol'] = n_mol

    key['energy'] = [0., 10.] # meV

    coupling_g = np.zeros((n_mode, nstate))
    coupling_g[0,0] = 1821. # meV/AA
    coupling_g[1,1] = 2231. # meV/AA
    key['coupling_g'] = coupling_g

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

    key['dimer_label'] = {}
    key['cif'] = 'H2OBPc.cif'
    key['n_site'] = np.array([6, 6, 1])

    obj = Dynamics(key, total_time=total_time)
    obj.kernel()
    obj.plot_time_variables('test.png')
