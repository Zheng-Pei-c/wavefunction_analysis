from wavefunction_analysis import np
from wavefunction_analysis.utils import read_number, read_array, print_matrix
from wavefunction_analysis.dynamics.dimers_in_crystal import read_unit_cell_info, add_molecules_cell

def read_energy_coupling(outfile, nstate=2):
    r"""
    Read exciton energy and Coulomb coupling from the output file.

    Parameters
        outfile : str
            The q-chem output file name
        nstate : int
            The number of exciton states per site

    Returns
        energy : (nstate,) array
            Exciton energy of each site in eV
        coupling : (nstate, nstate) array
            Exciton-exciton coupling between two sites in
        trans_dipole : (2, nstate, 3) array
            Transition dipole moments of two sites in au
    """
    energy = read_number(outfile, 'excitation energy', n=1, o=7, dtype=float)
    energy = np.array(energy).reshape(2,-1)[:, :nstate]

    trans_dipole = []
    for x in range(3):
        transx = read_number(outfile, 'Trans. Mom.:', n=1, o=2*(x+1), dtype=float)
        trans_dipole.append(transx)
    trans_dipole = np.array(trans_dipole).reshape(3,2,-1).transpose(1,2,0)[:,:nstate,:]

    # state numbers to be coupled
    sts_trans = read_number(outfile, 'sts_trans', n=1, o=1, dtype=str)
    sts_trans = [int(s.split('-')[-1]) for s in sts_trans]
    coupling = read_array(outfile, 'Coulomb (eV)    Exchange (eV)  Interaction (eV)',
                          nline=np.prod(sts_trans), ncol=6, nrange=[3,4], same=False)
    coupling = np.array(coupling).reshape(sts_trans)[:nstate,:nstate]

    for i, di in enumerate(trans_dipole[0]):
        signi = np.sign(di[np.argmax(np.abs(di))])
        trans_dipole[0][i] *= signi
        for j, dj in enumerate(trans_dipole[1]):
            signj = np.sign(dj[np.argmax(np.abs(dj))])
            trans_dipole[1][j] *= signj
            coupling[i,j] *= signi*signj
    #print_matrix('coupling:', coupling)

    return energy, coupling, trans_dipole


def sort_index(i, j, k, l, nx, ny, nz, ns):
    return i*ny*nz*ns + j*nz*ns + k*ns + l


def process_parameters(cif_file, n_cell, outfile_dir, nstate=2, npairs=50,
                       center=0, debug=0):
    r"""
    Process the input parameters and get real parameters:
    on-site energies, transition dipoles, state couplings for dimers.

    Parameters
        cif_file : str
            The cif file name
        n_cell : list of int
            The number of unit cells in each direction [nx, ny, nz]
        outfile_dir : str
            The output file directory
        nstate : int
            The number of exciton states per site

    Returns
        unit_cell : dict
            The unit cell information
        energy : (nstate,) array
            Exciton energy of each site in eV
        coupling_j : (ndimer, nstate, nstate) array
            Exciton-exciton couplings between dimers in eV
        dipole : (2, nstate, 3) array
            Transition dipole moments of two sites in au
        neighbor_index : list of [int, list of int]
            The neighbor index list for each dimer
    """
    mol = cif_file.replace('.cif', '')

    # find out the unit cell information
    abc, angles, elements, scales, n_mol = read_unit_cell_info(mol+'.cif')
    unit_cell = {'abc': abc, 'angles': angles, 'elements': elements,
                 'scales': scales, 'n_mol': n_mol}

    elements_all, coordinates, centers_all, site_label = add_molecules_cell(n_cell, abc, angles, elements, scales)

    distances = []
    n_total = n_mol * np.prod(n_cell)
    i = int(n_total//2) # center site (B)
    i -= center # shift if needed (A if center=1)
    if debug:
        print('i:', i, site_label[i])
    for j in range(n_total):
        distances.append(np.linalg.norm(centers_all[i]-centers_all[j]))
    distances = np.array(distances)
    order = distances.argsort()
    order = order[:npairs+1]

    # given neighboring pairs
    index = [list(map(int, site_label[order[k]].split(','))) for k in range(len(order))]
    sort = [sort_index(d[0], d[1], d[2], d[3], n_cell[0], n_cell[1], n_cell[2], 2) for d in index]
    if debug:
        print('index:')
        for k, idx in enumerate(index):
            print('%3d ' % (order[k]+1), end='')
            for d in idx:
                print('%2d ' %d, end='')
            print(' %10.5f' % distances[order[k]])

    # read energy, dipoles, and couplings from output files
    energy, coupling, trans_dipole = [], [], []
    for k in order[1:npairs+1]:
        outfile = mol+'-'+str(i+1)+'-'+str(k+1)+'-dimer'+'_%4.2f-dc.out' % distances[k]

        e, c, d = read_energy_coupling(outfile)
        #energy.append(e)
        coupling.append(c)
        energy = e[0]
        trans_dipole = d[0]

    energy, coupling = np.array(energy), np.array(coupling)

    index = np.array(index, dtype=int)

    coupling_parameter = [[] for _ in range(3)]
    coupling_index_label = [[] for _ in range(3)]
    center_idx = index[0] # B molecule
    previous_idx = np.array([-999, -999, -999], dtype=int)
    for k, idx in enumerate(index[1:]):
        if idx[3] == center_idx[3]: # same molecules in different cells B-B
            if not np.all(-idx[:3] == previous_idx[:3]): # remove the opposite direction parameters
                if debug:
                    print('BB idx:', idx, 'dist:', distances[order[k+1]])
                    print_matrix('coupling:', coupling[k])
                # same direction with transposed coupling due to the mirror symmetry between the two molecules
                # so that same coupling for opposite direction
                coupling_index_label[0].append(idx[:3]) # A-A
                coupling_parameter[0].append(coupling[k].T) # A-A
                coupling_index_label[1].append(idx[:3]) # B-B
                coupling_parameter[1].append(coupling[k]) # B-B
                previous_idx = idx
        else: # different molecules in different cells B-A
            if debug:
                print('BA idx:', idx, 'dist:', distances[order[k+1]])
                print_matrix('coupling:', coupling[k])
            coupling_index_label[2].append(idx[:3])
            coupling_parameter[2].append(coupling[k])
            # inverse direction with transposed coupling A-B
            # added during the hamiltonian transpose


    # get neighbor index list
    neighbor_index = []
    for j, idx in enumerate(index[1:], start=1):
        k = index[0][3] # B molecule as center
        if sort[j] < sort[0]: # A molecule as center
            k = abs(1-k)
            a, b, c, d = idx
            idx = [-a, -b, -c, abs(1-d)] # checked by index from using center (i -= 1)
        neighbor_index.append([k, idx])

    keys = {'unit_cell': unit_cell,
            'energy': energy,
            'coupling_j': coupling,
            'dipole': trans_dipole,
            'neighbor_index': neighbor_index,
            }
    return keys
