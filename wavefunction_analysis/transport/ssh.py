from wavefunction_analysis import np, sys
from wavefunction_analysis import itertools
from wavefunction_analysis.utils import read_number, read_array, print_matrix, convert_units
from wavefunction_analysis.dynamics.dimers_in_crystal import read_unit_cell_info, add_molecules_cell

import matplotlib.pyplot as plt
import matplotlib.cm as cm

np.set_printoptions(precision=4, linewidth=200, suppress=True,)

def read_energy_coupling(outfile):
    energy = read_number(outfile, 'excitation energy', n=1, o=7, dtype=float)
    trans_dipole = []
    for x in range(3):
        transx = read_number(outfile, 'Trans. Mom.:', n=1, o=2*(x+1), dtype=float)
        trans_dipole.append(transx)
    trans_dipole = np.reshape(trans_dipole, (3,2,-1))[:,:,:2].transpose(1,2,0)

    coupling = read_array(outfile, 'Coulomb (eV)    Exchange (eV)  Interaction (eV)', 8, 6, nrange=[3,4], same=False)
    coupling = np.array([coupling[0], coupling[1], coupling[6], coupling[7]]).reshape(2,2)

    for i, di in enumerate(trans_dipole[0]):
        signi = np.sign(di[np.abs(di).argmax()])
        for j, dj in enumerate(trans_dipole[1]):
            signj = np.sign(dj[np.abs(dj).argmax()])
            coupling[i,j] *= signi*signj

    #print(coupling)

    dipoles = np.array([di*np.sign(di[np.abs(di).argmax()]) for di in trans_dipole[0]])

    return energy[:2], coupling, dipoles


def hamil_AB(potential, hopping, N):
    # AB model has two alternating hoppings in a 1D chain (ABAB...)
    nstates = hopping.shape[1] if hopping.ndim>1 else 1
    H = np.zeros((N, nstates, N, nstates), dtype=hopping.dtype)

    for j in range(2):
        for i in range(j, N-1, 2):
            H[i, :, i+1] = hopping[j]

    H = H.reshape((N*nstates, N*nstates))
    H += H.conj().T

    np.fill_diagonal(H, np.tile(potential, (N, 1)).ravel())

    e, v = np.linalg.eigh(H)

    return H, e, v


def _get_dk(N):
    return [(np.pi /n) for n in N]


def discrete_theta_k(N, ndim=None, extra=1):
    # be consistent to _get_dk
    if type(N) is int: N = [N]
    if ndim is None: ndim = len(N)
    else: N = N*ndim

    thetas = []
    for i in range(ndim):
        thetas.append(np.arange(-N[i]-extra, N[i]+1+extra) * (np.pi/N[i]))
        print('theta:', thetas[-1].shape, '\n', thetas[-1])
    thetas = itertools.product(*thetas)

    return thetas


def hamil_ssh(potential, hopping, indices, N, extra=1):
    # potential is the on-site energy
    # hopping is the coupling between two sites
    # N gives the numbers of k-points
    # extra points are added for finite-difference calculations of berry phase
    if type(potential) is float: potential = [[potential], [potential]]
    potential = np.array(potential)
    nsite = potential.shape[0]
    nstate = potential.shape[1] if potential.ndim>1 else 1
    print('nsite:', nsite, 'nstate:', nstate)
    if nsite != 2:
        raise ValueError('other than 2 sites per unit cell is unimplemented')

    thetas = discrete_theta_k(N, extra=extra)
    thetas = list(thetas) # save thetas

    Hs, es, vs = [], [], []
    for theta in thetas:
        H = hamil_ssh_k(potential, hopping, indices, theta, nsite, nstate)
        #print('H:', H)
        e, v = np.linalg.eigh(H)

        #Hs.append(H)
        es.append(e)
        vs.append(v)

    #print('es:', np.array(es).shape)
    return np.array(es), np.array(vs), thetas


def hamil_ssh_0(potential, hopping, nsite, nstate):
    # k-independent part
    H = np.zeros((nsite, nstate, nsite, nstate), dtype=np.complex128)
    H[0,:,1] = hopping
    H = H.reshape((nsite*nstate, -1))
    H += H.conj().T

    np.fill_diagonal(H, potential.flatten())

    return H


def hamil_ssh_k(potential, hopping, indices, theta, nsite, nstate):
    # k-dependent part
    # SSH model has nsite per unit cell, nstate per site
    # and alternating neighboring hoppings in a 1D/2D/3D lattice (ABAB...)
    if len(indices) != 3:
        raise ValueError('three types parameters are required!')

    H = np.zeros((nsite, nstate, nsite, nstate), dtype=np.complex128)

    hs = [None]*3
    for t, _indices in enumerate(indices):
        _coupling = hopping[t]

        h = np.zeros((nstate, nstate), dtype=H.dtype)
        for l, index in enumerate(_indices):
            #print('l:', l, 'index:', index, 'coupling', _coupling[l])
            phase = np.dot(index, theta)
            h += _coupling[l] * np.exp(-1j*phase)
        hs[t] = h

    H[0,:,0] = hs[0]
    H[1,:,1] = hs[1]
    H[1,:,0] = hs[2]

    H = H.reshape((nsite*nstate, -1))
    H += H.conj().T

    # add on-site energy
    np.fill_diagonal(H, H.diagonal() + potential.flatten())

    return H


def reshape_eigenvectors(vectors, N, nband, extra=1):
    nd = len(N)
    nband = vectors.shape[-1] if vectors.ndim==3 else 1

    N = [N[i]*2 for i in range(nd)]
    n = 1+2*extra
    if nd == 1:
        vectors = vectors.reshape((N[0]+n, -1, nband))
    elif nd == 2:
        vectors = vectors.reshape((N[0]+n, N[1]+n, -1, nband))
    elif nd == 3:
        vectors = vectors.reshape((N[0]+n, N[1]+n, N[2]+n, -1, nband))
    return vectors


def berry_phase_fd(vectors, N, extra=1):
    if type(N) is int: N = [N]
    nd = len(N)

    dk = _get_dk(N)
    nband = vectors.shape[-1]

    connection, phase = [], []
    for i in range(nd):
        v = np.swapaxes(np.copy(vectors), 0, i)
        grad = np.ma.diff(v, axis=0) / dk[i] # g[i] = (a[i+1] - a[i])
        _A = np.einsum('k...pq,k...pq->k...q', v[1:-1].conj(), grad[1:]).imag
        _p = np.sum(_A.reshape((-1, nband)), axis=0)

        connection.append(np.swapaxes(_A, 0, i))
        phase.append(_p)

    curvature = None
    if nd == 1:
        connection = connection[0]
        phase = phase[0]

    elif nd == 2:
        curvature  = np.ma.diff(connection[0], axis=1)[:,:-1] / dk[1]
        curvature -= np.ma.diff(connection[1], axis=0)[:-1] / dk[0]

    #elif nd == 3:
    #    curvature = [None]*3
    #    curvature[0]  = np.ma.diff(connection[1], axis=2)[:,:,:-1] / dk[2]
    #    curvature[0] -= np.ma.diff(connection[2], axis=1)[:,:-1] / dk[1]
    #    curvature[1]  = np.ma.diff(connection[2], axis=0)[:-1] / dk[0]
    #    curvature[1] -= np.ma.diff(connection[0], axis=2)[:,:,:-1] / dk[2]
    #    curvature[2]  = np.ma.diff(connection[0], axis=1)[:,:-1] / dk[1]
    #    curvature[2] -= np.ma.diff(connection[1], axis=0)[:-1] / dk[0]

    #    curvature = np.array(curvature)

    return np.array(connection), np.array(phase), curvature


def berry_phase_fd_2():
    """
    doi: 10.1143/JPSJ.74.1674
    """
    return


def berry_phase_ln(vectors, N):
    if type(N) is int: N = [N]
    nd = len(N)

    dk = _get_dk(N)
    nband = vectors.shape[-1]

    connection, phase = [], []
    for i in range(nd):
        v = np.swapaxes(np.copy(vectors), 0, i)
        _A = np.log(np.einsum('k...pq,k...pq->k...q', v[:-1].conj(), v[1:])).imag
        _p = np.sum(_A.reshape((-1, nband)), axis=0)

        connection.append(np.swapaxes(_A, 0, i))
        phase.append(_p)

    curvature = None
    if nd == 1:
        connection = connection[0]
        phase = phase[0]

    return np.array(connection), np.array(phase)


def energy_ssh_1d(potential, hopping, thetas):
    # exact solution of A-B=A-B SSH hamiltonian
    v, w = hopping
    e = np.sqrt(v**2 + w**2 + 2*v*w*np.cos(thetas))
    return e


def berry_phase_ssh_1d(hopping, theta):
    # exact analytic solution of 2-by-2 SSH hamiltonian
    v, w = hopping
    cos, sin = np.cos(theta), np.sin(theta)
    phi = w * sin / (v + w * cos)

    # arctan derivative is 1/(1+x^2)
    dpdk = 1./(1. + phi**2) * (-w * cos / (v + w * cos) - (w * sin)**2 / ((v + w * cos)**2))

    return phase


def block_diagonal(H):
    E, T = H[:2,:2], H[:2,2:]
    e = E[1,1]
    t1, t2, t3 = T[0,0], T[0,1], T[1,1]

    r11 = -1./(2.*e) * (t1.conj() + t2.conj() / t2 * t3)
    r22 = -1./(2.*e) * (t3.conj() + t2.conj() / t2 * t1)

    a = -t2
    b = - (2.*e + r11 * t1 + r22 * t3)
    c = t2.conj() - r11 * r22 * t2
    delta = np.sqrt(b**2 - 4*a*c)

    r12 = (-b + delta) / (2.*a)
    r21 = (-b - delta) / (2.*a)

    R1 = np.array([[r11, r12], [r21, r22]])
    R2 = R1.conj().T #np.linalg.inv(R1)
    print('R1:\n', R1)
    print('R2:\n', R2)


    A = E + T @ R1
    B = E + T.conj().T @ R2

    Hn = np.block([[A, np.zeros((2,2))], [np.zeros((2,2)), B]])
    W = np.block([[np.eye(2), R2], [R1, np.eye(2)]])

    H2 = W @ Hn @ np.linalg.inv(W)
    print('H:\n', H)
    print('H2:\n', H2)
    print(np.linalg.eigvals(A), np.linalg.eigvals(B))
    return A, B


def check_error():
    fig, axs = plt.subplots(1, 2, figsize=(12, 9), layout='constrained')
    lw = 1

    E = np.array([.0, 0.])
    #T = np.array([[[1.5, .1], [.1, 1.]], [[1., .5], [.5, 2.]]])
    T = np.array([.3, .8])
    Ns = [50, 100, 200, 400, 800, 1600]

    error = []
    for i, N in enumerate(Ns):
        H, e, v = hamil_AB(E, T, N)
        #print(H)
        #print(e)

        es, vs = hamil_ssh(E, T, N//2)
        #print(es)
        e = e[np.where(e>0)]
        es = es[np.where(es>0)].ravel()
        es = es.ravel()
        print(e.shape, es.shape)
        if len(e) > len(es): e = e[1:]
        elif len(e) < len(es): e = np.insert(e, 0, 0)
        error.append(np.linalg.norm(e-es)/N)

        axs[0].hlines(e, i, .9+i, lw=lw/(i+1), color='C0')
        if i < 5:
            axs[0].hlines(es, i, .9+i, lw=lw/(i+1), color='C1')
        axs[1].hlines(es, i, .9+i, lw=lw/(i+1), color='C1')
    #axs[2].semilogx(Ns, error)
    #axs[2].set_xscale("log", base=10)

    ylim = [np.min(es)-.01, np.max(es)+.01]
    for i in range(2):
        axs[i].set_ylim(ylim)

    plt.show()



def sort_index(i, j, k, l, nx, ny, nz, ns):
    return i*ny*nz*ns + j*nz*ns + k*ns + l



if __name__ == "__main__":
    E = [.0, .0]
    hopping = [[0., 0., 0.], # taa
               [0., 0., 0.], # tbb
               [1., 0.5], # tba
               ]
    indices = [[1,2,3],
               [1,2,3],
               [0,1]]
    N = [20]
    extra = 1
    es, vs, thetas = hamil_ssh(E, hopping, indices, N, extra)
    #print_matrix('es:', es)
    #print('vs:\n', vs)
    nband = es.shape[-1]
    vs = reshape_eigenvectors(vs, N, nband, extra)

    fig, ax = plt.subplots()

    connection, phase, curvature = berry_phase_fd(vs, N, extra)
    ax.plot(thetas[1:-1], connection[:,0], label='fd')
    print_matrix('phase_fd:\n', phase)
    connection, phase = berry_phase_ln(vs, N)
    ax.plot(thetas[1:-1], connection[1:,0], label='ln')
    print_matrix('phase_ln:\n', phase)

    ax.set_xticks(np.linspace(-np.pi, np.pi, 5))
    ax.set_xticklabels(['$-\\pi$', '$-\\pi/2$', '$0$', '$\\pi/2$', '$\\pi$'])
    ax.set_xlabel('k')
    ax.set_ylabel('Berry Connection')
    ax.legend()
#    plt.show()


    #sys.exit()

    # real parameters
    mol = 'H2OBPc'
    n_images = [5, 5, 5]
    n_total = 2 * np.prod(n_images)
    npairs = 50

    abc, angles, elements, scales = read_unit_cell_info(mol+'.cif')
    centers_all, site_label = add_molecules_cell(n_images, abc, angles, elements, scales)[-2:]

    distances = []
    i = int(n_total//2) # center site
    #i -= 1
    print('i:', i, site_label[i])
    for j in range(n_total):
        distances.append(np.linalg.norm(centers_all[i]-centers_all[j]))
    distances = np.array(distances)
    order = distances.argsort()
    order = order[:npairs+1]

    # given neighboring pairs
    index = [list(map(int, site_label[order[k]].split(','))) for k in range(len(order))]
    sort = [sort_index(d[0], d[1], d[2], d[3], n_images[0], n_images[1], n_images[2], 2) for d in index]
    print('index:')
    for k, idx in enumerate(index):
        print('%3d ' % (order[k]+1), end='')
        for d in idx:
            print('%2d ' %d, end='')
        print(' %10.5f' % distances[order[k]])

    index2 = [[-d[0], -d[1], -d[2], 1-d[3]] for d in index]
    sort2 = [sort_index(d[0], d[1], d[2], d[3], n_images[0], n_images[1], n_images[2], 2) for d in index2]
    print_matrix('sort', sort, 10, digits=[5,0,'f'])

    print('index:')
    for k, idx in enumerate(index):
        print('%3d ' % (order[k]+1), end='')
        for d in idx:
            print('%2d ' %d, end='')
        print('%4d  ' % sort[k], end=' ')

        idx2 = index2[k]
        for d in idx2:
            print('%2d ' %d, end='')
        print('%4d ' % sort2[k], end=' ')
        print(' %10.5f' % distances[order[k]], end=' ')

        if sort[k] < sort[0]:
            print(' sort1  0 0 0 %d' % idx[3], end=' ')
            for j, d in enumerate(idx):
                if j == 3:
                    print('%2d ' % index[0][3], end='')
                else:
                    print('%2d ' % (-d), end='')
            print('%3d %3d' % (sort_index(0,0,0,idx[3],5,5,5,2), sort_index(-idx[0],-idx[1],-idx[2],index[0][3],5,5,5,2)), end=' ')

        if sort2[k] < sort2[0]:
            print(' sort2  0 0 0 %d' % idx2[3], end=' ')
            for j, d in enumerate(idx2):
                if j == 3:
                    print('%2d ' % index2[0][3], end='')
                else:
                    print('%2d ' % (-d), end='')
            print('%3d %3d' % (sort_index(0,0,0,idx2[3],5,5,5,2), sort_index(-idx2[0],-idx2[1],-idx2[2],index2[0][3],5,5,5,2)), end=' ')
        print('')


    energy, coupling, trans_dipole = [], [], []
    for k in order[1:npairs+1]:
        outfile = mol+'-'+str(i+1)+'-'+str(k+1)+'-dimer'+'_%4.2f-dc.out' % distances[k]

        e, c, d = read_energy_coupling(outfile)
        energy.append(e)
        coupling.append(c)
        trans_dipole = d

    energy, coupling = np.array(energy), np.array(coupling)
    energy = energy[:2]
    print('energy', energy)

    index = np.array(index, dtype=int)

    coupling_parameter = [[] for _ in range(3)]
    coupling_index_label = [[] for _ in range(3)]
    center_idx = index[0] # B molecule
    previous_idx = np.array([-999, -999, -999], dtype=int)
    for k, idx in enumerate(index[1:]):
        if idx[3] == center_idx[3]: # same molecules in different cells B-B
            if not np.all(-idx[:3] == previous_idx[:3]): # remove the opposite direction parameters
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
            print('BA idx:', idx, 'dist:', distances[order[k+1]])
            print_matrix('coupling:', coupling[k])
            coupling_index_label[2].append(idx[:3])
            coupling_parameter[2].append(coupling[k])
            # inverse direction with transposed coupling A-B
            # added during the hamiltonian transpose

    N = [30, 10, 2]
    es, vs, thetas = hamil_ssh(energy, coupling_parameter, coupling_index_label, N, extra)

#    nband = es.shape[-1]
#    vs = reshape_eigenvectors(vs, N, nband, extra)
#
#    connection, phase, curvature = berry_phase_fd(vs, N, extra)
#    ax.plot(thetas[1:-1], connection[:,0], label='fd')
#    print_matrix('phase_fd:\n', phase)
#
#    ax.set_xticks(np.linspace(-np.pi, np.pi, 5))
#    ax.set_xticklabels(['$-\\pi$', '$-\\pi/2$', '$0$', '$\\pi/2$', '$\\pi$'])
#    ax.set_xlabel('k')
#    ax.set_ylabel('Berry Connection')
#    ax.legend()
#    plt.show()

#    sys.exit()


    n_site = [30, 10, 2]
    n_mol = 2
    nstate = 2
    (nx, ny, nz), nt, ns = n_site, n_mol, nstate
    n_site_tot = np.prod(n_site)
    hamiltonian = np.zeros((nx, ny, nz, nt, ns, nx, ny, nz, nt, ns))


    # get neighbor index list
    neighbor_index = []
    for j, idx in enumerate(index[1:], start=1):
        k = index[0][3] # B molecule as center
        if sort[j] < sort[0]: # A molecule as center
            k = abs(1-k)
            a, b, c, d = idx
            idx = [-a, -b, -c, abs(1-d)] # checked by index from using center (i -= 1)
        neighbor_index.append([k, idx])


    for icount, (i, j, k) in enumerate(itertools.product(range(nx), range(ny), range(nz))):
        for x, (l, (a, b, c, d)) in enumerate(neighbor_index):
            a, b, c = i+a, j+b, k+c
            if 0<=a<nx and 0<=b<ny and 0<=c<nz:
                hamiltonian[i,j,k,l,:,a,b,c,d] = coupling[x]
            #else:
            #    print(a, b, c)

    hamiltonian = np.reshape(hamiltonian, (n_site_tot*n_mol*nstate, -1))
    hamiltonian += hamiltonian.T

    np.fill_diagonal(hamiltonian, np.tile(energy, (n_site_tot*n_mol, 1)).ravel())

    #print_matrix('hamiltonian:', hamiltonian, 10)

    e, v = np.linalg.eigh(hamiltonian)
    v = v.reshape(nx, ny, nz, nt, ns, -1)
    trans_dipole = np.einsum('abctsk,sx->kx', v, trans_dipole)
    f = (2./3.) * np.einsum('k,kx,kx->k', e, trans_dipole, trans_dipole)

    from wavefunction_analysis.plot import broadening
    x, y = broadening(convert_units(e, 'ev', 'nm'), f, xrange=[600, 1100], wid=.01, method='lorentzian')
    fig = plt.figure()
    plt.plot(x, y)
    plt.show()
