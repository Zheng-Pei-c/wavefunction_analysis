from wavefunction_analysis import np
from wavefunction_analysis.utils import print_matrix

from wavefunction_analysis.plot import plt

fig = plt.figure()

ntot = 3

for k, nfrag in enumerate([3]):
    es = np.sort(np.random.random(ntot))
    es = np.array([np.ones(nfrag)*e for e in es]).T
    for s in range(ntot):
        es[:,s] += s + np.random.random(nfrag)
    vs = np.random.random((nfrag, ntot)) * .1
    w = np.average(es[:,0])

    H1 = np.zeros((nfrag*ntot+1, nfrag*ntot+1))
    H1[-1,:-1] += vs.ravel()
    H1 += H1.T
    np.fill_diagonal(H1, np.append(es.ravel(), w))
    print_matrix('H:', H1)

    e, v = np.linalg.eigh(H1)
    print_matrix('e:', e)
    print_matrix('v:', v)

    plt.plot(e, np.ones(len(e))*k, '*', color='C1')


    e, v = [], []
    for i in range(nfrag):
        _H = np.zeros((ntot+1, ntot+1))
        _H[-1,:-1] += vs[i]
        _H += _H.T
        np.fill_diagonal(_H, np.append(es[i], w/nfrag))

        _e, _v = np.linalg.eigh(_H)
        #arg = np.argsort(-np.abs(_v[-1])) # sort photon character
        arg = np.argsort(np.abs(_e))
        _e, _v = _e[arg], _v[:,arg]
        e.append(_e)
        v.append(_v)
    e, v = np.array(e), np.array(v)

    for nstate in [2, ntot+1]:
        print('nstate:', nstate)
        v2 = np.zeros((nfrag*ntot+1, nfrag, nstate))
        for i in range(nfrag):
            v2[i*ntot:(i+1)*ntot,i] = v[i,:ntot,:nstate]
            v2[-1,i] = v[i,-1,:nstate]
        e2 = e[:,:nstate].ravel()
        v2 = v2.reshape(nfrag*ntot+1, -1)
        print_matrix('e2:', e2)
        print_matrix('v2:', v2)

        H3 = np.einsum('k,qk->qk', np.sqrt(e2), v2)
        print_matrix('H3:', H3)
        u, s, vt = np.linalg.svd(H3)
        print_matrix('s:', (s**2.)[::-1])
        print_matrix('u:', u[:,::-1])
        H3 = np.einsum('pk,qk->pq', H3, H3)
        print_matrix('H3:', H3)

    #H3 = np.einsum('pk,k,qk->pq', v, e, v)
    #print_matrix('H3:', H3)
    #e, v = np.linalg.eigh(H3)
    #print_matrix('e:', e)




    plt.plot(e, np.ones(len(e))*k-.1, '.', color='C2')

#plt.show()


beta = 1./0.0257 #eV
n = 10
H = np.random.random((n, n)) /100.
H += H.T
e, v = np.linalg.eigh(H)
exp = np.exp(-beta*e)
print_matrix('exp:', exp)
from scipy.linalg import expm
exp = expm(-beta*H)
exp = np.einsum('ik,ij,jk->k', v, exp, v)
print_matrix('exp:', exp)

c = np.einsum('k,ik->i', exp, v)
print_matrix('c:', c)
c = np.einsum('ij,jk->i', expm(-beta*H), v)
print_matrix('c:', c)
