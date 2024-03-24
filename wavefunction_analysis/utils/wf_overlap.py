import numpy as np

from pyscf import scf, tdscf, gto

#from wavefunction_analysis.utils import print_matrix

def cal_wf_overlap(Xm, Ym, Xn, Yn, Cm, Cn, S):
    has_n = True if isinstance(Xn, np.ndarray) else False
    has_y = True if isinstance(Ym, np.ndarray) else False

    nroots, no, nv = Xm.shape
    smo = np.einsum('mp,mn,nq->pq', Cm, S, Cn)
    #print_matrix('smo:', smo, 5, 1)

    smo_oo = np.copy(smo[:no,:no])
    dot_0 = np.linalg.det(smo_oo)

    # Cramer's rule
    # Ax_j = b_j <=> x_j = det(A_j(b_j)) / det(A) where A_j(b_j) is replacing A's j-column with b_j
    vec1 = np.linalg.solve(smo_oo.T, smo[no:,:no].T) # replace rows
    vec2 = np.linalg.solve(smo_oo, smo[:no,no:]) # replace columns

    #vec3 = np.linalg.solve(smo_oo, Xm.transpose(1,0,2).reshape(no, -1))
    #vec3 = vec3.reshape(no, nroots, nv).transpose(1,0,2)
    #vec4 = smo[no:,no:] - np.einsum('ia,ib->ab', vec1, smo[:no,no:])
    #vec4 = np.einsum('ab,kib->kia', vec4, Xn)

    ovlp1 = np.einsum('kia,ia->k', Xm, vec1)
    ovlp_m0 = np.copy(ovlp1)

    if has_y:
        ovlp2 = np.einsum('kia,ia->k', Ym, vec2)
        ovlp_m0 += ovlp2

    ovlp_m0 *= 2.*dot_0**2

    if not has_n:
        return np.array([dot_0**2, ovlp_m0])

    else:
        ovlp3 = np.einsum('kia,ia->k', Xn, vec2)
        ovlp_0n = np.copy(ovlp3)
        if has_y:
            ovlp4 = np.einsum('kia,ia->k', Yn, vec1)
            ovlp_0n += ovlp4

        ovlp_0n *= 2.*dot_0**2

        ovlp_mn = np.einsum('m,n->mn', ovlp1, ovlp3)
        if has_y:
            ovlp_mn -= np.einsum('m,n->mn', ovlp2, ovlp4)

        for a in range(nv):
            for i in range(no):
                ts0 = np.copy(smo[:no,:])
                ts0[i,:] = smo[no+a,:]
                vec3 = np.linalg.solve(ts0[:,:no], ts0[:,no:])
                ovlp_mn += np.einsum('m,njb,jb->mn', Xm[:,i,a], Xn, vec3) * vec1[i,a]

                if has_y:
                    ovlp_mn -= np.einsum('mjb,n,jb->mn', Ym, Yn[:,i,a], vec3) *vec1[i,a]

        ovlp_mn *= 2.*dot_0**2

        return np.block([[dot_0**2., ovlp_0n.reshape(1,-1)], [ovlp_m0.reshape(-1,1), ovlp_mn]])


def cal_wf_overlap0(Xm, Ym, Xn, Yn, Cm, Cn, S):
    has_n = True if isinstance(Xn, np.ndarray) else False
    has_y = True if isinstance(Ym, np.ndarray) else False

    nroots, no, nv = Xm.shape
    smo = np.einsum('mp,mn,nq->pq', Cm, S, Cn)

    smo_oo = np.copy(smo[:no,:no])
    dot_0 = np.linalg.det(smo_oo)

    ovlp_m0, ovlp_0n, ovlp_mn = 0., 0., 0.
    ovlp1, ovlp2 = [], []
    ovlp3, ovlp4 = [], []

    # e-g
    for a in range(nv):
        for i in range(no):
            ts = np.copy(smo_oo)
            ts[i,:] = smo[no+a,:no]
            dot = np.linalg.det(ts)
            ovlp1.append(Xm[:,i,a] * dot)

            if has_n and has_y: # Y part of g-e
                ovlp4.append(Yn[:,i,a] * dot)

    ovlp_m0 = np.sum(ovlp1, axis=0)*dot_0 # e-g

    if has_y or has_n: # Y part of e-g
        for a in range(nv):
            for i in range(no):
                ts = np.copy(smo_oo)
                ts[:,i] = smo[:no,no+a]
                dot = np.linalg.det(ts)

                if has_y: ovlp2.append(Ym[:,i,a] * dot)
                if has_n: ovlp3.append(Xn[:,i,a] * dot)

        ovlp_m0 += np.sum(ovlp2, axis=0)*dot_0 # e-g from Y

    if not has_n:
        return 2.*ovlp_m0
    else:
        ovlp_0n = np.sum(ovlp3, axis=0)*dot_0 # g-e
        if has_y:
            ovlp_0n += np.sum(ovlp4, axis=0)*dot_0

        # e-e
        for a in range(nv):
            for i in range(no):
                #tmp0 = np.copy(Cm[:,:no])
                #tmp0[:,i] = Cm[:,no+a]

                for b in range(nv):
                    for j in range(no):
                        #tmp1 = np.copy(Cn[:,:no])
                        #tmp1[:,j] = Cn[:,no+b]
                        #ts0 = np.einsum('pi,pq,qj->ij', tmp0, S, tmp1)
                        ts0 = np.copy(smo_oo)
                        ts0[i,:] = smo[no+a,:no]
                        ts0[:,j] = smo[:no,no+b]
                        ts0[i,j] = smo[no+a,no+b]
                        dot = np.linalg.det(ts0) * dot_0

                        ovlp_mn += np.einsum('m,n->mn', Xm[:,i,a], Xn[:,j,b]) * dot

                        if has_y:
                            ovlp_mn -= np.einsum('m,n->mn', Ym[:,j,b], Yn[:,i,a]) * dot

        # e-g * g-e
        ovlp_mn += np.einsum('im,jn->mn', ovlp1, ovlp3)
        if has_y:
            ovlp_mn -= np.einsum('jm,in->mn', ovlp2, ovlp4)

        return 2.*np.block([[dot_0**2/2., ovlp_0n.reshape(1,-1)], [ovlp_m0.reshape(-1,1), ovlp_mn]])


def change_phase(x0, y0, x1, y1, mo0, mo1, ovlp):
    nroots, no, nv = x1.shape
    ovlp = np.einsum('mp,mn,nq->pq', mo0, ovlp, mo1)
    idx = np.argmax(np.abs(ovlp), axis=0) # large index for each column
    #print('idx:', idx)

    for i, j in enumerate(idx):
        if ovlp[i,j] < 0.:
            print(i, j)
            mo1[:,j] *= -1
            if j < no:
                x1[:,j,:] *= -1.
            else:
                x1[:,:,j-no] *= -1.
            if isinstance(y1, np.ndarray):
                if j < no:
                    y1[:,j,:] *= -1.
                else:
                    y1[:,:,j-no] *= -1.

    return x1, y1, mo1


def sign_fixing(mat):
    """
    refer Zhou, Subotnik JCTC 2020, 10.1021/acs.jctc.9b00952
    """
    #U, s, Vt = np.linalg.svd(mat)
    #mat = np.einsum('ij,jk->ik', U, Vt)

    if np.linalg.det(mat) < 0.:
        mat[:,0] *= -1.

    nroots = mat.shape[0]

    # Jacobi sweeps
    converged = False
    while not converged:
        converged = True

        for i in range(nroots):
            for j in range(nroots):
                dot  = 3.* (mat[i,i]**2 + mat[j,j]**2)
                dot += 6.* (mat[i,j] * mat[j,i])
                dot += 8.* (mat[i,i] + mat[j,j])
                dot -= 3.* (np.dot(mat[i,:], mat[:,i]) + np.dot(mat[j,:], mat[:,j]))

                if dot < 0.:
                    mat[:,i] *= -1.
                    mat[:,j] *= -1.
                    converged = False

    return mat



if __name__ == '__main__':
    h2o = """
            H    1.6090032   -0.0510674    0.4424329
            O    0.8596350   -0.0510674   -0.1653507
            H    0.1102668   -0.0510674    0.4424329
    """

    functional = 'hf'
    basis = '6-311+g*'
    spin = 0 # Nalpha - Nbeta
    charge = 0
    verbose = 0

    rpa = 1
    nroots = 5

    td_model = tdscf.TDDFT if rpa else tdscf.TDA

    mol = gto.M(
        atom    = h2o,
        basis   = basis,
        spin    = spin,
        charge  = charge,
        verbose = verbose
    )

    mf = scf.RKS(mol) # in coherent state
    mf.xc = functional
    mf.grids.prune = True
    e0 = mf.kernel()

    td = td_model(mf)
    td.nroots  = nroots
    td.verbose = 0
    td.kernel()

    nocc = (mf.mo_occ>0).sum()

    mo0 = mf.mo_coeff
    xy0 = td.xy
    x0, y0 = [None]*nroots, [None]*nroots
    for i in range(nroots):
        x0[i] = xy0[i][0].reshape(nocc, -1)
        if rpa:
            y0[i] = xy0[i][1].reshape(nocc, -1)
    x0, y0 = np.array(x0), np.array(y0)
    if not rpa: y0 = None


    coords = mol.atom_coords() # bohr
    #coords[0, 2] += .1 # bohr
    coords[1, 2] -= .1 # bohr

    mol1 = mol.set_geom_(coords, inplace=False, unit='bohr')
    mf1 = scf.RKS(mol1) # in coherent state
    mf1.xc = functional
    mf1.grids.prune = True

    e1 = mf1.kernel()

    td1 = td_model(mf1)
    td1.nroots  = nroots
    td1.verbose = 0
    td1.kernel()

    mo1 = mf1.mo_coeff
    xy1 = td1.xy
    x1, y1 = [None]*nroots, [None]*nroots
    for i in range(nroots):
        x1[i] = xy1[i][0].reshape(nocc, -1)
        if rpa:
            y1[i] = xy1[i][1].reshape(nocc, -1)
    x1, y1 = np.array(x1), np.array(y1)
    if not rpa: y1 = None

    ovlp = gto.intor_cross('int1e_ovlp', mol, mol1)

    x1, y1, mo1 = change_phase(x0, y0, x1, y1, mo0, mo1, ovlp)

    #print_matrix('mo0:', mo0, 5, 1)
    #print_matrix('mo1:', mo1, 5, 1)
    #print_matrix('x0:', x0.reshape(nroots, -1), 5, 1)
    #print_matrix('x1:', x1.reshape(nroots, -1), 5, 1)

    state_ovlp = cal_wf_overlap(x0, y0, x1, y1, mo0, mo1, ovlp)
    state_ovlp = sign_fixing(state_ovlp)
    print('state_ovlp:\n', state_ovlp)
