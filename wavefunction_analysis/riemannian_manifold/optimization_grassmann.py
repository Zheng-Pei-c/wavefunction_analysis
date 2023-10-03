import os, sys
import numpy as np
np.set_printoptions(precision=4, linewidth=200, suppress=True,)

import pyscf
from pyscf import lib, gto, scf

from utils import print_matrix

def get_orthogonal_basis(S):
    e, v = np.linalg.eigh(S)
    idx = e > 1e-15
    Z = np.dot(v[:,idx]/np.sqrt(e[idx]), v[:,idx].conj().T)
    L = np.dot(v[:,idx]*np.sqrt(e[idx]), v[:,idx].conj().T)
    inv = np.dot(v[:,idx]/e[idx], v[:,idx].conj().T)
    return Z, L, inv


def cs_decompose(T, full=True, scale=1.):
    U, s, Vt = np.linalg.svd(T, full_matrices=full)
    print('singular values:', s, np.cos(s), np.sin(s))
    #print('Ut*U:\n', np.einsum('ji,jk->ik', U, U))
    #print('Vt*V:\n', np.einsum('ij,kj->ik', Vt, Vt))

    #s *= scale
    c, s = np.cos(s), np.sin(s)
    return U, c, s, Vt


def geodesic_svd(B):
    v, o = B.shape
    U, c, s, Vt = cs_decompose(B, scale=.5)

    print('u vt:', U.shape, Vt.shape)
    z = np.zeros((v, o))
    bas = np.block([[Vt.T, z.T],
                    [z, U]])

    k, l = np.min([o, v]), np.abs(v-o)
    z = np.zeros((l, k))
    c, s = np.diag(c), np.diag(s)
    cs = np.block([[c, -s, z.T],
                   [s, c, z.T],
                   [z, z, np.eye(l)]])

    U, s, Vt = np.linalg.svd(B, full_matrices=True)
    cs = np.block([[np.zeros((k,k)), -np.diag(s), z.T],
                   [np.diag(s), np.zeros((k,k)), z.T],
                   [z, z, np.zeros((l,l))]])

    print_matrix('bas:\n', bas)
    print_matrix('cs:\n', cs)
    return np.einsum('ij,jk,lk->il', bas, cs, bas)


def geodesic_svd_compact(Y, T):
    U, c, s, Vt = cs_decompose(T, full=False)

    Y1 = np.einsum('ji,j,jk->ik', Vt, c, Vt)
    Y1 = np.einsum('ij,jk->ik', Y, Y1)
    Y1 += np.einsum('ij,j,jk->ik', U, s, Vt)
    return Y1


class Grassmann(object):
    def __init__(self, mf=None, S=None, Q=None, Y=None, P=None, method='steepest_descent'):
        self._mf = mf
        nocc = mf.mol.nelectron // 2

        if S is None: S = mf.get_ovlp()
        if Q is None: Q = mf.mo_coeff
        if Y is None: Y = mf.mo_coeff[:, :nocc]
        #P = mf.get_init_guess() *.5
        if P is None: P = mf.make_rdm1() *.5

        self.nocc = nocc
        self.nvir = Q.shape[0] - nocc

        self.S = S
        self._Q = Q
        self._Y = Y
        self._P = P
        self.Z, self.L, self.sinv = get_orthogonal_basis(S)

        self.method = method

        self.energy_tot = mf.energy_tot
        self.get_fock = mf.get_fock


    def kernel(self):
        C = self.init_guess()
        print_matrix('C initial:\n', C)

        #P = np.einsum('ij,jk,lk->il', C, self.I, C)
        #P = np.einsum('ji,jk,kl->il', self.Z, P, self.Z)
        P = self._P
        energy = self.energy_tot(dm=P*2)

        gradient = self.get_fock(dm=P*2) *0.5
        G = self.gradient_on_tangent_space(gradient, C)
        T = -np.copy(G)
        C, P = self.update(C, T)

        for i in range(1):
            C_old = np.copy(C)
            G_old = np.copy(G)
            energy_old = np.copy(energy)

            gradient = self.get_fock(dm=P*2) *0.5
            G = self.gradient_on_tangent_space(gradient, C)
            print_matrix('tangent:\n', G[self.nocc:, :self.nocc])

            if self.method == 'barzilai_borwein':
                T = self.barzilai_borwein_step(G_old, G, T)
            elif self.method == 'conjugate_gradient':
                T = self.conjugate_gradient_step(C, G_old, G, T)
            C, P = self.update(C, T)

            energy = self.energy_tot(dm=P*2)
            error = np.linalg.norm(C-C_old)
            print('i:', i+1, 'error:', error, 'energy:', energy, energy-energy_old)

            if error < 1e-5:
                break



class Projection_Grassmann(Grassmann):
    def gradient_on_tangent_space(self, P, gradient):
        PG = np.einsum('ij,jk->ik', P, gradient)
        GP = np.einsum('ij,jk->ik', gradient, P)
        return PG + GP - 2.*np.einsum('ij,jk->ik', P, GP)
        #return gradient - np.einsum('ij,jk->ik', P, gradient)



class Quotient_Grassmann(Grassmann):
    def init_guess(self):
        Y = np.copy(self._Y)
        Y = np.einsum('ji,jk->ik', self.L, Y)
        return Y


    def gradient_on_tangent_space(self, gradient, Y):
        P = np.einsum('ij,kj->ik', Y, Y)
        gradient = np.einsum('ij,jk->ik', gradient, Y)
        #return np.einsum('ij,jk->ik', (self.sinv - self._P), gradient)
        return np.einsum('ij,jk->ik', (np.eye(P.shape[0]) - P), gradient)


    #def tangent_retraction(self, tangent, method='svd'):
    #    if method == 'svd':
    #        U = exp_decompose_cs(tangent)
    #    elif method == 'multiply':
    #        U = exp_decompose_cs2(tangent)
    #    elif method == 'qr':
    #        U = exp_decompose_qr(tangent)
    #    elif method == 'direct':
    #        U = np.exp(tangent)


    def tangent_parallel_transport(self, Y, G, T):
        U, s, Vt = np.linalg.svd(T, full_matrices=False)
        sVt = np.einsum('i,ik->ik', s, Vt)
        c, s = np.cos(s), np.sin(s)

        dT = -np.einsum('ij,kj,k->ik', Y, Vt, s)
        dT += np.einsum('ij,j->ij', U, c)
        dG = dT - U
        dT = np.einsum('ij,jk->ik', dT, sVt)
        dG = np.einsum('ij,kj,kl->il', dG, U, G)
        dG += G
        return dT, dG


    def conjugate_gradient_step(self, Y, G0, G1, T0):
        dT, dG = self.tangent_parallel_transport(Y, G0, T0)
        gamma = np.einsum('ij,ij->', G1-dG, G1) / np.einsum('ij,ij->', G0, G0)
        return (gamma*dT) - G1


    def barzilai_borwein_step(self, G0, G1, T0):
        dG = G1 - G0
        alpha = np.einsum('ij,ij->', dG, T0) / np.einsum('ij,ij->', dG, dG)
        return -alpha*G1


    def update(self, Y, T):
        Y = geodesic_svd_compact(Y, T)

        P = np.einsum('ij,kj->ik', Y, Y)
        P = np.einsum('ji,jk,kl->il', self.Z, P, self.Z) # AO

        return Y, P



class Involution_Grassmann(Grassmann):
    def init_guess(self):
        self.I = np.diag(np.concatenate((np.ones(self.nocc), -np.ones(self.nvir))))

        C = np.copy(self._Q)
        C = np.einsum('ji,jk->ik', self.L, C)
        return C


    def gradient_on_tangent_space(self, gradient, V):
        gradient = np.einsum('ij,jk,lk->il', self.Z, gradient, self.Z)
        return np.einsum('ji,jk,kl->il', V, gradient, V)


    def barzilai_borwein_step(self, G0, G1, T0):
        dG = G1[self.nocc:, :self.nocc] - G0[self.nocc:, :self.nocc]
        alpha = np.einsum('ij,ij->', dG, T0[self.nocc:, :self.nocc]) / np.einsum('ij,ij->', dG, dG)
        print('alpha:', alpha)
        return -alpha*G1


    def update(self, C, T):
        #print('C:\n', C)
        #print('I:\n', I)
        #Q = np.einsum('ij,jk,lk->il', C, self.I, C)
        #print('trQ:\n', np.trace(Q))
        #print('Q:\n', Q)
        #print('QQ:\n', np.einsum('ij,jk->ik', Q, Q))

        #V, R = np.linalg.qr(.5*(np.eye(C.shape[0]) + Q))
        #print('V:\n', V)
        ##print('R:\n', R)

        exp = geodesic_svd(T[self.nocc:, :self.nocc])
        print_matrix('exp:\n', exp)
        exp = np.zeros(T.shape)
        exp[self.nocc:, :self.nocc] = np.copy(T[self.nocc:, :self.nocc])
        exp[:self.nocc, self.nocc:] = -np.copy(T[:self.nocc, self.nocc:])
        print_matrix('exp2 tmp:\n', exp)
        exp = np.exp(exp)
        print('exp2:\n', exp)
        C = np.einsum('ij,jk->ik', C, exp) # orthogonal
        #print_matrix('C:\n', C)
        print_matrix('C:\n', np.einsum('ji,jk->ik', self.Z, C))

        #P = np.einsum('ij,jk,lk->il', C, self.I, C)
        P = np.einsum('ij,kj->ik', C[:, :self.nocc], C[:, :self.nocc])
        P = np.einsum('ji,jk,kl->il', self.Z, P, self.Z) # AO

        return C, P



if __name__ == '__main__':
    atom = """
    O       0.   0.     0.
    H       0.  -0.757  0.587
    H       0.   0.757  0.587
    """

    mol = gto.Mole(
            atom = atom,
            basis = 'sto-3g'
            )
    mol.build()

    mf = scf.RHF(mol)
    mf.max_cycle = 1
    mf.kernel()

    models = ['projection', 'quotient', 'involution']
    methods = ['barzilai_borwein', 'conjugate_gradient']

    model, method = models[int(sys.argv[1])], methods[int(sys.argv[2])]

    if model == 'projection':
        grass = Projection_Grassmann(mf=mf)
    elif model == 'quotient':
        grass = Quotient_Grassmann(mf=mf, method=method)
    elif model == 'involution':
        grass = Involution_Grassmann(mf=mf, method=method)

    grass.kernel()


    #mf.verbose = 5
    mf.max_cycle = 50
    mf.kernel()
    print('C:\n', mf.mo_coeff)
    #mo = mf.mo_coeff
    #gradient = mf.get_fock(dm=mf.make_rdm1()) *0.5
    #print('Fock:', np.einsum('ji,jk,kl->il', mo, gradient, mo))
