import os, sys
import numpy as np

from pyscf import scf, lib
from pyscf.mp.mp2 import MP2

from wavefunction_analysis.utils.pyscf_parser import *
from wavefunction_analysis.utils import print_matrix
from wavefunction_analysis.entanglement.mol_lo_tools import partition_lo_to_imps
from wavefunction_analysis.utils import get_ortho_basis


def normalize_wf(t2):
    #norm = np.linalg.norm(t2.ravel())
    c2 = np.einsum('ijab,ijab', t2, t2)
    print('c2:', c2)
    norm = 1. / np.sqrt(1. + c2)
    c0, c2 = norm,  t2*norm
    print('c0,c2:', c0*c0 +np.einsum('ijab,ijab', c2, c2))
    return c0, c2


def get_1rdm(c0, c2):
    nocc, nvir = c2.shape[1:3]
    c2d = np.subtract(2.*c2, c2.transpose(0,1,3,2)) # t_ijab - t_ijba
    #c2d = np.subtract(2.*c2d, c2d.transpose(1,0,2,3)) # (t_ijab) - (t_jiab)

    dmoo = -np.einsum('ijab,kjab->ik', c2, c2d)
    dmoo += np.diag(np.ones(nocc)*2.) # normalized

    dmvv = np.einsum('ijab,ijcb->ca', c2, c2d)

    print_matrix('dmoo:', dmoo, nind=1)
    print_matrix('dmvv:', dmvv, nind=1)

    eoo, v = np.linalg.eigh(dmoo)
    evv, v = np.linalg.eigh(dmoo)

    return np.concatenate((eoo, evv))


def get_2rdm(c0, c2):
    nocc, nvir = c2.shape[1:3]
    c2d = np.subtract(c2, c2.transpose(0,1,3,2)) # t_ijab - t_ijba

    diag = c0*c0 + np.einsum('ijab,ijab', c2, c2)
    den = np.diag(np.ones(n2)*diag)

    c2 = np.einsum('ijab,jlab->ijab', c2, c2)
    den1 = np.einsum('ijab,klab->ijkl', c2, c2)

    den = den + den1.reshape((n2, n2))

    e, v = np.linalg.eigh(den)
    return e.reshape(nocc, nocc)


def get_rdm_entropy(rho):
    return -np.einsum('i...,i...->i...', rho, np.log(rho))


def get_orbital_entropy_info(s1, s2, imethod=0):
    Iij = lib.direct_sum('i+j->ij', s1, s1) - s2
    if imethod == 0:
        return Iij
    elif imethod == 1: # White et al.
        np.fill_diagonal(Iij, 0.)
        return -.5*Iij



if __name__ == '__main__':
    infile = 'h2o.in'
    if len(sys.argv) >= 2: infile = sys.argv[1]
    parameters = parser(infile)
    results = run_pyscf_final(parameters)
    mol, mf = results['mol'], results['mf']
    print('mf energy:', mf.energy_tot())
    nbas = mf.mo_coeff.shape[0]

    pt = MP2(mf)
    #pt = MP2(scf.density_fit(mf, 'weigend'))
    emp2, t2 = pt.kernel()
    print('emp2:', emp2)
    nocc, nvir = t2.shape[1:3]
    print('nbas:', nbas, 'nocc:', nocc, 'nvir:', nvir)
    nov = nocc * nvir

    arg = np.argsort(-np.abs(t2.ravel()))
    print(arg)
    args = np.zeros([nov*nov, 4], dtype=int)
    ic = 0
    for i in range(nocc):
        for j in range(nocc):
            for a in range(nvir):
                for b in range(nvir):
                    args[ic] = np.array([i,j,a,b])
                    ic += 1
    for ic in range(20):
        i, j, a, b = args[arg[ic]]
        print('%2d %2d %2d %2d %10.6f' % (i+1, j+1, a+1+nocc, b+1+nocc, t2[i,j,a,b]))

    from pyscf.mp.mp2 import _gamma1_intermediates
    dmoo, dmvv = _gamma1_intermediates(pt)
    print_matrix('dmoo gamma:', dmoo, nind=1)
    print_matrix('dmvv gamma:', dmvv, nind=1)
    c0, c2 = normalize_wf(t2)
    #c0, c2 = 1., t2
    p1 = get_1rdm(c0, c2)
    print_matrix('p1:', p1)
    p2 = get_2rdm(c0, c2)
    print_matrix('p2:', p2)

    s1 = get_rdm_entropy(p1)
    print_matrix('s1:', s1)
    s2 = get_rdm_entropy(p2)
    print_matrix('s2:', s2)
    Iij = get_orbital_entropy_info(s1, s2)
    print_matrix('orbital-pair mutual information:', Iij)

