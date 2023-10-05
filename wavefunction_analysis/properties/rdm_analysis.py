import os, sys
import numpy as np

from pyscf import scf, tdscf, gto, lib
from utils.pyscf_parser import *

def get_transition_dm(xy, coeff1, coeff2=None, scale=2.):
    """xy is a tuple (x, y)"""
    o, v = xy[0].shape
    rdm1 = np.einsum('pi,ia,qa->pq', coeff1[:,:o], xy[0], coeff1[:,o:].conj())
    if isinstance(xy[1], np.ndarray):
        rdm1 += np.einsum('pi,ia,qa->qp', coeff1[:,:o], xy[1], coeff1[:,o:].conj())

    return scale * rdm1


def get_attach_dm(xy1, xy2, orbv1, orbv2=None, scale=2.):
    """xy is a tuple (x, y)"""
    rdm1 = np.einsum('ia,ib->ab', xy1[0], xy2[0])
    if isinstance(xy1[1], np.ndarray):
        rdm1 += np.einsum('ia,ib->ba', xy1[1], xy2[1])
    return scale * np.einsum('pa,ab,qb->pq', orbv1, rdm1, orbv1.conj())


def get_detach_dm(xy1, xy2, orbo1, orbo2=None, scale=2.):
    """xy is a tuple (x, y)"""
    rdm1 = np.einsum('ia,ja->ij', xy1[0], xy2[0])
    if isinstance(xy1[1], np.ndarray):
        rdm1 += np.einsum('ia,ja->ji', xy1[1], xy2[1])
    return -scale * np.einsum('pi,ij,qj->pq', orbo1, rdm1, orbo1.conj())


def get_difference_dm(xy1, xy2, coeff1, coeff2=None, scale=2.):
    """xy is a tuple (x, y)"""
    o, v = xy1[0].shape
    orbo1, orbv1 = coeff1[:,:o], coeff1[:,o:]
    orbo2, orbv2 = None, None
    if isinstance(coeff2, np.ndarray):
        orbo2, orbv2 = coeff2[:,:o], coeff2[:,o:]

    rdm1  = get_attach_dm(xy1, xy2, orbv1, orbv2, scale)
    rdm1 += get_detach_dm(xy1, xy2, orbo1, orbo2, scale)

    return rdm1


def get_dipoles(dip_mat, rdm):
    return np.einsum('xpq,...pq->...x', dip_mat, rdm)


def cal_rdm1(xys, coeff, scale=2., itype='trans'):
    rdm1 = []
    nstate = len(xys)
    if 'trans' in itype:
        for i in range(nstate):
            rdm1.append(get_transition_dm(xys[i], coeff, None, scale))
    if 'diff' in itype:
        for i in range(nstate):
            for j in range(i, nstate):
                rdm1.append(get_difference_dm(xys[i], xys[j], coeff, None, scale))

    return np.array(rdm1)


def cal_dipoles(dip_mat, xys, coeff, scale=2., itype='trans'):
    dipoles = []
    nstate = len(xys)
    if 'trans' in itype:
        for i in range(nstate):
            rdm1 = get_transition_dm(xys[i], coeff, None, scale)
            dipoles.append(get_dipoles(dip_mat, rdm1))
    if 'diff' in itype:
        for i in range(nstate):
            for j in range(i, nstate):
                rdm1 = get_difference_dm(xys[i], xys[j], coeff, None, scale)
                dipoles.append(get_dipoles(dip_mat, rdm1))

    return np.array(dipoles)



if __name__ == '__main__':
    infile = sys.argv[1]
    parameters = parser(infile)
    mol, mf, td = run_pyscf_final(parameters)
    if not isinstance(mol, list):
        mol, mf, td = [mol], [mf], [td]

    dipoles = []
    for n in range(len(mol)):
        #print_matrix('dipole reference:', td[n].transition_dipole())
        coeff = mf[n].mo_coeff
        xys = td[n].xy
        dip_mat = mol[n].intor('int1e_r', comp=3)
        dipole = cal_dipoles(dip_mat, xys, coeff, scale=2., itype='diff')
        #dipoles.append(dipole)
        #print_matrix('dipole:', dipole)

        nstate = len(xys)
        dipole2 = np.zeros((nstate, nstate, 3))
        icount = 0
        for i in range(len(xys)):
            for j in range(i, len(xys)):
                dipole2[i,j] = dipole2[j,i] = dipole[icount]
                icount += 1

        print_matrix('dipole:', dipole2.reshape(nstate, -1))
    #dipoles = np.array(dipoles)
