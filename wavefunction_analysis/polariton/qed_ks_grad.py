import sys
import numpy as np

from pyscf import lib
from pyscf import gto
from pyscf.lib import logger
from pyscf.dft.rks import RKS
from pyscf.grad import rks as rks_grad

from wavefunction_analysis.utils.pyscf_parser import *
from wavefunction_analysis.utils import convert_units, print_matrix, fdiff
from wavefunction_analysis.polariton.qed_ks import polariton_cs


def check_multipole_matrix_fd(mol, dm=None, origin=None, norder=2, step_size=1e-4):
    if origin is None:
        origin = np.zeros(3)

    coords = mol.atom_coords() # in bohr
    natoms = mol.natm
    #print_matrix('coords:\n', coords)

    dipole_d1, quadrupole_d1 = [], []
    for n in range(natoms):
        for x in range(3):
            fd = fdiff(norder, step_size)
            coords_new = fd.get_x(coords, [n, x])

            dipole, quadrupole = [], []
            for k in range(coords_new.shape[0]):
                mol_new = mol.set_geom_(coords_new[k], inplace=False, unit='bohr')
                #print_matrix('mol_new:', mol_new.atom_coords())
                with mol_new.with_common_orig(origin):
                    ints = mol_new.intor('int1e_r', comp=3, hermi=0)
                    dipole.append(ints)

                    ints = mol_new.intor('int1e_rr', comp=9, hermi=0)
                    quadrupole.append(ints)

            dipole_d1.append( fd.compute_fdiff(np.array(dipole)) )
            quadrupole_d1.append( fd.compute_fdiff(np.array(quadrupole)) )

    if isinstance(dm, np.ndarray): # combine with density matrix
        dipole_d1 = np.einsum('ixpq,pq->ix', dipole_d1, dm)
        quadrupole_d1 = np.einsum('ixpq,pq->ix', quadrupole_d1, dm).reshape(natoms*3,3,3)
    else:
        nao = mol.nao_nr()
        dipole_d1 = np.array(dipole_d1)
        quadrupole_d1 = np.array(quadrupole_d1).reshape(natoms*3,3,3,nao,nao)

    return dipole_d1, quadrupole_d1


def check_multipole_matrix_d1(mol, dm=None, origin=None):
    if origin is None:
        origin = np.zeros(3)

    natoms = mol.natm
    nao = mol.nao_nr()
    with mol.with_common_orig(origin):
        # move derivative as the first index and the ket as bra at the same time
        dipole = mol.intor('int1e_irp', comp=9, hermi=0).reshape(3,3,nao,nao).transpose(1,0,3,2)
        quadrupole = mol.intor('int1e_irrp', comp=27, hermi=0).reshape(9,3,nao,nao).transpose(1,0,3,2)

    #print_matrix('dipole 0:', dipole, 5, 1)

    combine = True if isinstance(dm, np.ndarray) else False
    if combine:
        dipole_d1 = np.zeros((natoms, 3, 3))
        quadrupole_d1 = np.zeros((natoms, 3, 9))
    else:
        dipole_d1 = np.zeros((natoms, 3, 3, nao, nao))
        quadrupole_d1 = np.zeros((natoms, 3, 9, nao, nao))

    atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia,2:]

        tmp1 = dipole[:,:,p0:p1]
        tmp2 = quadrupole[:,:,p0:p1]

        if combine:
            dipole_d1[k] = - 2.* np.einsum('ixpq,pq->ix', tmp1, dm[p0:p1])
            quadrupole_d1[k] = - 2.* np.einsum('ixpq,pq->ix', tmp2, dm[p0:p1])
        else:
            dipole_d1[k,:,:,p0:p1] -= tmp1
            dipole_d1[k,:,:,:,p0:p1] -= tmp1.transpose(0,1,3,2)

            quadrupole_d1[k,:,:,p0:p1] -= tmp2
            quadrupole_d1[k,:,:,:,p0:p1] -= tmp2.transpose(0,1,3,2)

    if combine:
        return dipole_d1.reshape(natoms*3,3), quadrupole_d1.reshape(natoms*3,3,3)
    else:
        return dipole_d1.reshape(natoms*3,3,nao,nao), quadrupole_d1.reshape(natoms*3,3,3,nao,nao)


def get_multipole_matrix_d1(mol, c_lambda, origin=None):
    if origin is None:
        origin = np.zeros(3)

    nao = mol.nao_nr()
    with mol.with_common_orig(origin):
        # this derivatives have extra minus
        dipole = mol.intor('int1e_irp', comp=9, hermi=0).reshape(3,3,nao,nao)
        quadrupole = mol.intor('int1e_irrp', comp=27, hermi=0).reshape(3,3,3,nao,nao)

    c2 = np.einsum('...x,...y->...xy', c_lambda, c_lambda)
    dipole = - np.einsum('xypq,...x->yqp', dipole, c_lambda)
    quadrupole = - np.einsum('xyzpq,...xy->zqp', quadrupole, c2)
    return dipole, quadrupole


def get_dipole_2e(dipole, dm, with_j=False, scale_k=.5): # c_lambda is included
    # scale k by 1/2 for restricted orbital case by default
    # dipole could be its derivative
    if dm.ndim == 2:
        vk = np.einsum('...rs,qr->...qs', dipole, dm)
        if with_j is False:
            return vk*scale_k
        else:
            vj = np.einsum('rs,...rs->...', dipole, dm)
            return [vj, vk*scale_k]
    else: # multiple density matrices, ie uhf
        vk = np.einsum('...rs,iqr->i...qs', dipole, dm)
        if with_j is False:
            return vk*scale_k
        else:
            vj = np.einsum('...rs,irs->i...', dipole, dm)
            return [vj, vk*scale_k]



class Gradients(rks_grad.Gradients):
    def get_veff(self, mol=None, dm=None):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()

        vxc = super().get_veff(mol, dm)
        exc = vxc.exc1_grid

        # here the dipoles and quadrupoles have aleady combined with coupling
        dipole_d1, quadrupole_d1 = get_multipole_matrix_d1(mol, self.base.c_lambda)
        vdse_k = get_dipole_2e(self.base.dipole, dm, with_j=False)
        vdse_k = np.einsum('xpq,...qs->x...ps', dipole_d1, vdse_k)

        vxc += (.5*quadrupole_d1 - vdse_k)

        return lib.tag_array(vxc, exc1_grid=exc)

polariton_cs.Gradients = lib.class_as_method(Gradients)



if __name__ == '__main__':
    #infile = 'h2o.in'
    #parameters = parser(infile)

    #charge, spin, atom = parameters.get(section_names[0])[1:4]
    #functional, basis = get_rem_info(parameters.get(section_names[1]))[:2]
    #mol = build_single_molecule(charge, spin, atom, basis, verbose=0)

    hf = """
            H    0. 0. -0.459
            F    0. 0.  0.459
    """
    h2o = """
            H    1.6090032   -0.0510674    0.4424329
            O    0.8596350   -0.0510674   -0.1653507
            H    0.1102668   -0.0510674    0.4424329
    """
    atom = locals()[sys.argv[1]] if len(sys.argv) > 1 else hf

    functional = 'hf'
    mol = build_single_molecule(0, 0, atom, '3-21g')

    frequency = 0.42978 # gs doesn't depend on frequency
    coupling = np.array([0, 0, .05])

    mf = polariton_cs(mol) # in coherent state
    mf.xc = functional
    mf.grids.prune = True
    mf.get_multipole_matrix(coupling)

    e_tot = mf.kernel()
    print('scf energy:', e_tot)
    dm = mf.make_rdm1()
    #print_matrix('dm:', dm, 5, 1)

    g = mf.Gradients()
    de1 = g.kernel()
    print_matrix('de1:', de1)

    norder, step_size = 2, 1e-4

    natoms = mol.natm
    coords = mol.atom_coords()

    fde = np.zeros((natoms, 3))
    for n in range(natoms):
        for x in range(3):
            fd = fdiff(norder, step_size)
            coords_new = fd.get_x(coords, [n, x])
            d = fd.get_d(3, x)

            energy = []
            for k in range(coords_new.shape[0]):
                mol_new = mol.set_geom_(coords_new[k], inplace=False, unit='bohr')
                mf1 = polariton_cs(mol_new) # in coherent state
                mf1.xc = functional
                mf1.grids.prune = True
                # the finite-difference need to change the origin to the com!
                mf1.get_multipole_matrix(coupling, origin=get_center_of_mass(mol_new))

                e_tot = mf1.kernel()
                energy.append(e_tot)

            fde[n,x] = fd.compute_fdiff(np.array(energy))

    print_matrix('fde:', fde)
    print('gradient diff:', np.linalg.norm(fde-de1))


    # check the multipole matrix derivatives
    #dipole_fd, quadrupole_fd = check_multipole_matrix_fd(mol, dm=dm, norder=norder, step_size=step_size)
    #dipole_d1, quadrupole_d1 = check_multipole_matrix_d1(mol, dm=dm)

    #print_matrix('dipole_fd:', dipole_fd, 5, 1)
    #print_matrix('dipole_d1:', dipole_d1, 5, 1)

    #print_matrix('quadrupole_fd:', quadrupole_fd, 5, 1)
    #print_matrix('quadrupole_d1:', quadrupole_d1, 5, 1)

    #print('dipole_d1 diff:', np.linalg.norm(dipole_fd-dipole_d1))
    #print('quadrupole_d1 diff:', np.linalg.norm(quadrupole_fd-quadrupole_d1))
