import sys
import numpy as np

from pyscf import lib
from pyscf import gto
from pyscf.lib import logger
from pyscf.dft.rks import RKS
from pyscf.grad import rks as rks_grad

from wavefunction_analysis.utils.pyscf_parser import *
from wavefunction_analysis.utils import convert_units, print_matrix, fdiff
from wavefunction_analysis.polariton.qed_ks import polariton_cs, get_lambda2
from wavefunction_analysis.utils.fdiff import change_matrix_phase_c

def finite_difference(mf, norder=2, step_size=1e-4, ideriv=2, extra=False):
    mo = mf.mo_coeff
    functional, prune = mf.xc, mf.grids.prune
    coupling = mf.c_lambda if hasattr(mf, 'c_lambda') else None

    mol = mf.mol
    natoms = mol.natm
    coords = mol.atom_coords()

    fd_e, fd_mo, fd_dip, fd_g = [], [], [], []
    for n in range(natoms):
        for x in range(3):
            fd = fdiff(norder, step_size)
            coords_new = fd.get_x(coords, [n, x])

            g1, mo1, gs_dipole1, transp = [], [], [], []
            for k in range(coords_new.shape[0]):
                mol_new = mol.set_geom_(coords_new[k], inplace=False, unit='bohr')
                mf1 = polariton_cs(mol_new) # in coherent state
                mf1.xc = functional
                mf1.grids.prune = prune
                mf1.get_multipole_matrix(coupling)

                e_tot = mf1.kernel()

                if ideriv == 1:
                    g1.append(e_tot)
                elif ideriv == 2:
                    de1 = mf1.Gradients().kernel()
                    g1.append(de1)

                if extra:
                    mo1.append(change_matrix_phase_c(mo, mf1.mo_coeff))
                    #print_matrix('mo '+str(n)+' '+str(x)+' '+str(k)+':', mo1[-1], 5, 1)
                    gs_dipole1.append(mf1.dip_moment(mol_new, unit='au'))
                    if isinstance(extra, float) or isinstance(extra, list) or isinstance(extra, np.ndarray): # extra is the frequency
                        transp.append(np.einsum('...pq,qp,...->...', mf1.dipole, mf1.make_rdm1(), extra))

            fd_e.append( fd.compute_fdiff(np.array(g1)))

            if extra:
                fd_mo.append( fd.compute_fdiff(np.array(mo1)))
                fd_dip.append( fd.compute_fdiff(np.array(gs_dipole1)))
                if isinstance(extra, float) or isinstance(extra, list) or isinstance(extra, np.ndarray): # extra is the frequency
                    fd_g.append( fd.compute_fdiff(np.array(transp)) )

    return np.reshape(fd_e, (3*natoms, -1)), np.array(fd_mo), np.array(fd_dip), np.array(fd_g)


def cal_multipole_matrix_fd(mol, dm=None, origin=None, norder=2, step_size=1e-4, ideriv=1):
    # no coupling strength
    if origin is None:
        origin = np.zeros(3)

    coords = mol.atom_coords() # in bohr
    natoms = mol.natm
    nao = mol.nao_nr()
    #print_matrix('coords:\n', coords)

    combine = True if isinstance(dm, np.ndarray) else False

    dipole_d1, quadrupole_d1 = [], []
    for n in range(natoms):
        for x in range(3):
            fd = fdiff(norder, step_size)
            coords_new = fd.get_x(coords, [n, x])

            dipole, quadrupole = [], []
            for c in range(coords_new.shape[0]):
                mol_new = mol.set_geom_(coords_new[c], inplace=False, unit='bohr')
                #print_matrix('mol_new:', mol_new.atom_coords())
                with mol_new.with_common_orig(origin):
                    if ideriv == 1:
                        ints1 = mol_new.intor('int1e_r', comp=3, hermi=0)
                        ints2 = mol_new.intor('int1e_rr', comp=9, hermi=0)
                        if combine:
                            ints1 = np.einsum('xpq,...pq->...x', ints1, dm)
                            ints2 = np.einsum('xpq,...pq->...x', ints2, dm)
                            if dm.ndim == 2:
                                ints2 = ints2.reshape(3,3)
                            elif dm.ndim == 3:
                                ints2 = ints2.reshape(-1,3,3)
                        else:
                            ints2 = ints2.reshape(3,3,nao,nao)

                    elif ideriv == 2:
                        ints1, ints2 = cal_multipole_matrix_d1(mol_new, dm, origin)

                    dipole.append(ints1)
                    quadrupole.append(ints2)


            dipole_d1.append( fd.compute_fdiff(np.array(dipole)) )
            quadrupole_d1.append( fd.compute_fdiff(np.array(quadrupole)) )

    dipole_d1 = np.array(dipole_d1)
    quadrupole_d1 = np.array(quadrupole_d1)

    ## get pyscf style compact matrices
    #if not combine and ideriv == 2:
    #    dip_d1, qua_d1 = np.zeros((3,3,3,nao,nao)), np.zeros((3,3,3,3,nao,nao))
    #    dipole_d1 = dipole_d1.reshape(natoms, 3, natoms, 3, 3, nao, nao)
    #    quadrupole_d1 = quadrupole_d1.reshape(natoms, 3, natoms, 3, 3, 3, nao, nao)

    #    atmlst = range(mol.natm)
    #    aoslices = mol.aoslice_by_atom()
    #    for k, ia in enumerate(atmlst):
    #        p0, p1 = aoslices[ia,2:]

    #        tmp1 = dipole[:,:,p0:p1]
    #        tmp2 = quadrupole[:,:,p0:p1]

    return dipole_d1, quadrupole_d1


def cal_multipole_matrix_d1(mol, dm=None, origin=None): # no coupling strength
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

    # these derivatives don't distinguish nuclei
    dipole = - np.einsum('mxpq,...m->xqp...', dipole, c_lambda) # move mode index to the last for convenience latter
    quadrupole = - np.einsum('mnxpq,mn->xqp', quadrupole, get_lambda2(c_lambda))
    return dipole, quadrupole


def get_dse_2e(dipole, dipole_d1, dm, with_j=False, scale_k=.5): # c_lambda is included
    # scale k by 1/2 for restricted orbital case by default
    # we moved the mode index for lambda-dipole derivative to the last
    if dm.ndim == 2:
        vk = np.einsum('xpq...,...rs,qr->xps', dipole_d1, dipole, dm)
        if with_j is False:
            return vk*scale_k
        else:
            vj = np.einsum('xpq...,...rs,sr->xpq', dipole_d1, dipole, dm)
            return [vj, vk*scale_k]
    else: # multiple density matrices, ie uhf
        vk = np.einsum('xpq...,...rs,iqr->ixps', dipole_d1, dipole, dm)
        if with_j is False:
            return vk*scale_k
        else:
            vj = np.einsum('xpq...,...rs,isr->ixpq', dipole_d1, dipole, dm)
            return [vj, vk*scale_k]



class Gradients(rks_grad.Gradients):
    def get_veff(self, mol=None, dm=None):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()

        vxc = super().get_veff(mol, dm)
        exc = vxc.exc1_grid

        vxc += self.dse_fock_d1(mol, dm)

        return lib.tag_array(vxc, exc1_grid=exc)


    # add this to the gradient class so that can be called for the cphf
    def dse_fock_d1(self, mol, dm, dipole_d1=None, quadrupole_d1=None):
        mf = self.base
        # here the dipoles and quadrupoles have aleady combined with coupling
        if not isinstance(dipole_d1, np.ndarray):
            dipole_d1, quadrupole_d1 = get_multipole_matrix_d1(mol, mf.c_lambda, mf.origin)
        vdse_k = get_dse_2e(mf.dipole, dipole_d1, dm, with_j=False)
        return (.5*quadrupole_d1 - vdse_k)


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


    fde = finite_difference(mf, norder, step_size, ideriv=1)[0]
    fde = fde.reshape(-1, 3)
    print_matrix('fde:', fde)
    print('gradient diff:', np.linalg.norm(fde-de1))


    # check the multipole matrix derivatives
    #dipole_fd, quadrupole_fd = cal_multipole_matrix_fd(mol, dm=dm, norder=norder, step_size=step_size)
    #dipole_d1, quadrupole_d1 = cal_multipole_matrix_d1(mol, dm=dm)

    #print_matrix('dipole_fd:', dipole_fd, 5, 1)
    #print_matrix('dipole_d1:', dipole_d1, 5, 1)

    #print_matrix('quadrupole_fd:', quadrupole_fd, 5, 1)
    #print_matrix('quadrupole_d1:', quadrupole_d1, 5, 1)

    #print('dipole_d1 diff:', np.linalg.norm(dipole_fd-dipole_d1))
    #print('quadrupole_d1 diff:', np.linalg.norm(quadrupole_fd-quadrupole_d1))
