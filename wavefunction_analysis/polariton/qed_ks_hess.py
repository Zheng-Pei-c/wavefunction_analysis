import sys
import numpy as np

from pyscf import lib
from pyscf.lib import logger
from pyscf.dft.rks import RKS
from pyscf.hessian import rhf as rhf_hess
from pyscf.hessian import rks as rks_hess

from wavefunction_analysis.utils.pyscf_parser import *
from wavefunction_analysis.utils import convert_units, print_matrix, fdiff
from wavefunction_analysis.polariton.qed_ks import polariton_cs, get_lambda2
from wavefunction_analysis.polariton.qed_ks_grad import get_multipole_matrix_d1, cal_multipole_matrix_fd
from wavefunction_analysis.utils.fdiff import change_matrix_phase_c


def resemble_deriv_on_atoms(mol, mat0):
    # mat0 has dimension of (3, nao, nao)
    atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()
    nao = mat0.shape[-1]

    mat1 = np.zeros((mol.natm, 3, nao, nao))
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia,2:]
        mat1[k,:,p0:p1] += mat0[:,p0:p1]
        mat1[k,:,:,p0:p1] += mat0[:,p0:p1].transpose(0,2,1)

    return mat1.reshape(-1, nao, nao)


def get_multipole_matrix_d2(mol, c_lambda, origin=None):
    if origin is None:
        origin = np.zeros(3)

    #natoms = mol.natm
    #nao = mol.nao_nr()
    #with mol.with_common_orig(origin):
    #    # this derivatives have extra minus
    #    dipole = mol.intor('int1e_irp', comp=9, hermi=0).reshape(3,3,nao,nao)
    #    quadrupole = mol.intor('int1e_irrp', comp=27, hermi=0).reshape(3,3,3,nao,nao)
    #c2 = np.einsum('...x,...y->...xy', c_lambda, c_lambda)
    #if c2.ndim == 3: # contract modes
    #    c2 = np.sum(c2, axis=0)
    #dipole = - np.einsum('mxypq,...m->xyqp', dipole, c_lambda)
    #quadrupole = - np.einsum('mnxypq,...mn->xyqp', quadrupole, c2)

    norder, step_size = 3, 1e-4
    dipole, quadrupole = cal_multipole_matrix_fd(mol, dm=None, norder=norder, step_size=step_size, ideriv=2)

    # x and y are nuclear derivatives (3N, 3N)
    dipole = np.einsum('xympq,...m->...xyqp', dipole, c_lambda)
    quadrupole = np.einsum('xymnpq,mn->xyqp', quadrupole, get_lambda2(c_lambda))

    return dipole, quadrupole


def get_dse_2e_a(dipole, dipole_d2, dm, with_j=False, scale_k=.5): # c_lambda is included
    # scale k by 1/2 for restricted orbital case by default
    if dm.ndim == 2:
        vk = np.einsum('sp,...rs,qr->...qp', dm, dipole, dm)
    else: # multiple density matrices, ie uhf
        vk = np.einsum('isp,...rs,iqr->...qp', dm, dipole, dm)
    vk = np.einsum('...xypq,...qp->xy', dipole_d2, vk)
    if with_j is False:
        return vk*scale_k

    if dm.ndim == 2:
        vj = np.einsum(',...rs,sr->...', dipole, dm)
        vj = np.einsum('...xypq,qp,...->xy', dipole_d2, dm, vj)
    else: # multiple density matrices, ie uhf
        vj = np.einsum('...rs,isr->...i', dipole, dm)
        vj = np.einsum('...xypq,iqp,...i->xy', dipole_d2, dm, vj)
    return [vj, vk*scale_k]


def get_dse_2e_s(dipole_d1, dm, with_j=False, scale_k=.5): # c_lambda is included
    # scale k by 1/2 for restricted orbital case by default
    if dm.ndim == 2:
        vk = np.einsum('...xrs,qr->...xqs', dipole_d1, dm)
        vk = np.einsum('...xqs,...ysq->xy', vk, vk)
        if with_j is False:
            return vk*scale_k
        else:
            vj = np.einsum('...rs,sr->...x', dipole_d1, dm)
            vj = np.einsum('...x,...y->xy', vj, vj)
            return [vj, vk*scale_k]
    else: # multiple density matrices, ie uhf
        vk = np.einsum('...xrs,iqr->ixqs', dipole_d1, dm)
        vk = np.einsum('...ixqs,...iysq->xy', vk, vk)
        if with_j is False:
            return vk*scale_k
        else:
            vj = np.einsum('...xrs,isr->i...x', dipole_d1, dm)
            vj = np.einsum('i...x,i...y->xy', vj, vj)
            return [vj, vk*scale_k]


def partial_hess_elec(hessobj, mo_energy=None, mo_coeff=None, mo_occ=None,
                      atmlst=None, max_memory=4000, verbose=None):
    mol = hessobj.mol
    mf = hessobj.base

    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    mo_coeff = mf.mo_coeff
    atmlst = range(mol.natm)

    nao, nmo = mo_coeff.shape
    mocc = mo_coeff[:,mo_occ>0]
    dm0 = np.dot(mocc, mocc.T) * 2

    de2 = rks_hess.partial_hess_elec(hessobj, mo_energy, mo_coeff, mo_occ,
                                     atmlst, max_memory, verbose)

    dipole_d2, quadrupole_d2 = get_multipole_matrix_d2(mol, mf.c_lambda, hessobj.base.origin)
    vdse = .5* np.einsum('xypq,...qp->xy', quadrupole_d2, dm0)

    vdse -= get_dse_2e_a(mf.dipole, dipole_d2, dm0, with_j=False)

    dipole_d1, _ = get_multipole_matrix_d1(mol, mf.c_lambda, mf.origin)
    dipole_d1 = resemble_deriv_on_atoms(mol, dipole_d1)
    vdse -= get_dse_2e_s(dipole_d1, dm0, with_j=False)

    natoms = mol.natm
    de2 += vdse.reshape(natoms, 3, natoms, 3).transpose(0,2,1,3)

    ## explicit derivatives of dse contribution
    #for i0, ia in enumerate(atmlst):
    #    shl0, shl1, p0, p1 = aoslices[ia]
    #    shls_slice = (shl0, shl1) + (0, mol.nbas)*3

    return de2


def dse_fock_d1_s(dipole, dipole_d1, dm, with_j=False, scale_k=.5): # c_lambda is included
    # scale k by 1/2 for restricted orbital case by default
    # the density passed through is piece by piece
    if dm.ndim == 2:
        vk = np.einsum('...pq,...xrs,qr->xps', dipole, dipole_d1, dm)
        if with_j is False:
            return vk*scale_k
        else:
            vj = np.einsum('...pq,...xrs,sr->xpq', dipole, dipole_d1, dm)
            return [vj, vk*scale_k]
    else: # multiple density matrices, ie uhf
        vk = np.einsum('...pq,...xrs,iqr->ixps', dipole, dipole_d1, dm)
        if with_j is False:
            return vk*scale_k
        else:
            vj = np.einsum('...pq,...xrs,isr->ixpq', dipole, dipole_d1, dm)
            return [vj, vk*scale_k]


# h1 is the right-hand-side of the CPHF for the MO response
# the S^[x] dependent terms are added in solve_mo1 function
# modified make_h1 based on rks hessian
def make_h1(hessobj, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
    mol = hessobj.mol
    if atmlst is None:
        atmlst = range(mol.natm)

    nao, nmo = mo_coeff.shape
    mocc = mo_coeff[:,mo_occ>0]
    dm0 = np.dot(mocc, mocc.T) * 2

    mf = hessobj.base

    grad_method = mf.nuc_grad_method()
    hcore_deriv = grad_method.hcore_generator(mol)

    # dse contributions
    dipole = mf.dipole
    dipole_d1, quadrupole_d1 = get_multipole_matrix_d1(mol, mf.c_lambda, mf.origin)
    dse_d1 = grad_method.dse_fock_d1(mol, dm0, dipole_d1, quadrupole_d1)

    ni = mf._numint
    ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
    hybrid = ni.libxc.is_hybrid_xc(mf.xc)

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, mf.max_memory*.9-mem_now)
    h1ao = rks_hess._get_vxc_deriv1(hessobj, mo_coeff, mo_occ, max_memory)
    aoslices = mol.aoslice_by_atom()
    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        shls_slice = (shl0, shl1) + (0, mol.nbas)*3
        if hybrid:
            vj1, vj2, vk1, vk2 = \
                    rhf_hess._get_jk(mol, 'int2e_ip1', 3, 's2kl',
                                     ['ji->s2kl', -dm0[:,p0:p1],  # vj1
                                      'lk->s1ij', -dm0         ,  # vj2
                                      'li->s1kj', -dm0[:,p0:p1],  # vk1
                                      'jk->s1il', -dm0         ], # vk2
                                     shls_slice=shls_slice)
            veff = vj1 - hyb * .5 * vk1
            veff[:,p0:p1] += vj2 - hyb * .5 * vk2
            if omega != 0:
                with mol.with_range_coulomb(omega):
                    vk1, vk2 = \
                        rhf_hess._get_jk(mol, 'int2e_ip1', 3, 's2kl',
                                         ['li->s1kj', -dm0[:,p0:p1],  # vk1
                                          'jk->s1il', -dm0         ], # vk2
                                         shls_slice=shls_slice)
                veff -= (alpha-hyb) * .5 * vk1
                veff[:,p0:p1] -= (alpha-hyb) * .5 * vk2
        else:
            vj1, vj2 = rhf_hess._get_jk(mol, 'int2e_ip1', 3, 's2kl',
                                        ['ji->s2kl', -dm0[:,p0:p1],  # vj1
                                         'lk->s1ij', -dm0         ], # vj2
                                        shls_slice=shls_slice)
            veff = vj1
            veff[:,p0:p1] += vj2

        veff -= dse_fock_d1_s(dipole, dipole_d1[:,p0:p1], dm0[:,p0:p1], with_j=False)
        veff[:,p0:p1] += dse_d1[:,p0:p1]
        h1ao[ia] += veff + veff.transpose(0,2,1)
        h1ao[ia] += hcore_deriv(ia)

    if chkfile is None:
        return h1ao
    else:
        for ia in atmlst:
            lib.chkfile.save(chkfile, 'scf_f1ao/%d'%ia, h1ao[ia])
        return chkfile



class Hessian(rks_hess.Hessian):
    partial_hess_elec = partial_hess_elec
    make_h1 = make_h1

polariton_cs.Hessian = lib.class_as_method(Hessian)




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
    mol = build_single_molecule(0, 0, atom, 'sto-3g')

    frequency = 0.42978 # gs doesn't depend on frequency
    coupling = np.array([0, 0, .1])

    mf = polariton_cs(mol) # in coherent state
    mf.xc = functional
    mf.grids.prune = True
    mf.get_multipole_matrix(coupling)

    natoms = mol.natm

    e_tot = mf.kernel()
    hess = mf.Hessian().kernel()
    hess = hess.transpose(0,2,1,3).reshape(natoms*3, natoms*3)
    print_matrix('hess:', hess, 5, 1)


    norder, step_size = 2, 1e-4

    coords = mol.atom_coords()

    nocc = (mf.mo_occ>0).sum()
    dm = mf.make_rdm1()
    mo = mf.mo_coeff
    nao = mo.shape[0]
    cct = np.einsum('mp,np->mn', mo, mo)
    ovlp = mf.get_ovlp()
    print_matrix('mo0:', mo, 5, 1)

    ovlp1 = mol.intor('int1e_ipovlp', comp=3)
    s1 = np.zeros((natoms, 3, nao, nao))
    atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia,2:]
        s1[k,:,p0:p1] += ovlp1[:,p0:p1]
        s1[k,:,:,p0:p1] += ovlp1[:,p0:p1].transpose(0,2,1)
    s1 = s1.reshape(-1,nao,nao)


    fde = np.zeros((natoms, 3, natoms, 3))
    fd_moe, fd_mo, theta = [], [], []
    for n in range(natoms):
        for x in range(3):
            fd = fdiff(norder, step_size)
            coords_new = fd.get_x(coords, [n, x])

            e1, mo1, g1 = [], [], []
            for k in range(coords_new.shape[0]):
                mol_new = mol.set_geom_(coords_new[k], inplace=False, unit='bohr')
                mf1 = polariton_cs(mol_new) # in coherent state
                mf1.xc = functional
                mf1.grids.prune = True
                mf1.get_multipole_matrix(coupling)

                e_tot = mf1.kernel()
                #mo1.append(mf1.mo_coeff)
                mo1.append(change_matrix_phase_c(mo, mf1.mo_coeff))
                print_matrix('mo '+str(n)+' '+str(x)+' '+str(k)+':', mo1[-1], 5, 1)
                #e1.append(np.einsum('pi,pq,qj->ij', mo1[-1][:,:nocc], mf1.get_fock(), mo1[-1][:,:nocc]))

                de1 = mf1.Gradients().kernel()
                g1.append(de1)

            fde[n,x] = fd.compute_fdiff(np.array(g1))

            #e1 = fd.compute_fdiff(np.array(e1))
            mo1 = fd.compute_fdiff(np.array(mo1))
            fd_mo.append(mo1)

    fd_mo = np.array(fd_mo)
    fd_mo[0::3,5,3] = 0.
    fd_mo[np.abs(fd_mo)>2.] = 0.
    #print_matrix('fd_mo:', np.array(fd_mo), 5, 1)

    sinv_sx = np.einsum('mn,xnp->xmp', cct, s1) * .5
    mo1_s = np.einsum('xmn,np->xmp', sinv_sx, mo)
    #print_matrix('mo1_s:', mo1_s[:,:,:nocc], 5, 1)

    theta = fd_mo - mo1_s
    print_matrix('theta pi:', theta, 5, 1)
    theta = np.einsum('ma,mn,xni->xai', mo[:,nocc:], ovlp, theta[:,:,:nocc])
    #print_matrix('theta:', theta, 5, 1)

    mo1_theta = np.einsum('pa,xai->xpi', mo[:,nocc:], theta)
    #print_matrix('mo1_theta', mo1_theta, 5, 1)
    print_matrix('mo1', mo1_theta+mo1_s[:,:,:nocc], 5, 1)


    fde = fde.reshape(natoms*3, natoms*3)
    print_matrix('fde:', fde, 5, 1)
    print('gradient diff:', np.linalg.norm(fde-hess))

    #from wavefunction_analysis.utils import read_matrix
    #coeff1 = read_matrix('qc-diff', 1, nao*nao, 'Alpha MO coefficients', 5)
    #coeff1 = coeff1.reshape(2, nao, nao).transpose(0,2,1)
    #coeff1[1] = change_matrix_phase_c(coeff1[0], coeff1[1])
    #print_matrix('coeff1:', coeff1, 5, 1)
    #print_matrix('coeff diff:', (coeff1[1]-coeff1[0])/2*5e3, 5, 1)
