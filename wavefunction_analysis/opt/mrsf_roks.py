from wavefunction_analysis import os, sys
from wavefunction_analysis.utils import print_matrix

import numpy
from pyscf import scf, tdscf, gto, lib
from pyscf.lib import logger
from pyscf.tdscf._lr_eig import eigh as lr_eigh, eig as lr_eig, real_eig
from pyscf.scf import hf, rohf, uhf
from pyscf.data import nist
from pyscf import __config__

MO_BASE = getattr(__config__, 'MO_BASE', 1)

"""
Multi-reference spin-flip (MRSF) DFT uses higher spin reference state
and calculates the real ground-state and excited-states with spin flipping.
references from Cheol Ho Choi
JCP 2019. 10.1063/1.5086895
JCTC 2021. 10.1021/acs.jctc.0c01074
JPCA 2024. 10.1021/acs.jpca.4c04521
"""

def mrsf_dimension_transform(nocc, nvir, nbas, singlet=True):
    r"""
    Build the transform matrix to fix the dimention of amplitudes
    for MRSF excited-state calculation.
    Refer to JCP 2019 10.1063/1.5086895.
    The spin-pairing of the A matrix can be seen from Fig. 1.
            [U S  ] [alpha->beta,  alpha->beta ] [U S]
            [UCO  ] [alpha->alpha, alpha->alpha] [UOV]
            [UOV  ] [beta ->beta,  beta ->beta ] [UCO]
            [UCOCO] [alpha->beta,  beta ->alpha] [UCOCO]
            [UOVOV] [beta ->alpha, alpga->beta ] [UOVOV]

    Parameters
        nocc : numbers of alpha and beta occupied orbitals (electrons)
        nvir : numbers of alpha and beta virtual orbitals
        nbas : number of AO basis set
        singlet : bool to calculate singlet (True) or triplet (False)

    Returns
        U : transform matrix for singlet and triplet excited-states
            [singlet or triplet, CO1, CO2, O1V, O2V, CO1CO2, O1VO2V]
            excitation types: C_alpha --> O_beta, O_alpha --> V_beta
                              C_alpha --> V_beta, C_beta --> V_alpha
            used for bra side
        Ut : reorder U as
            [s or t, O2V, O1V, CO2, CO1, CO1CO2, O1VO2V]
            for ket side
    """
    noa, nob = nocc
    nva, nvb = nvir
    print('noa:', noa, 'nob:', nob, 'nva:', nva, 'nvb:', nvb)
    nova = novb = noa * nvb # excitation space dimension
    nov = nova * 2 # maximum number of roots

    sqrt2 = 1. / numpy.sqrt(2.)

    U = numpy.zeros((7, noa, nvb)) # dimention transformation
    U[1, :nob, 0]   = 1. # UCO1  # C->O excitation Eqs. 2.6a and 2.7a
    U[2, :nob, 1]   = 1. # UCO2
    U[3, nob, 2:]   = 1. # UO1V  # O->V excitation Eqs. 2.6b and 2.7b
    U[4, nob+1, 2:] = 1. # UO2V
    U[5] = U[1] - U[2]   # UCO1CO2 where O is unoccupied
    U[6] = U[3] - U[4]   # UO1VO2V where O is occupied

    # CV + G + D + OO
    # Eqs. 2.4, 2.5, and 2.6c in JCP 2019.
    U[0] = numpy.ones((noa, nvb))
    if singlet:
        U[0, nob, 0]   = sqrt2
        U[0, nob+1, 1] = -sqrt2
    else: # triplet
        U[0, nob, 0]   = sqrt2
        U[0, nob, 1]   = 0.
        U[0, nob+1, 0] = 0.
        U[0, nob+1, 1] = sqrt2

    Ut = numpy.array((U[0], U[4], U[3], U[2], U[1], U[5], U[6])) # reorder U
    return U, Ut


# based on scf/_response_function.py
def _gen_rhf_response(mf, mo_coeff=None, mo_occ=None,
                      singlet=True, hermi=0, max_memory=None):
    '''Generate a function to compute the product of UHF response function and
    UHF density matrices.
    '''
    assert isinstance(mf, (rohf.ROHF, uhf.UHF))
    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if mo_occ is None: mo_occ = mf.mo_occ
    mol = mf.mol
    if isinstance(mf, hf.KohnShamDFT):
        from pyscf.pbc.dft import multigrid
        ni = mf._numint
        ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
        if mf.do_nlc():
            logger.warn(mf, 'NLC functional found in DFT object.  Its second '
                        'derivative is not available. Its contribution is '
                        'not included in the response function.')
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
        hybrid = ni.libxc.is_hybrid_xc(mf.xc)

        # mf might be pbc.dft.RKS object with multigrid
        if not hybrid and isinstance(getattr(mf, 'with_df', None), multigrid.MultiGridFFTDF):
            dm0 = mf.make_rdm1(mo_coeff, mo_occ)
            return multigrid._gen_rhf_response(mf, dm0, singlet, hermi)

        rho0, vxc, fxc = ni.cache_xc_kernel(mol, mf.grids, mf.xc,
                                            mo_coeff, mo_occ, 1)
        dm0 = None

        if max_memory is None:
            mem_now = lib.current_memory()[0]
            max_memory = max(2000, mf.max_memory*.8-mem_now)

        fxc *= .5
        def vind(dm1):
            if hermi == 2:
                v1 = numpy.zeros_like(dm1)
            else:
                # nr_rks_fxc_st requires alpha of dm1, dm1*.5 should be scaled
                v1 = ni.nr_rks_fxc_st(mol, mf.grids, mf.xc, dm0, dm1, 0, True,
                                   rho0, vxc, fxc, max_memory=max_memory)
            if hybrid:
                if omega == 0:
                    vj, vk = mf.get_jk(mol, dm1, hermi)
                    vk *= hyb
                elif alpha == 0: # LR=0, only SR exchange
                    vj = mf.get_j(mol, dm1, hermi)
                    vk = mf.get_k(mol, dm1, hermi, omega=-omega)
                    vk *= hyb
                elif hyb == 0: # SR=0, only LR exchange
                    vj = mf.get_j(mol, dm1, hermi)
                    vk = mf.get_k(mol, dm1, hermi, omega=omega)
                    vk *= alpha
                else: # SR and LR exchange with different ratios
                    vj, vk = mf.get_jk(mol, dm1, hermi)
                    vk *= hyb
                    vk += mf.get_k(mol, dm1, hermi, omega=omega) * (alpha-hyb)
                #if hermi != 2:
                #    v1 += vj - .5 * vk
                #else:
                #    v1 -= .5 * vk
            elif hermi != 2:
                v1 += mf.get_j(mol, dm1, hermi=hermi)
            #return v1
            return vj, -vk

    elif singlet and hermi != 2:
        def vind(dm1):
            vj, vk = mf.get_jk(mol, dm1, hermi=hermi)
            #v1 = vj - vk
            #return v1
            return vj, -vk

    else:
        def vind(dm1):
            return -mf.get_k(mol, dm1, hermi=hermi)

    return vind


# for MRSF adopted from tdscf/rhf.py
def gen_tda_operation(mf, fock_ao=None, singlet=True, wfnsym=None):
    '''A x

    Kwargs:
        wfnsym : int or str
            Point group symmetry irrep symbol or ID for excited CIS wavefunction.
    '''
    assert isinstance(mf, rohf.ROHF)

    mol = mf.mol
    mo_coeff = mf.mo_coeff
    assert (mo_coeff.dtype == numpy.double)
    mo_ea = mf.mo_energy.mo_ea
    mo_eb = mf.mo_energy.mo_eb
    mo_occ = mf.mo_occ
    nao, nmo = mo_coeff.shape
    occidxa = mo_occ > 0
    occidxb = mo_occ == 2
    viridxa = ~occidxa
    viridxb = ~occidxb
    orboa = mo_coeff[:,occidxa]
    orbob = mo_coeff[:,occidxb]
    orbva = mo_coeff[:,viridxa]
    orbvb = mo_coeff[:,viridxb]
    nocca = orboa.shape[1]
    noccb = orbob.shape[1]
    nvira = orbva.shape[1]
    nvirb = orbvb.shape[1]

    if wfnsym is not None and mol.symmetry:
        if isinstance(wfnsym, str):
            wfnsym = symm.irrep_name2id(mol.groupname, wfnsym)
        wfnsym = wfnsym % 10  # convert to D2h subgroup
        x_sym_a, x_sym_b = _get_x_sym_table(mf)
        sym_forbid = numpy.append(x_sym_a.ravel(), x_sym_b.ravel()) != wfnsym

    # beta virtual energy - alpha occupied energy
    fock_mo = lib.einsum('mp,imn,nq->ipq', mo_coeff.conj(), fock_ao, mo_coeff)
    e_ia = hdiag = numpy.diag(fock_mo[1])[viridxb] - numpy.diag(fock_mo[0])[occidxa,None]
    #e_ia = hdiag = mo_eb[viridxb] - mo_ea[occidxa,None]
    #print_matrix('e_ia:', e_ia, nind=1)
    if wfnsym is not None and mol.symmetry:
        hdiag[sym_forbid] = 0
    hdiag = e_ia.ravel()

    # it seems using pointers so don't move it above e_ia!!!
    # otherwise one needs to use numpy.copy()
    Fa_o, Fb_v = fock_mo[0, :nocca, :nocca], fock_mo[1, noccb:, noccb:]
    # remove diagonal elements for 1e cotributions in vind()
    Fa_o -= numpy.diag(numpy.diag(Fa_o))
    Fb_v -= numpy.diag(numpy.diag(Fb_v))

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, mf.max_memory*.8-mem_now)
    vresp = _gen_rhf_response(mf, hermi=0, max_memory=max_memory)

    # transform matrix
    sign = 1 if singlet else -1 # scale integrals Eq. 2.13
    U, Ut = mrsf_dimension_transform([nocca, noccb], [nvira, nvirb], nao,
                                     singlet=singlet)

    def vind(zs):
        nz = len(zs)
        zs = numpy.asarray(zs)
        if wfnsym is not None and mol.symmetry:
            zs = numpy.copy(zs)
            zs[:,sym_forbid] = 0

        # transform amplitudes
        zs = zs.reshape(nz,nocca,nvirb)
        zs_new = lib.einsum('kov,xov->kxov', Ut, zs)
        #if singlet:
        #    zs_new[0,:,noccb,0]  /= numpy.sqrt(2.)
        #    zs_new[0,:,noccb+1,1] = -zs_new[0,:,noccb,0]
        #else:
        #    zs_new[0,:,noccb,0]  /= numpy.sqrt(2.)
        #    zs_new[0,:,noccb,1] = zs_new[0,:,noccb+1,0] = 0.
        #    zs_new[0,:,noccb+1,1] = zs_new[0,:,noccb,0]
        zs_new[5,:,:, [0, 1]] = zs_new[5,:,:, [1, 0]] # swap columns of virtual O1 and O2 for UCO1CO2
        zs_new[6,:, [noccb, noccb+1]] = zs_new[6,:, [noccb+1, noccb]] # swap rows of occupied O1 and O2 for UO1VO2V
        dms = lib.einsum('kxov,pv,qo->kxpq', zs_new, orbvb, orboa.conj())
        dms[5:] = dms[5:].transpose(0,1,3,2) # use transpose for the last two dms
        dms = dms.reshape(-1, nao, nao)

        vj, vk = vresp(dms) # vk has included -1
        vj = vj.reshape((7, nz, nao, nao))
        vk = vk.reshape((7, nz, nao, nao))
        # Eq. 2.14
        vk[1:5] = sign * (vj[1:5] + vk[1:5])
        vk[6:] = -sign * vk[6:]
        v1ao = vk

        v1mo = lib.einsum('kxpq,qo,pv->kxov', v1ao, orboa, orbvb.conj())
        v1mo[5,:,:, [0, 1]] = v1mo[5,:,:, [1, 0]] # swap columns back
        v1mo[6,:, [noccb, noccb+1]] = v1mo[6,:, [noccb+1, noccb]] # swap rows back
        v1mo = lib.einsum('kov,kxov->xov', U, v1mo) # sum over the contributions
        #print_matrix('v1mo:', v1mo.reshape(nz,-1).T, nind=1)

        # add one-electron terms
        zs = zs_new[0]
        v1mo += lib.einsum('xia,ia->xia', zs, e_ia)
        v1mo += lib.einsum('xia,ab->xib', zs, Fb_v)
        v1mo -= lib.einsum('xia,ij->xja', zs, Fa_o)

        return v1mo.reshape(v1mo.shape[0],-1)

    return vind, hdiag
gen_tda_hop = gen_tda_operation


# based on tdscf/rhf.py
def analyze(tdobj, verbose=None):
    log = logger.new_logger(tdobj, verbose)
    mol = tdobj.mol
    mo_coeff = tdobj._scf.mo_coeff
    mo_occ = tdobj._scf.mo_occ
    nocca = numpy.count_nonzero(mo_occ > 0)
    noccb = numpy.count_nonzero(mo_occ == 2)

    e_ev = numpy.asarray(tdobj.e) * nist.HARTREE2EV
    e_wn = numpy.asarray(tdobj.e) * nist.HARTREE2WAVENUMBER
    wave_length = 1e7/e_wn

    if tdobj.singlet:
        log.note('\n** Singlet excitation energies and oscillator strengths **')
    else:
        log.note('\n** Triplet excitation energies and oscillator strengths **')

    if mol.symmetry:
        orbsym = hf_symm.get_orbsym(mol, mo_coeff)
        x_sym = symm.direct_prod(orbsym[mo_occ==2], orbsym[mo_occ==0], mol.groupname)
    else:
        x_sym = None

    f_oscillator = tdobj.oscillator_strength()
    for i, ei in enumerate(tdobj.e):
        x, y = tdobj.xy[i]
        if x_sym is None:
            log.note('Excited State %3d: %12.5f eV %9.2f nm  f=%.4f',
                     i+1, e_ev[i], wave_length[i], f_oscillator[i])
        else:
            wfnsym = _analyze_wfnsym(tdobj, x_sym, x)
            log.note('Excited State %3d: %4s %12.5f eV %9.2f nm  f=%.4f',
                     i+1, wfnsym, e_ev[i], wave_length[i], f_oscillator[i])

        if log.verbose >= logger.INFO:
            o_idx, v_idx = numpy.where(abs(x) > 0.1)
            for o, v in zip(o_idx, v_idx):
                log.info('    %4d -> %-4d %12.5f',
                         o+MO_BASE, v+MO_BASE+noccb, x[o,v])

    if log.verbose >= logger.INFO:
        log.info('\n** Transition electric dipole moments (AU) **')
        log.info('state          X           Y           Z        Dip. S.      Osc.')
        trans_dip = tdobj.transition_dipole()
        for i, ei in enumerate(tdobj.e):
            dip = trans_dip[i]
            log.info('%3d    %11.4f %11.4f %11.4f %11.4f %11.4f',
                     i+1, dip[0], dip[1], dip[2], numpy.dot(dip, dip),
                     f_oscillator[i])

        log.info('\n** Transition velocity dipole moments (imaginary part, AU) **')
        log.info('state          X           Y           Z        Dip. S.      Osc.')
        trans_v = tdobj.transition_velocity_dipole()
        f_v = tdobj.oscillator_strength(gauge='velocity', order=0)
        for i, ei in enumerate(tdobj.e):
            v = trans_v[i]
            log.info('%3d    %11.4f %11.4f %11.4f %11.4f %11.4f',
                     i+1, v[0], v[1], v[2], numpy.dot(v, v), f_v[i])

        log.info('\n** Transition magnetic dipole moments (imaginary part, AU) **')
        log.info('state          X           Y           Z')
        trans_m = tdobj.transition_magnetic_dipole()
        for i, ei in enumerate(tdobj.e):
            m = trans_m[i]
            log.info('%3d    %11.4f %11.4f %11.4f',
                     i+1, m[0], m[1], m[2])
    return tdobj


# based on tdscf/rhf.py
# change molecular orbitals
def _contract_multipole(tdobj, ints, hermi=True, xy=None):
    '''ints is the integral tensor of a spin-independent operator'''
    if xy is None: xy = tdobj.xy
    nstates = len(xy)
    pol_shape = ints.shape[:-2]
    nao = ints.shape[-1]

    if not tdobj.singlet:
        return numpy.zeros((nstates,) + pol_shape)

    mo_coeff = tdobj._scf.mo_coeff
    mo_occ = tdobj._scf.mo_occ
    orbo = mo_coeff[:,mo_occ>0]
    orbv = mo_coeff[:,mo_occ<2]

    #Incompatible to old numpy version
    #ints = numpy.einsum('...pq,pi,qj->...ij', ints, orbo, orbv.conj())
    ints = lib.einsum('xpq,pi,qj->xij', ints.reshape(-1,nao,nao), orbo, orbv.conj())
    pol = numpy.array([numpy.einsum('xij,ij->x', ints, x) * 2 for x,y in xy])
    if isinstance(xy[0][1], numpy.ndarray):
        if hermi:
            pol += [numpy.einsum('xij,ij->x', ints, y) * 2 for x,y in xy]
        else:  # anti-Hermitian
            pol -= [numpy.einsum('xij,ij->x', ints, y) * 2 for x,y in xy]
    pol = pol.reshape((nstates,)+pol_shape)
    return pol



class MRSF_TDA(tdscf.rks.TDA):

    singlet = True
    positive_eig_threshold = -0.3 # keep ground-state

    def gen_vind(self, mf=None):
        '''Generate function to compute Ax'''
        if mf is None:
            mf = self._scf
        fock_ao = mf.get_fock()
        fock_ao = numpy.array([fock_ao.focka, fock_ao.fockb])
        return gen_tda_hop(mf, fock_ao, singlet=self.singlet, wfnsym=self.wfnsym)

    def init_guess(self, mf, nstates=None, wfnsym=None, return_symmetry=False):
        if nstates is None: nstates = self.nstates
        if wfnsym is None: wfnsym = self.wfnsym

        mol = mf.mol
        mo_ea = mf.mo_energy.mo_ea
        mo_eb = mf.mo_energy.mo_eb
        mo_occ = mf.mo_occ
        occidxa = mo_occ > 0
        occidxb = mo_occ == 2
        viridxa = ~occidxa
        viridxb = ~occidxb
        e_ia = (mo_eb[viridxb] - mo_ea[occidxa,None]).ravel()
        nova = e_ia.size
        nstates = min(nstates, nova)

        # Find the nstates-th lowest energy gap
        e_threshold = numpy.partition(e_ia, nstates-1)[nstates-1]
        e_threshold += self.deg_eia_thresh

        #print_matrix('e_ia:', e_ia[e_ia<=e_threshold])
        idx = numpy.where(e_ia <= e_threshold)[0]
        x0 = numpy.zeros((idx.size, nova))
        for i, j in enumerate(idx):
            x0[i, j] = 1  # Koopmans' excitations

        if return_symmetry:
            if mf.mol.symmetry:
                x_sym = numpy.append(x_sym_a.ravel(), x_sym_b.ravel())
                x0sym = x_sym[idx]
            else:
                x0sym = None
            return x0, x0sym
        else:
            return x0

    def kernel(self, x0=None, nstates=None):
        '''TDA diagonalization solver
        '''
        cpu0 = (logger.process_clock(), logger.perf_counter())
        self.check_sanity()
        self.dump_flags()
        if nstates is None:
            nstates = self.nstates
        else:
            self.nstates = nstates
        log = logger.Logger(self.stdout, self.verbose)
        mol = self.mol

        vind, hdiag = self.gen_vind(self._scf)
        precond = self.get_precond(hdiag)

        def pickeig(w, v, nroots, envs):
            idx = numpy.where(w > self.positive_eig_threshold)[0]
            return w[idx], v[:,idx], idx

        x0sym = None
        if x0 is None:
            x0, x0sym = self.init_guess(
                self._scf, self.nstates, return_symmetry=True)
        elif mol.symmetry:
            x_sym_a, x_sym_b = _get_x_sym_table(self._scf)
            x_sym = numpy.append(x_sym_a.ravel(), x_sym_b.ravel())
            x0sym = [rhf._guess_wfnsym_id(self, x_sym, x) for x in x0]

        self.converged, self.e, x1 = lr_eigh(
            vind, x0, precond, tol_residual=self.conv_tol, lindep=self.lindep,
            nroots=nstates, x0sym=x0sym, pick=pickeig, max_cycle=self.max_cycle,
            max_memory=self.max_memory, verbose=log)

        nmo = self._scf.mo_occ.size
        nocca = (self._scf.mo_occ>0).sum()
        noccb = (self._scf.mo_occ==2).sum()
        nvira = nmo - nocca
        nvirb = nmo - noccb
        self.xy = [(xi.reshape(nocca,nvirb),  # X_alpha_to_beta
                    0)  # (Y_alpha_to_beta
                   for xi in x1]

        print_matrix('amplitudes:', x1.reshape(-1,nocca,nvirb).transpose(0,2,1), nind=1)
        if self.chkfile:
            lib.chkfile.save(self.chkfile, 'tddft/e', self.e)
            lib.chkfile.save(self.chkfile, 'tddft/xy', self.xy)

        log.timer('TDA', *cpu0)
        self._finalize()
        return self.e, self.xy

    analyze = analyze
    _contract_multipole = _contract_multipole

    to_gpu = lib.to_gpu

MRSF_CIS = MRSF_TDA



if __name__ == '__main__':
    # ethylene tw-pyr CI
    #atom = """
    #C       -0.001716     0.029274    -0.037218
    #C        1.378613     0.007268    -0.008058
    #H        2.012945    -0.887831     0.020480
    #H        0.055158    -0.660513    -0.949800
    #H       -0.481163    -0.736532     0.584588
    #H        1.974826     0.910467    -0.129649
    #"""

    atom = """Be      0.000000      0.000000      0.000000"""

    spin = 2
    basis = '6-31g'
    functional = 'hf' #'bhandhlyp'
    nstates = 3

    mol = gto.M(
            atom = atom,
            spin = spin,
            basis = basis,
            )

    mf = scf.ROKS(mol)
    mf.xc = functional
    e0 = mf.kernel()

    td = MRSF_TDA(mf)
    td.nstates = nstates
    td.verbose = 4
    e, xys = td.kernel()
    td.analyze()
