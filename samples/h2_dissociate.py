from lumeq import sys, np
from lumeq.opt.mrsf_roks import MRSF_TDA
from lumeq.plot import plt
from lumeq.utils import convert_units, print_matrix

from pyscf import gto, scf

from SF_CIS import do_SF_UCIS

if __name__ == "__main__":
    bond = [0.5, 3.0]  # Bond length range in Angstroms
    bond_length = np.linspace(bond[0], bond[1], int(bond[1]-bond[0])*20+1)
    functional = 'hf'#'bhandhlyp'
    basis = '6-31g'#'6-311++g**'
    nstates = 5

    atom = "H 0 0 -{0}; H 0 0 {0}"
    mol = gto.M(
            atom = atom.format(bond_length[0]),
            basis = basis,
            charge = 0,
            spin = 2,
            verbose = 0,
            )

    dm0 = None

    energy = [[], np.zeros((len(bond_length), 2, nstates))]
    for i, r in enumerate(bond_length):
        mol.set_geom_(atom.format(r/2), unit='Angstrom')

        mf = scf.ROKS(mol)
        mf.xc = functional
        e0 = mf.kernel(dm0=dm0)
        #dm0 = mf.make_rdm1()

        td = MRSF_TDA(mf)
        td.nstates = nstates
        td.verbose = 0
        td.conv_tol = 1e-7
        e, xys = td.kernel()
        #td.analyze()
        energy[0].append(e0)
        energy[1][i,0, :len(e)] = e+e0
        if r == 0.75:
            print("Debugging at bond length:", r)
            print('e:', e0, e+e0)
            fock_ao = mf.get_fock()
            fock_ao = np.array([fock_ao.focka, fock_ao.fockb])
            mf.mo_energy = np.array([mf.mo_energy.mo_ea, mf.mo_energy.mo_eb])
            mf.mo_coeff = np.array([mf.mo_coeff, mf.mo_coeff])
            fock = np.einsum('imp,imn,inq->ipq', mf.mo_coeff, fock_ao, mf.mo_coeff)
            print_matrix('fock:', fock)
            mf.mo_energy = fock
            nocca, noccb, nmo = 2, 0, mf.mo_coeff[0].shape[1]
            esf, vsf = do_SF_UCIS(mol, mf, nocca, noccb, nmo)
            print('roks sf-e:', e0, esf+e0)

            mf = scf.UKS(mol)
            mf.xc = functional
            e0 = mf.kernel()
            fock_ao = mf.get_fock()
            fock = np.einsum('imp,imn,inq->ipq', mf.mo_coeff, fock_ao, mf.mo_coeff)
            print_matrix('fock:', fock)
            mf.mo_energy = fock
            esf, vsf = do_SF_UCIS(mol, mf, nocca, noccb, nmo)
            print('uks sf-e:', e0, esf+e0)


        td.singlet = False
        e, xys = td.kernel()

        energy[1][i,1, :len(e)] = e+e0

    # Plotting
    plt.figure(figsize=(12, 6))
    for i in range(nstates-1, -1, -1):
        plt.plot(bond_length, energy[1][:, 1, i], '--', label=f'$T_{i+1}$')
        plt.plot(bond_length, energy[1][:, 0, i], label=f'$S_{i+1}$')
    plt.plot(bond_length, energy[0], label='$T_{Ref}$')

    plt.title(f'H2 Dissociation from MRSF-TDA/{functional}/{basis.upper()}')
    plt.xlabel('Bond Length (Angstrom)')
    plt.ylabel('Excitation Energy (Hartree)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('h2_dissociation_curve_mrsf_tda2.png', dpi=300)
    plt.show()
