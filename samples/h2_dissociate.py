from wavefunction_analysis import sys, np
from wavefunction_analysis.opt.mrsf_roks import MRSF_TDA
from wavefunction_analysis.plot import plt
from wavefunction_analysis.utils import convert_units, print_matrix

from pyscf import gto, scf


if __name__ == "__main__":
    bond = [0.5, 3.0]  # Bond length range in Angstroms
    bond_length = np.linspace(bond[0], bond[1], int(bond[1]-bond[0])*20+1)
    functional = 'hf'#'bhandhlyp'
    basis = '6-311++g**'
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

        td.singlet = False
        e, xys = td.kernel()

        energy[1][i,1, :len(e)] = e+e0

    # Plotting
    plt.figure(figsize=(12, 6))
    for i in range(nstates-1, -1, -1):
        plt.plot(bond_length, energy[1][:, 1, i], '--', label=f'$T_{i+1}$')
        plt.plot(bond_length, energy[1][:, 0, i], label=f'$S_{i+1}$')
    plt.plot(bond_length, energy[0], label='$T_{Ref}$')

    plt.title('H2 Dissociation Curve using MRSF-TDA')
    plt.xlabel('Bond Length (Angstrom)')
    plt.ylabel('Excitation Energy (Hartree)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('h2_dissociation_curve_mrsf_tda.png', dpi=300)
    plt.show()
