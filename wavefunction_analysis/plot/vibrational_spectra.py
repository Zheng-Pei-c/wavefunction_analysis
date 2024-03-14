import sys
import numpy as np

from pyscf import lib
from pyscf.lib import logger
from pyscf.dft.rks import RKS
from pyscf.hessian import thermo

from wavefunction_analysis.utils.pyscf_parser import *
from wavefunction_analysis.utils import convert_units, print_matrix

import matplotlib.mlab as mlab
from wavefunction_analysis.plot import plt

def broadening(heighs, cen, wid=0.0005, d=0.05, method='gaussian', min_e=0):
    mi, ma = np.min(cen), np.max(cen)
    if min_e > 0.: mi = min_e
    x = np.arange(mi-d, ma+d, (ma-mi)/1001)
    y = 0.
    if method == 'lorentzian':
        for n in range(len(cen)):
            y += heighs[n] * wid**2 / ((x-cen[n])**2 + wid**2)
    elif method == 'gaussian':
        for n in range(len(cen)):
            y += np.sqrt(2*np.pi) * wid * heighs[n] * norm.pdf(x, cen[n], wid)
    return x, y


def fit_val(positions, heighs, broaden):
    num_vib = len(positions)
    x_min = positions[0] - 50
    if x_min < 0:
        x_min = 0
    x_max = positions[num_vib-1] + 50
    if x_max > 4000:
        x_max = 4000
    ix = np.linspace(x_min, x_max, int(x_max-x_min))

    iy = 0.0
    for peak in range(0, num_vib):
        iy += 2.51225 * broaden * heighs[peak] \
             * mlab.normpdf(ix,positions[peak],broaden)
    return (ix, iy)


def plot_spectra(peak_centers, peak_intens, broaden, fig_name):
    ix, iy = fit_val(peak_centers, peak_intens, broaden)
    plt.plot(ix, iy)
    plt.xlim(800,3600)
    plt.ylim(0, 1200)
    plt.xticks([1000,1500,2000,2500,3000,3500], size=14)
    plt.yticks([0,400,800,1200], size=14)
    plt.xlabel("Frequency (cm$^{-1}$)",fontsize=16)
    plt.ylabel("Intensity",fontsize=16)
    plt.tight_layout()
    plt.savefig(fig_name)


def infrared(dip_dev, normal_mode):
    factor = 974.8802478
    if normal_mode.ndim == 3:
        # first index is mode
        normal_mode = normal_mode.reshape(-1, dip_dev.shape[0])

    trans_dip = np.einsum('qx,iq->ix', dip_dev, normal_mode)
    sir = np.einsum('ix,ix->i', trans_dip, trans_dip)
    return sir * factor


def get_dipole_dev(mf, hessobj, origin=None):
    if origin is None:
        if hasattr(mf, 'origin'): origin = mf.origin
        else: origin = np.zeros(3)

    mol = mf.mol
    natm = mol.natm
    atmlst = range(mol.natm)

    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    mocc = mo_coeff[:,mo_occ>0]
    mo_energy = mf.mo_energy
    max_memory = 4000
    nao = mo_coeff.shape[0]

    dipole = mol.intor('int1e_r', comp=3, hermi=0)
    dm = mf.make_rdm1()

    dipole_d1 = mol.intor('int1e_irp', comp=9, hermi=0).reshape(3,3,nao,nao).transpose(1,0,3,2)

    log = lib.logger.new_logger(hessobj, None)
    h1ao = hessobj.make_h1(mo_coeff, mo_occ, hessobj.chkfile, atmlst, log)
    mo1, mo_e1 = hessobj.solve_mo1(mo_energy, mo_coeff, mo_occ, h1ao,
                                   None, atmlst, max_memory, log)
    mo1 = lib.chkfile.load(mo1, 'scf_mo1')
    mo1 = {int(k): mo1[k] for k in mo1}

    g1 = np.zeros((natm, 3, 3))
    aoslices = mol.aoslice_by_atom()
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia,2:]
        dm1 = np.einsum('xpi,qi->xpq', mo1[ia], mocc)

        g1[k] += np.einsum('ypq,xqp->xy', dipole, dm1)*2.
        g1[k] -= np.einsum('xypq,qp->xy', dipole_d1[:,:,p0:p1], dm[:,p0:p1]) # dipole_d1 has extra minus

    g1 *= -2. # electron dipole is negative

    z = mol.atom_charges()
    for i in range(natm): # nuclear dipole derivative
        g1[i] += np.eye(3)*z[i]

    return g1.reshape(natm*3, 3)



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
    co2 = """
            O    0. 0. 0.386715
            C    0. 0. 1.550000
            O    0. 0. 2.713285
    """
    atom = locals()[sys.argv[1]] if len(sys.argv) > 1 else hf

    functional = 'hf'
    mol = build_single_molecule(0, 0, atom, 'sto-3g')

    mf = RKS(mol) # in coherent state
    mf.xc = functional
    mf.grids.prune = True

    e_tot = mf.kernel()
    hessobj = mf.Hessian()
    h = hessobj.kernel()
    print_matrix('hess:', h, 5, 1)

    results = thermo.harmonic_analysis(mol, h) # only molecular block
    print('freq_au:', results['freq_au'])
    print('freq_wavenumber:', results['freq_wavenumber'])
    print('force_const_dyne:', results['force_const_dyne'])
    print('reduced_mass:', results['reduced_mass'])

    dip_dev = get_dipole_dev(mf, hessobj)
    print_matrix('dip_dev:', dip_dev)
    sir = infrared(dip_dev, results['norm_mode'])
    print('infrared intensity:', sir)
