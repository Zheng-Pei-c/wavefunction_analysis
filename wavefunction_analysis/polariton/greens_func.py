import numpy as np

from wavefunction_analysis.utils import print_matrix
from wavefunction_analysis.plot import plt

"""
This Green's function approach follows:
 10.1063/5.0086027 (Gaussian distribution)
 10.1103/PhysRevB.105.064205 (Lorentzian distribution)
"""

def greens_func_e0_gauss(x, center, width, scaling=1., itype='both', appro=False):
    """
    Greens function of the unperturbed exciton states
    disordered state energy spectrum has Gaussian distribution
    """
    from numpy import sqrt, pi, exp

    v = scaling / width
    x = (x - center) / (sqrt(2.) * width)

    if itype == 'imaginary' or itype == 'both':
        g_i = (- sqrt(pi/2.) * v) * exp(-x**2)

        if itype == 'imaginary':
            return g_i

    if itype == 'real' or itype == 'both':
        if appro:
            def dawsn(x, appro):
                y = 1. / (2. * x) # first-order
                if appro > 1:
                    x2 = x**2
                    y += 1. / (4. * x * x2) # second-order
                if appro > 2:
                    y += 3. / (8. * x * x2**2) # third-order
        else: # exact dawson integral
            from scipy.special import dawsn

        g_r = (sqrt(2.) * v) * dawsn(x)

        if itype == 'real':
            return g_r

    if itype == 'both':
        return np.array([g_r, g_i])
    else:
        raise ValueError('itype has to be imaginary, real, or both')


def greens_func_e0_lorentz(x, center, width, eta=1e-4, scaling=1., itype='both'):
    """
    Greens function of the unperturbed exciton states
    disordered state energy spectrum has Lorentzian distribution
    # eta is a small phenomenological parameter
    """
    x = x - center
    eta = eta + width
    tot = x**2 + eta**2

    if itype == 'imaginary' or itype == 'both':
        g_i = -scaling * eta / tot

        if itype == 'imaginary':
            return g_i

    if itype == 'real' or itype == 'both':
        g_r = scaling * x / tot

        if itype == 'real':
            return g_r

    if itype == 'both':
        return np.array([g_r, g_i])
    else:
        raise ValueError('itype has to be imaginary, real, or both')


def self_energy_same_coupling(x, coupling, center, width, eta=1e-4, method='gauss-0',
                              itype='both'):
    """
    this function assumes:
    1. same coupling strengths for all the states
    2. disordered state energy spectrum following Gaussian or Lorentzian distribution
    """
    scaling = coupling**2

    if 'gauss' in method:
        appro = int(method[-1])
        return greens_func_e0_gauss(x, center, width, scaling, itype, appro)
    elif 'lorentz' in method:
        return greens_func_e0_lorentz(x, center, width, eta, scaling, itype)


def greens_func_p(x, sigma, omega, eta=1e-4, itype='both'):
    """
    Greens function of the polariton states
    # sigma[0] and sigma[1] are the real and imaginary parts, respectively
    # omega is the photon energy
    # eta is a small phenomenological parameter
    """

    # has same functionility as greens_func_e0_lorentz
    return greens_func_e0_lorentz(x-sigma[0], omega, -sigma[1], eta, itype=itype)


def density_of_states_p(x, sigma, omega, eta=1e-4):
    """
    density of states of the polariton states
    # sigma[0] and sigma[1] are the real and imaginary parts, respectively
    # omega is the photon energy
    # eta is a small phenomenological parameter
    """
    dos = greens_func_p(x, sigma, omega, eta, 'imaginary')
    return dos / (-np.pi)


def greens_func_e(g0, gc, coupling, itype='both'):
    """
    Greens function of exciton states
    """
    coupling = coupling**2

    g0_r, g0_i = g0
    gc_r, gc_i = gc

    tmp1 = g0_r**2 - g0_i**2
    tmp2 = 2. * g0_r * g0_i

    if itype == 'imaginary' or itype == 'both':
        g_i = gc_i * tmp1 + g0_r * tmp2
        g_i = g_i * coupling + g0_i

        if itype == 'imaginary':
            return g_i

    if itype == 'real' or itype == 'both':
        g_r = gc_r * tmp1 - g0_i * tmp2
        g_r = g_r * coupling + g0_r

        if itype == 'real':
            return g_r

    if itype == 'both':
        return np.array([g_r, g_i])
    else:
        raise ValueError('itype has to be imaginary, real, or both')


def density_of_states_e(g0, gc, coupling):
    """
    density of states of the polariton states
    # sigma[0] and sigma[1] are the real and imaginary parts, respectively
    # omega is the photon energy
    # eta is a small phenomenological parameter
    """
    dos = greens_func_e(g0, gc, coupling, 'imaginary')
    return dos / (-np.pi)


def absorption_spectra(x, coupling, dos):
    """
    assume same coupling
    """
    return (-coupling**2) * x * dos


def plot_sigma(ax, x, coupling, omegac, eta, omega0, width, method, axis_label):
    #print('V/$\sigma$:', coupling, width, coupling/width)
    g0 = self_energy_same_coupling(x, 1., omega0, width, eta, method)
    sigma = coupling**2 * g0
    gc = greens_func_p(x, sigma, omegac, eta)
    gm = greens_func_e(g0, gc, coupling)

    dos_c, dos_e = gc[1]/(-np.pi), gm[1]/(-np.pi)
    absorp = absorption_spectra(x, coupling, dos_e)

    ax.plot(x, sigma[0], label=r'$\Sigma_R$')
    ax.plot(x, -sigma[1], label=r'-$\Sigma_I$')
    ax.plot(x, dos_c/np.max(dos_c), label=r'$\rho_c$')
    ax.plot(x, dos_e/np.max(dos_e), label=r'$\rho_e$')
    ax.plot(x, absorp/np.max(absorp), label=r'$\alpha$')

    ax.plot(x, x-omegac, color='black', ls='--', lw=.7, label=r'$\omega-\omega_c$')

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.hlines(0., xmin=xmin, xmax=xmax, color='gray', ls='--', lw=.5)
    ax.vlines(omega0, ymin=ymin, ymax=ymax, color='gray', ls='--', lw=.5)

    if axis_label[0]:
        ax.set_xlabel(r'$\omega$ (eV)')
    if axis_label[1]:
        ax.set_ylabel(r'Self-Energy $\Sigma$ (eV)')
    ax.set_title(r'$\omega_c$:%2.1f, V:%2.1f, $\sigma$:%3.2f, V/$\sigma$:%2.1f' % (omegac, coupling, width, coupling/width))

    ax.legend()


if __name__ == '__main__':
    #fig_name = 'sigma'
    fig_name = 'dos'

    #method = 'gauss-0'
    method = 'lorentz'

    eta = 1e-4
    omega0 = 0.
    N = 1500

    fig = plt.figure(figsize=(21, 9), dpi=300)
    axs = fig.subplots(4, 3)

    i, j = 0, 0
    for omegac in [0.]:
        delta = omegac - omega0
        ls = delta if delta < 0. else 0.
        rs = delta if delta > 0. else 0.
        x = np.linspace(omega0-.5+ls, omega0+.5+rs, 41)

        for coupling in [3e-3, 6e-3, 1e-2, 2e-2]:
            coupling *= np.sqrt(N)

            for width in [0.05, .1, .2]: # Gaussian width

                xlabel = True if i == 4 else False
                ylabel = True if j == 0 else False
                plot_sigma(axs[i,j], x, coupling, omegac, eta, omega0, width, method, [xlabel, ylabel])

                #print('i:', i, 'j:', j)
                j += 1
                if j%3 == 0:
                    j = 0
                    i += 1

    plt.tight_layout()

    if fig_name:
        plt.savefig(fig_name+'_'+method)
    else:
        plt.show()
