import numpy as np

from wavefunction_analysis.plot import plt

def self_energy_v1(x, coupling, center, width, itype='imaginary', appro=False):
    """
    this function assumes:
    1. same coupling strengths for all the states
    2. disordered state energy spectrum following Gaussian distribution
    """
    from numpy import sqrt, pi, exp

    v = coupling**2 / width
    x = (x - center) / (sqrt(2.) * width)

    if itype == 'imaginary' or itype == 'both':
        sigma_i = (- sqrt(pi/2.) * v) * exp(-x**2)

        if itype == 'imaginary':
            return sigma_i

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

        sigma_r = (- sqrt(2.) * v) * dawsn(x)

        if itype == 'real':
            return sigma_r

    if itype == 'both':
        return sigma_r, sigma_i
    else:
        raise ValueError('itype has to be imaginary, real, or both')


def greens_function(f):
    pass


def plot_sigma(ax, x, coupling, omegac, omega0, width, axis_label):
    #print('V/$\sigma$:', coupling, width, coupling/width)
    sigma = self_energy_v1(x, coupling, omega0, width, itype='both')

    ax.plot(x, -sigma[0], label=r'$\Sigma_R$')
    ax.plot(x, -sigma[1], label=r'$\Sigma_I$')

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
    fig_name = 'sigma'

    omega0 = 0.
    N = 1500

    fig = plt.figure(figsize=(21, 9), dpi=300)
    axs = fig.subplots(4, 6)

    i, j = 0, 0
    for omegac in [0., .5]:
        ls = omegac if omegac < 0. else 0.
        rs = omegac if omegac > 0. else 0.
        x = np.linspace(-.5+ls, .5+rs, 41)

        for coupling in [3e-3, 6e-3, 1e-2, 2e-2]:
            coupling *= np.sqrt(N)

            for width in [0.05, .1, .2]: # Gaussian width

                xlabel = True if i == 5 else False
                ylabel = True if j == 0 else False
                plot_sigma(axs[i,j], x, coupling, omegac, omega0, width, [xlabel, ylabel])

                #print('i:', i, 'j:', j)
                j += 1
                if j%6 == 0:
                    j = 0
                    i += 1

    plt.tight_layout()

    if fig_name:
        plt.savefig(fig_name)
    else:
        plt.show()

