from lumeq import np
from lumeq.utils import print_matrix, convert_units
from lumeq.plot import plt

from lumeq.dynamics.oscillator_dynamics import get_boltzmann_beta

def bose_einstein_distribution(energy, chemical_potential=0., temperature=300, unit='eh'):
    energy = energy - chemical_potential
    if unit not in ['eh', 'hartree']:
        energy = convert_units(energy, unit, 'eh')

    beta = get_boltzmann_beta(temperature)
    occupation = 1. / (np.exp(beta*energy) - 1.)
    #print_matrix('occupation:', occupation)
    return np.where(np.abs(occupation)>1e3, 0., occupation)


if __name__ == '__main__':
    fig_name = 'boson_occupation.png'

    potential = 0.
    energy = np.arange(.001, 2, .01)
    n = len(energy)

    temperature = [50, 100, 200, 300]
    m = len(temperature)

    fig, axs = plt.subplots(1, m, figsize=(8, 3), dpi=300, constrained_layout=True)

    number = np.zeros((m, n))
    for j, temp in enumerate(temperature):
        number[j] = bose_einstein_distribution(energy, potential, temp, unit='mev')

        axs[j].plot(energy, number[j], label=str(temp)+'K')

        axs[j].set_xlabel('Energy (meV)')
        if j==0:
            axs[j].set_ylabel('Occupation Number')

        axs[j].legend()

    plt.savefig(fig_name)
