from wavefunction_analysis.utils.wick_contraction import sqo_evaluation
from wavefunction_analysis.utils.wick_contraction import commutator
from wavefunction_analysis import itertools

if __name__ == '__main__':
    latex = True
    colors = ['red', 'blue', 'wickgreen', 'black', 'wickorange', 'purple']

    # 1e and 2e Hamiltonian terms
    h1 = 'p_sigma^dagger q_tau'
    h2 = 'p_sigma^dagger r_varsigma^dagger ell_kappa q_tau'

    # open-shell excitation operators
    Tsa = 's_alpha^dagger a_alpha' # bra side
    Tbt = 'b_alpha^dagger t_alpha' # ket side
    exceptions = [tuple(Tsa.split()), tuple(Tbt.split())]

    for h in [h1, h2]:
        term_type = '1e' if h == h1 else '2e'
        title = f'Open-shell excited-state {term_type} term contractions:'
        sqo_evaluation(Tsa, h, Tbt, exceptions=exceptions, title=title,
                       colors=colors, latex=latex)

    # spin-flip excitation operators diagonal terms
    Tst_mr = ['s_beta^dagger t_alpha', 's_alpha^dagger t_beta'] # Ms=+1,-1 reference bra
    Tts_mr = ['t_alpha^dagger s_beta', 't_beta^dagger s_alpha'] # Ms=+1,-1 reference ket
    Tia_mr = ['i_alpha^dagger a_beta', 'i_beta^dagger a_alpha'] # bra side sf excitation
    Tbj_mr = ['b_beta^dagger j_alpha', 'b_alpha^dagger j_beta'] # ket side sf excitation

    ijk = []
    for (i, j, k, l) in itertools.product(range(2), repeat=4):
        ijk.append((i, j, k, l))

    for loop, (Tst, Tts, Tia, Tbj) in enumerate(itertools.product(Tst_mr, Tts_mr, Tia_mr, Tbj_mr)):
        print('\nloop:', loop+1, ijk[loop])
        exceptions = [tuple(Tst.split()), tuple(Tts.split()), tuple(Tia.split()), tuple(Tbj.split())]

        for h in [h1, h2]:
            term_type = 'metric' if h == '' else ('1e' if h == h1 else '2e')
            title = f'Spin-flip excited-state {term_type} term contractions:'
            operators, factors = commutator(Tia, h, Tbj)
            for i, operator in enumerate(operators):
                l, m, r = operator
                bra = ' '.join([Tst, l])
                ket = ' '.join([r, Tts])
                print(f'Operator {i+1} with factor {factors[i]}:')
                print('bra:', bra, '\nm:', m, '\nket:', ket)
                sqo_evaluation(bra, m, ket, exceptions=exceptions, title=title,
                               hamiltonian=h, latex=latex, diagram=False)

            print('')
