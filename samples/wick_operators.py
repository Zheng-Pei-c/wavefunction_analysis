from lumeq.utils.wick_contraction import sqo_evaluation
from lumeq.utils.wick_contraction import commutator
from lumeq import itertools

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

    Xpj = 'p_sigma^dagger j_alpha'
    Xbq = 'b_beta^dagger q_tau'
    Xiq = 'i_alpha^dagger q_tau'
    Xpa = 'p_sigma^dagger a_beta'

    for h in [h1]:
        term_type = 'metric' if h == '' else ('1e' if h == h1 else '2e')
        title = f'Spin-flip excited-state {term_type} term contractions:'

        Tst = Tst_mr[0]
        Tts = Tts_mr[0]
        Tia = Tia_mr[0]
        Tbj = Tbj_mr[0]
        for i, X in enumerate([Xpj, Xbq]):
            print('i:', i+1)
            middle = Tia + ' ' + X
            exceptions = [tuple(Tst.split()), tuple(Tts.split()), tuple(Tia.split()), tuple(X.split())]
            sqo_evaluation(Tst, middle, Tts, exceptions=exceptions, title=title,
                           hamiltonian=h, latex=latex, diagram=False)

            print('')
        for i, X in enumerate([Xiq, Xpa]):
            print('i:', i+1)
            middle = X + ' ' + Tbj
            exceptions = [tuple(Tst.split()), tuple(Tts.split()), tuple(Tbj.split()), tuple(X.split())]
            sqo_evaluation(Tst, middle, Tts, exceptions=exceptions, title=title,
                           hamiltonian=h, latex=latex, diagram=False)

            print('')
