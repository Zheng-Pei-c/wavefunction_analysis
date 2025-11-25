from wavefunction_analysis.utils.wick_contraction import sqo_evaluation

if __name__ == '__main__':
    latex = True

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
        sqo_evaluation(Tsa, h, Tbt, exceptions=exceptions, title=title, latex=latex)

    # spin-flip excitation operators diagonal terms
    Tst = 's_beta^dagger t_alpha' # Ms=1 reference bra
    Tts = 't_alpha^dagger s_beta' # Ms=1 reference ket
    Tia = 'i_alpha^dagger a_beta' # bra side sf excitation
    Tbj = 'b_beta^dagger j_alpha' # ket side sf excitation
    bra = [Tst, Tia]  # Ms=1 reference bra
    ket = [Tbj, Tts]  # Ms=1 reference ket
    exceptions = [tuple(Tst.split()), tuple(Tts.split()), tuple(Tia.split()), tuple(Tbj.split())]

    for h in ['', h1, h2]:
        term_type = 'metric' if h == '' else ('1e' if h == h1 else '2e')
        title = f'Spin-flip excited-state {term_type} term contractions:'
        sqo_evaluation(bra, h, ket, exceptions=exceptions, title=title, latex=latex)
