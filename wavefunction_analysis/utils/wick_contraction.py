from wavefunction_analysis import sys, np, itertools
from wavefunction_analysis.spins import sympy

import unicodedata
from collections import defaultdict

"""
Generate Wick contraction pairs and patterns from Fermionic second-quantization operator strings.
The operator here has the form like 'a^', 'i', 'b†', 'j_α', 'c_b^', 'k_beta',
representing creation and annihilation operators (with spin labels).
Remember to leave the dagger symbol at the end of the operator string.
"""

def is_creator(operator):
    """Check if the operator is a creation operator by dagger symbol."""
    return operator.endswith(('^', '†', '^dagger'))


def is_annihilator(operator):
    """Check if the operator is an annihilation operator."""
    return not is_creator(operator)


def remove_dagger(operator):
    """Remove the dagger symbol from the operator string."""
    return operator.split('^')[0].split('†')[0]


def get_orbital_index(operator):
    """Extract the orbital index from the operator string."""
    return operator[0]


def get_spin_label(operator):
    """Extract the spin label from the operator string."""
    if '_' in operator:
        spin = remove_dagger(operator.split('_')[1])
        if spin in ('a', 'alpha', 'up', 'α'):
            return 'alpha'
        elif spin in ('b', 'beta', 'down', 'β'):
            return 'beta'
        else:
            return spin
    return None


def is_same_pattern(contraction1, contraction2):
    """Check if two contraction patterns are the same, ignoring order."""
    set1 = {frozenset(pair) for pair in contraction1}
    set2 = {frozenset(pair) for pair in contraction2}
    return set1 == set2


def has_pattern(contraction_list, target_pattern):
    """Check if a target contraction pattern exists in a list of contraction patterns.

    Parameters
        contraction_list : list of contraction patterns (each pattern is a list of pairs)
        target_pattern : a contraction pattern (list of pairs) to search for

    Returns
        exists : True if target_pattern exists in contraction_list, False otherwise
    """
    for pattern in contraction_list:
        if is_same_pattern(pattern, target_pattern):
            return True
    return False


def get_list(operators):
    """
    Convert input operators string to a list.

    Parameters
        operators : list or str separated by spaces or commas

    Returns
        operators_list : list of operator strings
    """
    if isinstance(operators, str):
        if ',' in operators:
            operators = operators.replace(',', ' ')
        operators = operators.split()
    return operators


def wick_pairs(operators, exceptions=[], index=False):
    """
    Pick all the possible Wick contraction pairs from a string of creation and annihilation operators.

    Parameters
        operators : list or str separated by spaces or commas
            A list of strings representing creation and annihilation operators with optional spin labels.
            Eg. ['a^', 'i', 'b†', 'j_α', 'c_b^', 'k_beta']
                 or "a^ i b† j_α c_b^ k_beta"
        exceptions : (list of) tuple pairs of operators that should not be contracted
        index : if True, return the indices of the operators instead of the operator strings

    Returns
        pairs : a list of the possible contraction pairs of the operators
    """
    known_spins = ['alpha', 'beta', None] # known spin labels
    operators = get_list(operators)

    if isinstance(exceptions, tuple):
        exceptions = [exceptions]

    n = len(operators)
    if n % 2 != 0:
        raise ValueError("The number of operators must be even for Wick contraction.")

    creators = np.array([is_creator(op) for op in operators], dtype=bool)
    annihilators = ~creators
    spins = [get_spin_label(op) for op in operators]

    pairs = []

    for i in range(n):
        op_i = operators[i]
        cr_i = creators[i]
        an_i = annihilators[i]
        sp_i = spins[i]

        for j in range(i + 1, n):
            op_j = operators[j]
            cr_j = creators[j]
            an_j = annihilators[j]
            sp_j = spins[j]

            # exclude exception pairs of operators
            skip = [(op_i, op_j) == exc or (op_j, op_i) == exc for exc in exceptions]
            if np.any(skip):
                continue

            if (is_creator(op_i) and is_annihilator(op_j)) or (is_annihilator(op_i) and is_creator(op_j)):
                if sp_i in known_spins and sp_j in known_spins:
                    if sp_i == sp_j: # both known spins and should match
                        pairs.append((i, j))
                else:
                    pairs.append((i, j))

    if not index: # convert indices back to operator strings
        pairs = index_to_operators(operators, pairs)

    return pairs


def index_to_operators(operators, pairs_index):
    """
    Convert pairs of operator indices to operator strings.

    Parameters
        operators : list of operator strings
        pairs_index : list of tuple pairs of indices

    Returns
        pairs : list of tuple pairs of operator strings
    """
    pairs = [(operators[i], operators[j]) for (i, j) in pairs_index]
    return pairs


def wick_contraction(operators, pairs, expand=True):
    """
    Perform Wick contraction on the given pairs of operators.

    Parameters
        operators : list of operator strings
        pairs : list of tuple pairs of indices to be contracted
        expand : True return product patterns;
                 False return dict of lists of contraction patterns

    Returns
        contractions : list of contraction patterns in the form of strings
    """
    operators = get_list(operators)
    n_op = len(operators) # total number of operators
    # pairs of operator strings or indices
    is_index = True if isinstance(pairs[0][0], int) else False

    contractions = defaultdict(list) # set up empty lists for keys not in dict
    for (op1, op2) in pairs:
        contractions[op1].append(op2)

    if expand:
        key_list = list(contractions.keys())
        value_list = [contractions[k] for k in key_list]
        n_keys = len(key_list)
        dim = [len(v) for v in value_list]

        contractions = []
        for idx in itertools.product(*[range(d) for d in dim]):
            #c_list = [(key_list[i], value_list[i][idx[i]]) for i in range(n_keys)]
            c_list, c_flat = [], []
            for i in range(n_keys):
                a, b = key_list[i], value_list[i][idx[i]]
                if a not in c_flat and b not in c_flat:
                    c_flat += [a, b]
                    if is_index:
                        c_list.append((a, b, operators[a], operators[b]))
                    else:
                        c_list.append((a, b)) # operator strings

            if len(c_flat) == n_op and not has_pattern(contractions, c_list):
                #print(c_list)
                contractions.append(c_list)

    return contractions[::-1] # reverse the order for convenience


def wick_delta(contractions, latex=False):
    """
    Convert contraction pairs into delta functions.

    Parameters
        contractions : list of contraction patterns in the form of strings
        latex : if True, format the output of delta strings for LaTeX rendering

    Returns
        deltas : list of delta function strings representing the contractions
    """
    if isinstance(contractions[0], list): # loop over multiple contraction patterns
        return [wick_delta(pair, latex=latex) for pair in contractions]

    if len(contractions[0]) == 2:
        raise ValueError(f'Contractions should have indices as well to determine signs.\n' +
                         f'Use wick_pairs() with index=True option before wick_contraction.')

    delta = '\\delta' if latex else 'delta'
    underline = '_\\' if latex else '_'

    index = []
    deltas = []
    for (i1, i2, op1, op2) in contractions:
        orb1 = get_orbital_index(op1) + underline + get_spin_label(op1)
        orb2 = get_orbital_index(op2) + underline + get_spin_label(op2)
        deltas.append(delta+'_{'+orb1+','+orb2+'}')
        index.append((i1, i2))

    sign = find_delta_sign(index, dtype=str)
    return sign + ' '.join(deltas)


def find_delta_sign(contractions_index, dtype=str):
    """
    Determine the sign of the contraction based on the number of crossings.

    Parameters
        contractions : list of contraction pairs
        dtype : data type of the return value (str or int)

    Returns
        sign : '+' or '-' (+1 or -1) depending on the number of crossings
    """
    sign = 1
    n = len(contractions_index)
    for i, (i1, i2) in enumerate(contractions_index):
        if i1 > i2: # ensure i1 < i2
            i1, i2 = i2, i1
        for _, (j1, j2) in enumerate(contractions_index[i:]):
            if j1 > j2:
                j1, j2 = j2, j1

            if i1 < j1 < i2 < j2: # crossing detected
                sign *= -1

    if dtype == str:
        return '+' if sign == 1 else '-'
    else:
        return sign


def sqo_evaluation(bra, hamiltonian, ket, exceptions=[], title='', latex=True):
    """
    Evaluate the Wick contractions for the given second-quantization operator (sqo) strings of bra, Hamiltonian, and ket,
    while excluding specified operator pairs from contraction.

    Parameters
        bra : left side excitation operator string
        hamiltonian : Hamiltonian operator string
        ket : right side excitation operator string
        exceptions : list of tuples, each containing a pair of operators to exclude from contraction
        title : optional title for the evaluation
        latex : if True, format the output of delta strings for LaTeX rendering

    Returns
        contractions : list of contraction patterns
    """
    if isinstance(bra, list): # in order of left-to-right
        bra = ' '.join(bra)
    if isinstance(ket, list): # in order of left-to-right
        ket = ' '.join(ket)

    operators = bra + ' ' + hamiltonian + ' ' + ket
    print('operators:\n', operators)
    operators = get_list(operators)
    pairs = wick_pairs(operators, exceptions=exceptions, index=True)
    contractions = wick_contraction(operators, pairs, expand=True)

    print(title)
    for i, pattern in enumerate(contractions):
        print('Pattern:', i+1)
        for (i1, i2, op1, op2) in pattern:
            print(f'Contracting {op1} with {op2};')
    print('')

    deltas = wick_delta(contractions, latex=latex)
    string = ' '.join(deltas)
    print('Contraction result:\n', string)
    print('')

    return contractions, deltas



if __name__ == '__main__':
    #operators = 'a_a^ i_b b_b† j_α c_b^ k_beta'
    operators = 's_alpha^ a_alpha p_sigma^ q_tau b_alpha^ t_alpha'
    exceptions = [('s_alpha^', 'a_alpha'), ('b_alpha^', 't_alpha')]
    pairs = wick_pairs(operators, exceptions=exceptions, index=True)
    print('pairs:', pairs)
    contractions = wick_contraction(operators, pairs, expand=True)
    print('contractions:', contractions)
    deltas = wick_delta(contractions)
    print('deltas:', deltas)
