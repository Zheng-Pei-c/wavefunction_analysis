from wavefunction_analysis import sys, np, itertools
from wavefunction_analysis.spins import sympy

import unicodedata
from collections import defaultdict

"""
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
            return 'α'
        elif spin in ('b', 'beta', 'down', 'β'):
            return 'β'
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


def wick_pairs(operators, exceptions=[]):
    """
    Pick all the possible Wick contraction pairs from a string of creation and annihilation operators.

    Parameters
        operators : list or str separated by spaces or commas
            A list of strings representing creation and annihilation operators with optional spin labels.
            Eg. ['a^', 'i', 'b†', 'j_α', 'c_b^', 'k_beta']
                 or "a^ i b† j_α c_b^ k_beta"
        exceptions : (list of) tuple pairs of operators that should not be contracted

    Returns
        pairs : a list of the possible contraction pairs of the operators
    """
    if isinstance(operators, str):
        if ',' in operators:
            operators = operators.replace(',', ' ')
        operators = operators.split()

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
                if sp_i in ['α', 'β', None] and sp_j in ['α', 'β', None]:
                    if sp_i == sp_j: # both known spins and should match
                        pairs.append((op_i, op_j))
                else:
                    pairs.append((op_i, op_j))

    return pairs


def wick_contraction(pairs, n_operators, expand=True):
    """
    Perform Wick contraction on the given pairs of operators.

    Parameters
        pairs : list of tuple pairs of operators to be contracted
        n_operators : total number of operators to be contracted
        expand : True return product patterns;
                 False return dict of lists of contraction patterns

    Returns
        contractions : list of contraction patterns in the form of strings
    """
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
                    c_list.append((a, b))
                    c_flat += [a, b]

            if len(c_flat) == n_operators and not has_pattern(contractions, c_list):
                #print(c_list)
                contractions.append(c_list)

    return contractions


def wick_delta(contractions):
    """
    Convert contraction pairs into delta functions.

    Parameters
        contractions : list of contraction patterns in the form of strings

    Returns
        deltas : list of delta function strings representing the contractions
    """
    if isinstance(contractions[0], list): # loop over multiple contraction patterns
        return [wick_delta(pair) for pair in contractions]

    deltas = []
    for (op1, op2) in contractions:
        orb1 = get_orbital_index(op1) + '_' + get_spin_label(op1)
        orb2 = get_orbital_index(op2) + '_' + get_spin_label(op2)
        deltas.append('delta_{'+orb1+','+orb2+'}')
    return ' '.join(deltas)



if __name__ == '__main__':
    #operators = 'a_a^ i_b b_b† j_α c_b^ k_beta'
    operators = 's_alpha^ a_alpha p_sigma^ q_tau b_alpha^ t_alpha'
    exceptions = [('s_alpha^', 'a_alpha'), ('b_alpha^', 't_alpha')]
    pairs = wick_pairs(operators, exceptions=exceptions)
    print('pairs:', pairs)
    contractions = wick_contraction(pairs, n_operators=len(operators.split()), expand=True)
    print('contractions:', contractions)
    deltas = wick_delta(contractions)
    print('deltas:', deltas)
