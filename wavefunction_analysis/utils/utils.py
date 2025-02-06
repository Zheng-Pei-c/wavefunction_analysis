import numpy as np
import itertools

def random_matrix(n=10, mean=0., width=.1, func='uniform', sym=True, diag=True):
    if func == 'uniform':
        matrix = np.random.rand(n, n)*width + mean
        if sym:
            matrix = (matrix + matrix.conj().T) / 2.
        if diag is False: # remove diagonal elements
            mask = np.eye(n, dtype=bool)
            matrix[mask] = 0.
    elif func in {'gaussian', 'normal'}:
        matrix = np.random.normal(mean, width, n)
    elif func == 'standard':
        matrix = np.random.standard_normal(n)
    return matrix


def ishermitian(keyword, matrix, digits=[2,13,8,'f'], debug=0):
    print(keyword+' is Hermitian? ', end='')
    print(np.allclose(matrix, matrix.conj().T))
    if debug > 0:
        n = matrix.shape[0]
        w, width, precision, notation = digits
        for (i, j) in itertools.product(range(n), range(n)):
            f0, f1 = matrix[i,j], matrix[j,i]
            print(f'{i:{w}{'d'}} ', end='')
            print(f'{j:{w}{'d'}}: ', end='')
            print(f'{f0.real:{width}.{precision}{notation}} ', end='')
            print(f'{f1.real:{width}.{precision}{notation}} ', end='')
            print(f'{f0.imag:{width}.{precision}{notation}} ', end='')
            print(f'{f0.imag:{width}.{precision}{notation}} ')
        print()


def swap_largest_to_diagonal(matrix):
    for i in range(matrix.shape[0]):
        idx = np.argmax(np.abs(matrix[i]))
        matrix[i,i], matrix[i, idx] = matrix[i, idx], matrix[i,i]

    return matrix
