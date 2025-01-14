import numpy as np
import itertools

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
