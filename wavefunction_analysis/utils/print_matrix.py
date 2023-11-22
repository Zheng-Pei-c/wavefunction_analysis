import os, sys
import numpy as np

def print_matrix(keyword, matrix, nwidth=0, nind=0, trans=False, dtype=float):
    if '\n' in keyword[-3:]: keyword = keyword[:-2]
    print(keyword)

    if isinstance(matrix, list): matrix = np.array(matrix)

    # transpose the last two dimensions
    if trans: matrix = np.einsum('...ij->...ji', matrix)

    ndim = matrix.ndim
    if ndim == 1: # 1d array
        if nwidth==0: nwidth = 6
        for n in range(len(matrix)):
            if nind > 0: # column index
                print('%13d ' % n, end='')
                if (n+1)%nwidth==0: print('')
            print('%13.8f ' % matrix[n], end='')
            if (n+1)%nwidth==0: print('')
        print('\n')

    elif ndim == 2: # 2d array
        nrow, ncol = matrix.shape
        if nwidth==0:
            nloop = 1
        else:
            nloop = ncol//nwidth
            if nloop*nwidth<ncol: nloop += 1

        for n in range(nloop):
            s0, s1 = n*nwidth, (n+1)*nwidth
            if s1>ncol or nwidth==0: s1 = ncol

            if nind > 0: # column index
                for c in range(s0, s1):
                    print('%13d ' % (c+1), end='')
                print('')

            for r in range(nrow):
                if nind > 0: # row index
                    print('%3d ' % (r+1), end='')
                for c in range(s0, s1):
                    print('%13.8f ' % matrix[r,c], end='')
                print('')

            if nind == 0: # blank line if without column index
                print('')

    elif ndim == 3: # 3d array
        for i in range(matrix.shape[0]):
            print_matrix(keyword+' '+str(i+1)+' ', matrix[i], nwidth, nind)

    elif ndim == 4: # 4d array
        for i in range(matrix.shape[0]):
            print_matrix(keyword+' '+str(i+1)+' ', matrix[i], nwidth, nind)

    elif ndim == 5: # 5d array
        n1, n2, n3 = matrix.shape[:3]
        for i in range(n1):
            for j in range(n2):
                for k in range(n3):
                    print_matrix(keyword+' i: '+str(i+1)+'  j: '+str(j+1)+'  k: '+str(k+1)+'  ', matrix[i, j, k], nwidth, nind)

    elif ndim == 6: # 6d array
        n1, n2, n3, n4 = matrix.shape[:4]
        for i in range(n1):
            for j in range(n2):
                for k in range(n3):
                    for l in range(n4):
                        print_matrix(keyword+' i: '+str(i+1)+'  j: '+str(j+1)+'  k: '+str(k+1)+'  l: '+str(l+1)+'  ', matrix[i, j, k, l], nwidth, nind)

    else:
        warnings.warn('the matrix has higher dimension than this funciton can handle.')
