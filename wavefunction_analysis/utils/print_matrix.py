import os, sys
import numpy as np

def print_matrix(keyword, matrix, nwidth=0, nind=0):
    if '\n' in keyword[-3:]: keyword = keyword[:-2]
    print(keyword)

    if isinstance(matrix, list): matrix = np.array(matrix)

    if len(matrix.shape)==1: # 1d array
        if nwidth==0: nwidth = 6
        for n in range(len(matrix)):
            if nind > 0: # column index
                print('%13d ' % n, end='')
                if (n+1)%nwidth==0: print('')
            print('%13.8f ' % matrix[n], end='')
            if (n+1)%nwidth==0: print('')
        print('\n')

    elif len(matrix.shape)==2: # 2d array
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

    elif len(matrix.shape)==3: # 3d array
        for i in range(matrix.shape[0]):
            print_matrix(keyword+str(i+1), matrix[i], nwidth, nind)
    else:
        warnings.warn('the matrix has higher dimension than this funciton can handle.')
