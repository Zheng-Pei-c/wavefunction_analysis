import os, sys
import numpy as np

def read_time(filename):
    keyword = 'total wall time:'
    with open(filename, 'r') as infile:
        for line in infile:
            if line.find(keyword) >= 0:
                time = float(line.split(keyword)[1].split('sec')[0])
                if float(time) / 3600 > 1: time = str('{:.2f} h'.format(float(time) / 3600))
                elif float(time) / 60 > 1: time = str('{:.2f} min'.format(float(time) / 60))
                return time


def read_number(filename, keyword, n=-1, itype='i'):
    numbers = []
    with open(filename, 'r') as infile:
        for line in infile:
            if line.find(keyword) >= 0:
                if n == -1:
                    return float(line.split()[2][1:-1])
                else:
                    numbers.append(int(line.split()[n]))

    if len(numbers) == 1: numbers = numbers[0]
    return numbers


def read_matrix(filename, nrow, ncol, keyword, nwidth=10, nind=0, nskip=0):
    if nwidth == -1: nwidth = ncol

    nbatch = ncol // nwidth
    if nbatch * nwidth < ncol: nbatch += 1

    if nrow > 1:
        matrices = []

        with open(filename, 'r') as infile:
            for line in infile:
                if line.find(keyword) >= 0:

                    for n in range(nskip): # skip rows
                        line = next(infile)

                    matrix = np.zeros((nrow, ncol))
                    for k in range(nbatch):
                        if k > 0: line = next(infile)
                        elif nind > 0: line = next(infile) # skip the top index
                        for i in range(nrow):
                            data = next(infile).split()
                            for d in range(len(data)-nind):
                                if k*nwidth+d < ncol:
                                    matrix[i, k*nwidth+d] = float(data[d+nind])
                    matrices.append(matrix)

        if len(matrices) == 1: matrix = matrices[0]
        else: matrix = np.array(matrices)

        return matrix
    elif nrow == 1:
        matrices = []

        with open(filename, 'r') as infile:
            for line in infile:
                if line.find(keyword) >= 0:

                    matrix = np.zeros(ncol)
                    for k in range(nbatch):
                        if nind > 0: line = next(infile) # skip the top index
                        data = next(infile).split()
                        for d in range(len(data)-nind):
                            if k*nwidth+d < ncol:
                                matrix[k*nwidth+d] = float(data[d+nind])
                    matrices.append(matrix)

        if len(matrices) == 1: matrix = matrices[0]
        else: matrix = np.array(matrices)

        return matrix
