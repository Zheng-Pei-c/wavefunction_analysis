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


def read_number(filename, keyword, n=-1, o=1, dtype=int):
    """
    o takes 1 or -1 for the reading order when n>=0
    when n == -1: o takes the begin and end indices
    ftype can be int, float, or simply string
    """
    numbers = []
    with open(filename, 'r') as infile:
        for line in infile:
            if line.find(keyword) >= 0:
                if n == -1:
                    numbers.append(dtype(line[o[0]:(o[0]+o[1])]))
                else:
                    numbers.append(dtype(line.split()[n*o]))

    if len(numbers) == 1: numbers = numbers[0]
    return np.array(numbers, dtype=dtype)


def read_array(filename, ncol=4, nrange=[0,4], dtype=float, same=True):
    array = []
    with open(filename, 'r') as infile:
        for line in infile:
            data = line.split()
            if len(data) == ncol:
                issame = True
                if same:
                    for i in range(len(data)):
                        try:
                            f = dtype(data[i])
                        except:
                            issame = False
                if (not same) or issame:
                    for i in range(nrange[0], nrange[1]):
                        try:
                            array.append(dtype(data[i]))
                        except: # encounter a string
                            pass

    return np.array(array)


def read_matrix(filename, nrow, ncol, keyword, nwidth=6, nind=0, nskip=0, dtype=float):
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

                    matrix = np.zeros((nrow, ncol), dtype=dtype)
                    for k in range(nbatch):
                        if k > 0: line = next(infile)
                        elif nind > 0: line = next(infile) # skip the top index
                        for i in range(nrow):
                            data = next(infile).split()
                            for d in range(len(data)-nind):
                                if k*nwidth+d < ncol:
                                    matrix[i, k*nwidth+d] = dtype(data[d+nind])
                    matrices.append(matrix)

        if len(matrices) == 1: matrix = matrices[0]
        else: matrix = np.array(matrices)

        return matrix
    elif nrow == 1:
        matrices = []

        with open(filename, 'r') as infile:
            for line in infile:
                if line.find(keyword) >= 0:

                    matrix = np.zeros(ncol, dtype=dtype)
                    for k in range(nbatch):
                        if nind > 0: line = next(infile) # skip the top index
                        data = next(infile).split()
                        for d in range(len(data)-nind):
                            if k*nwidth+d < ncol:
                                matrix[k*nwidth+d] = dtype(data[d+nind])

                        for n in range(nskip): # skip rows
                            line = next(infile)
                    matrices.append(matrix)

        if len(matrices) == 1: matrix = matrices[0]
        else: matrix = np.array(matrices)

        return matrix
