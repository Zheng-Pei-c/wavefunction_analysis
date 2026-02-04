import os, sys
import numpy as np

from lumeq.utils.sec_mole import read_geometry, write_geometry, switch_atoms


if __name__ == '__main__':
    mol = 'ortho'

    infile = 'final'+mol+'.xyz'
    geometry = read_geometry(infile)

    # para
    atom_list = [1, 2, 5, 4, 3, 11, 9, 10, 7, 8,
                 6, 33, 47, 32, 30, 31, 28, 29, 27, 26,
                 46, 25, 23, 24, 21, 22, 20, 19, 45, 18,
                 16, 17, 14, 15, 13, 12, 44, 43, 34, 35,
                 36, 37, 38, 39, 40, 41, 42, 48]

    # meta
    atom_list = [1, 2, 5, 4, 3, 11, 9, 10, 7, 8,
                 6, 33, 48, 32, 30, 31, 28, 29, 27, 26,
                 47, 25, 23, 24, 21, 22, 20, 19, 46, 18,
                 16, 17, 14, 15, 13, 12, 45, 43, 42, 40,
                 41, 39, 44, 37, 35, 36, 34, 38]

    # ortho
    atom_list = [1, 2, 5, 4, 3, 11, 9, 10, 7, 8,
                 6, 33, 48, 32, 30, 31, 28, 29, 27, 26,
                 47, 25, 23, 24, 21, 22, 20, 19, 46, 18,
                 16, 17, 14, 15, 13, 12, 45, 42, 41, 39,
                 40, 38, 43, 37, 35, 36, 34, 44]

    geometry = switch_atoms(geometry, atom_list)

    newfile = 'final'+mol+'_new.xyz'
    write_geometry(newfile, geometry, 'w+')
