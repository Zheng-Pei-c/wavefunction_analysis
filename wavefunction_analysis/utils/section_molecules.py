import os, sys
import numpy as np


def write_geometry(infile, coords, open_file_method='w'):
    with open(infile, open_file_method) as f:
        if infile[-4:] == '.xyz': f.write('%d\n\n' int(len(coords)/4))

        for atom in range(int(len(coords)/4)):
            f.write('%2s ' % coords[4*atom])
            for x in range(1, 4):
                f.write('%14.8f ' % coords[4*atom+x])
            f.write('\n')


def write_mol_info(infile, charge='0', multiplicity='1', open_file_method='w',
        itype=0):
    """
    itype: 0 normal job; 1 second job; 2 fragment
    """
    with open(infile, open_file_method) as f:
        if itype == 1:
            f.write('@@@\n')
        if itype == 2:
            f.write("---\n")
        else:
            f.write('$molecule\n')
        f.write('%s %s\n' %(charge, multiplicity))
        if itype == 1:
            f.write('read\n')


def write_mol_info_coords(infile, coords, charge='0', multiplicity='1',
        frgm=False, open_file_method='w'):

    with open(infile, open_file_method) as f:
        if frgm == False:
            write_mol_info(infile, charge, multiplicity, 'a+', 0)
            write_geometry(infile, coords, 'a')
        #elif frgm == True:
        #    natom2 = 0
        #    for atom in range(0,int(len(coords)/4)):
        #        if str(coords[4*atom]).lower() == 'ag':
        #            natom2 += 1
        #    f.write("---\n0 1\n")
        #    for atom in range(0,int(len(coords)/4)-natom2):
        #        f.write("%2s %14.8f %14.8f %14.8f\n" \
        #            %(coords[4*atom], coords[4*atom+1], coords[4*atom+2], coords[4*atom+3]))
        #    f.write("---\n0 1\n")
        #    for atom in range(int(len(coords)/4)-natom2, int(len(coords)/4)):
        #        f.write("%2s %14.8f %14.8f %14.8f\n" \
        #            %(coords[4*atom], coords[4*atom+1], coords[4*atom+2], coords[4*atom+3]))
        f.write("$end\n\n")


