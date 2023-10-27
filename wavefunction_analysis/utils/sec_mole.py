import os, sys
import numpy as np


def read_geometry(infile, probe=1):
    geometry = []

    if probe == 1: # all
        with open(infile, 'r') as infile:
            for line in infile:
                info = line.split()
                if len(info) == 4:
                    geometry.append(info[0])
                    geometry.append(float(info[1]))
                    geometry.append(float(info[2]))
                    geometry.append(float(info[3]))
    elif probe == 2: # probe only
        with open(infile, 'r') as infile:
            for line in infile:
                info = line.split()
                if len(info) == 4:
                    if str(info[0]).lower()!='ag':
                        geometry.append(info[0])
                        geometry.append(float(info[1]))
                        geometry.append(float(info[2]))
                        geometry.append(float(info[3]))
    elif probe == 3: # metal only
        with open(infile, 'r') as infile:
            for line in infile:
                info = line.split()
                if len(info) == 4:
                    if str(info[0]).lower()=='ag':
                        geometry.append(info[0])
                        geometry.append(float(info[1]))
                        geometry.append(float(info[2]))
                        geometry.append(float(info[3]))
    return geometry


def read_symbols_coords(infile, probe=1):
    geometry = read_geometry(infile, probe)

    natom = int(len(geometry)/4)
    symbols, coords = [], np.zeros((natom, 3))

    for atom in range(natom):
        symbols.append(geometry[4*atom])
        for x in range(3):
            coords[atom, x] = geometry[4*atom+1+x]

    return symbols, coords


def write_geometry(infile, geometry, open_file_method='w'):
    with open(infile, open_file_method) as f:
        if infile[-4:] == '.xyz': f.write('%d\n\n' % int(len(geometry)/4))

        for atom in range(int(len(geometry)/4)):
            f.write('%2s   ' % geometry[4*atom])
            for x in range(1, 4):
                f.write('%14.8f ' % geometry[4*atom+x])
            f.write('\n')


def write_symbols_coords(infile, symbols, coords, open_file_method='w'):
    with open(infile, open_file_method) as f:
        if infile[-4:] == '.xyz': f.write('%d\n\n' % len(symbols))

        for atom in range(len(symbols)):
            f.write('%2s   ' % symbols[atom])
            for x in range(3):
                f.write('%14.8f ' % coords[atom, x])
            f.write('\n')


def switch_atoms(geometry, atom_list):
    if len(atom_list) != len(set(atom_list)):
        raise ValueError('atom_list has %2d duplicates' % (len(atom_list) - len(set(atom_list))))
    if len(atom_list) > len(geometry)//4:
        raise ValueError('atom_list has more elements')

    geometry_new = []
    for j in atom_list:
        j = j-1
        for x in range(4):
            geometry_new.append(geometry[4*j+x])

    return geometry_new


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


def write_mol_info_geometry(infile, charge='0', multiplicity='1',
        frgm=False, **kwargs):

    if frgm == False:
        write_mol_info(infile, charge, multiplicity, 'w+', 0)
        if 'geometry' in kwargs:
            write_geometry(infile, kwargs.get('geometry'), 'a+')
        elif 'symbols' in kwargs and 'coords' in kwargs:
            write_symbols_coords(infile, kwargs.get('symbols'), kwargs.get('coords'), 'a+')
        else:
            raise ValueError('need geometry info')

    with open(infile, 'a+') as f:
        f.write("$end\n\n")


def write_rem_info(infile, method='pbe0', basis='6-31g', open_file_method='a+'):
    with open(infile, open_file_method) as f:
        f.write('$rem\n')
        f.write('method         %s\n' % method)
        f.write('basis          %s\n' % basis)
        f.write('sym_ignore     true\n')
        f.write('thresh         14\n')
        f.write('$end\n')

