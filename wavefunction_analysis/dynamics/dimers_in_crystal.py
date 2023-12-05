import os, sys
import numpy as np
from numpy import cos, sin, arccos

# based on https://www.ucl.ac.uk/~rmhajc0/frorth.pdf
def read_unit_cell_info(ciff):
    if ciff[-4:] != '.cif':
        raise ValueError('read only cif file right now.')

    RAD = 180.0/np.pi

    elements = []
    scales = []
    with open(ciff, 'r') as cif_file:
        for line in cif_file:
            cols = line.split()
            ncols = len(cols)
            if ncols == 2 and cols[0] == '_cell_length_a':
                a = float(cols[1].split('(')[0])
            elif ncols == 2 and cols[0] == '_cell_length_b':
                b = float(cols[1].split('(')[0])
            elif ncols == 2 and cols[0] == '_cell_length_c':
                c = float(cols[1].split('(')[0])
            if ncols == 2 and cols[0] == '_cell_angle_alpha':
                alpha = float(cols[1].split('(')[0])/RAD
            elif ncols == 2 and cols[0] == '_cell_angle_beta':
                beta = float(cols[1].split('(')[0])/RAD
            elif ncols == 2 and cols[0] == '_cell_angle_gamma':
                gamma = float(cols[1].split('(')[0])/RAD
            if ncols > 10 and cols[6][0:1] == 'U':
                #print("ncols:", ncols, "line:", line)
                elements.append(cols[1])
                scales.append(float(cols[2].split('(')[0]))
                scales.append(float(cols[3].split('(')[0]))
                scales.append(float(cols[4].split('(')[0]))

    scales = np.array(scales).reshape((-1,3))

    #Eq. 7, alpha_star is the angle between y* and z* in the reciprocal space
    alpha_star = arccos((cos(beta)*cos(gamma)-cos(alpha))/sin(beta)/sin(gamma))

    return [a,b,c], [alpha,beta,gamma,alpha_star], elements, scales


def add_molecule(ix, iy, iz, inverse, abc, angles, elements, scales):
    #Eq. 17
    natoms = len(elements)
    coordinates = np.zeros_like(scales)

    a, b, c = abc[0], abc[1], abc[2]
    alpha, beta, gamma, alpha_star = angles[0], angles[1], angles[2], angles[3]

    for i in range(natoms):
        x = scales[i,0]
        y = scales[i,1]
        z = scales[i,2]
        if inverse == 1:
            x *= -1.0
            y *= -1.0
            z *= -1.0
        coordinates[i,0] = (ix+x) * a + (iy+y) * b * cos(gamma) + (iz+z) * c * cos(beta)
        coordinates[i,1] =              (iy+y) * b * sin(gamma) - (iz+z) * c * sin(beta) * cos(alpha_star)
        coordinates[i,2] =                                        (iz+z) * c * sin(beta) * sin(alpha_star)

    return coordinates


def write_xyz_files(elements, coordinates, mol):
    natoms = len(elements)
    with open(mol+'.xyz', 'w') as xyzf:
        xyzf.write('%3d \n\n' %(natoms))
        for i in range(0, natoms):
            xyzf.write('%1s %12.7f %12.7f %12.7f\n' %(elements[i], coordinates[i,0], coordinates[i,1], coordinates[i,2]))


def write_eda_files(elements, coordinates, mol):
    natoms = len(elements)
    with open(mol+'-dc.inp', 'w') as xyzf:
        xyzf.write('$molecule\n0 1\n--\n0 1\n')
        for i in range(natoms):
            xyzf.write('%1s %12.7f %12.7f %12.7f\n' %(elements[i], coordinates[i,0], coordinates[i,1], coordinates[i,2]))
            if i == natoms/2-1: xyzf.write('--\n0 1\n')
        xyzf.write('$end')


if __name__ == '__main__':
    mol = 'H2OBPc'
    abc, angles, elements, scales = read_unit_cell_info(mol+'.cif')
    print('abc:', abc, 'angles:', angles)
    natoms = len(elements)
    print('natoms:', natoms)

    n_images = [3,3,3]
    n_total = 2 * n_images[0] * n_images[1] * n_images[2]
    print('n_total:', n_total)
    ix_min = -int((n_images[0]-1)/2)
    iy_min = -int((n_images[1]-1)/2)
    iz_min = -int((n_images[2]-1)/2)

    elements_all = []
    coordinates = np.zeros((n_total*natoms, 3))
    centers_all = np.zeros((n_total, 3))

    icount = 0
    for ix in range(ix_min, ix_min+n_images[0]):
        for iy in range(iy_min, iy_min+n_images[1]):
            for iz in range(iz_min, iz_min+n_images[2]):
                for inverse in range(2):
                    for i in range(natoms):
                        elements_all.append(elements[i])

                    coordinates[icount*natoms:(icount+1)*natoms,:] = add_molecule(ix, iy, iz, inverse, abc, angles, elements, scales)
                    centers_all[icount, :] = .5 * (coordinates[icount*natoms+4,:] + coordinates[icount*natoms+8,:])
                    icount += 1

    fname = mol+'-'+str(n_images[0])+'-'+str(n_images[1])+'-'+str(n_images[2])
    write_xyz_files(elements_all, coordinates, fname)

    distances = []
    i = int(n_total//2)
    print('i:', i)
    for j in range(n_total):
        distances.append(np.linalg.norm(centers_all[i,:]-centers_all[j,:]))
    distances = np.array(distances)

    order = distances.argsort()
    print('distances:', distances)
    print('order:', order)
    print('distances:', distances[order[:12]])

    for k in range(1, 12):
      fname = mol+'-'+str(i+1)+'-'+str(order[k]+1)+'-dimer'
      coordinates_dimer = np.zeros((2*natoms, 3))
      coordinates_dimer[:natoms, :]         = np.copy(coordinates[i*natoms:(i+1)*natoms,:])
      coordinates_dimer[natoms:2*natoms, :] = np.copy(coordinates[order[k]*natoms:(order[k]+1)*natoms,:])
      write_xyz_files(elements_all[:2*natoms], coordinates_dimer, fname)
      write_eda_files(elements_all[:2*natoms], coordinates_dimer, fname)
