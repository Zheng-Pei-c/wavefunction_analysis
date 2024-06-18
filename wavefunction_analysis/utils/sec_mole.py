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


def read_geometries_standard(infile, screen='User input: 2 of 2'):
    geometries = []
    with open(infile, 'r') as infile:
        for line in infile:
            if screen and line.find(screen)>=0:
                geometries = []

            if line.find('Standard Nuclear Orientation (Angstroms)')>=0:
                geometry = []
                line = next(infile)
                dash = next(infile)
                line = next(infile)
                while line != dash:
                    geometry.append(line[8:])
                    line = next(infile)

                geometries.append(geometry)

    if len(geometries) == 1:
        geometries = geometries[0]

    return geometries


def read_symbols_coords(infile, probe=1):
    geometry = read_geometry(infile, probe)

    return get_symbols_coords(geometry)


def get_symbols_coords(geometry, string=False):
    if string:
        geometry = [atom.split() for atom in geometry]
        geometry = sum(geometry, []) # convert to 1d list

    natom = int(len(geometry)/4)
    symbols, coords = [], np.zeros((natom, 3))

    for atom in range(natom):
        symbols.append(geometry[4*atom])
        for x in range(3):
            coords[atom, x] = geometry[4*atom+1+x] # convert to float automatically by np

    return symbols, coords


def write_geometry(infile, geometry, energy=None, open_file_method='w'):
    with open(infile, open_file_method) as f:
        if infile[-4:] == '.xyz':
            f.write('%d\n' % int(len(geometry)/4))
            if energy:
                f.write('%f\n' % energy)
            else:
                f.write('\n')

        for atom in range(int(len(geometry)/4)):
            f.write('%2s   ' % geometry[4*atom])
            for x in range(1, 4):
                f.write('%14.8f ' % geometry[4*atom+x])
            f.write('\n')


def write_symbols_coords(infile, symbols, coords, energy=None, open_file_method='w'):
    with open(infile, open_file_method) as f:
        if infile[-4:] == '.xyz':
            f.write('%d\n' % len(symbols))
            if energy:
                f.write('%f\n' % energy)
            else:
                f.write('\n')

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
            write_geometry(infile, kwargs.get('geometry'), open_file_method='a+')
        elif 'symbols' in kwargs and 'coords' in kwargs:
            write_symbols_coords(infile, kwargs.get('symbols'), kwargs.get('coords'), open_file_method='a+')
        else:
            raise ValueError('need geometry info')

    with open(infile, 'a+') as f:
        f.write("$end\n\n")


def write_rem_info(infile, method='pbe0', basis='6-31g', open_file_method='a+'):
    with open(infile, open_file_method) as f:
        f.write('$rem\n')
        f.write('method         %s\n' % method)
        f.write('basis          %s\n' % basis)
        #f.write('purecart       2222\n')
        f.write('sym_ignore     true\n')
        f.write('thresh         14\n')
        f.write('$end\n')


def get_rotation_matrix(theta, axis='x'):
    """
    [cos -sin]
    [sin  cos]
    """
    #i, j = 0, 0
    if axis == 'x' or axis == 0:
        i, j = 1, 2
    elif axis == 'y' or axis == 1:
        i, j = 2, 0
    elif axis == 'z' or axis == 2:
        i, j = 0, 1

    rot = np.eye(3)
    cos, sin = np.cos(theta), np.sin(theta)

    rot[i,i] = rot[j,j] = cos
    rot[i,j] = -sin
    rot[j,i] = sin

    return rot


def get_moment_of_inertia(weights, coords, fix_sign=False):
    # weights is charges or masses
    U = np.zeros((3,3))
    for i in range(len(weights)):
        m = weights[i]
        x, y, z = coords[i]

        U[0,0] += m * (y*y+z*z)
        U[1,1] += m * (x*x+z*z)
        U[2,2] += m * (x*x+y*y)
        U[1,0] -= m * x*y
        U[2,0] -= m * x*z
        U[2,1] -= m * y*z

    U[0,1] = U[1,0]
    U[0,2] = U[2,0]
    U[1,2] = U[2,1]

    if fix_sign:
        d, U = np.linalg.eigh(U)
        for i in range(3):
            if -np.min(U[:,i]) > np.max(U[:,i]): U[:,i] *= -1.

        if d[0]*d[1]*d[2] < 0.:
            #if abs(d[2] - d[0]) < 1e-6:
            #    u *= -1.
            if abs(d[2] - d[1]) < 1e-6:
                for i in range(3):
                    U[i,1], U[i,2] = U[i,2], U[i,1]
            elif abs(d[1] - d[0]) < 1e-6:
                for i in range(3):
                    U[i,0], U[i,1] = U[i,1], U[i,0]
            else:
                U *= -1.

    return U


def get_charge_or_mass(symbols, itype='charge', isotope_avg=True):
    from pyscf.data.elements import charge, MASSES, ISOTOPE_MAIN
    chgs = []
    for i in range(len(symbols)):
        chgs.append(charge(symbols[i]))

    if itype == 'mass':
        mass_table = MASSES if isotope_avg else ISOTOPE_MAIN
        mass = []
        for i in range(len(symbols)):
            mass.append(mass_table[chgs[i]])
        chgs = mass

    return chgs


def get_molecular_center(weights, coords, itype='charge', isotope_avg=True):
    # weights is charges or masses
    if isinstance(weights[0], str): # atom symbols
        weights = get_charge_or_mass(weights, itype, isotope_avg)

    return np.einsum('z,zx->x', weights, coords) / np.sum(weights)


def get_center_property(weights, props, itype='charge', isotope_avg=True):
    # weights is charges or masses
    if isinstance(weights[0], str): # atom symbols
        weights = get_charge_or_mass(weights, itype, isotope_avg)

    return np.einsum('z,z...->...', weights, props) / len(weights)


def translate_molecule(symbols, coords, origin=None, itype='charge', isotope_avg=True):
    # default origin is the center of charges/masses of the molecule
    if origin is None:
        weights = get_charge_or_mass(symbols, itype, isotope_avg)
        origin = get_molecular_center(weights, coords)
        return (coords - origin), weights
    else:
        return (coords - origin)


def align_principal_axes(charges, coords):
    U = get_moment_of_inertia(charges, coords, True)

    return np.einsum('nx,xy->ny', coords, U)


def standard_orientation(symbols, coords, tol=4):
    # translate to center of charge
    coords, chgs = translate_molecule(symbols, coords, itype='charge')
    coords, _ = _standard_orientation(coords, None, tol)
    coords = align_principal_axes(chgs, coords)

    return coords


def standard_orientation2(symbols, coords, var, tol=4):
    """we need the intermediate translation and principal matrices"""
    # translate to center of charge
    chgs = get_charge_or_mass(symbols, itype='charge')
    origin = get_molecular_center(chgs, coords)

    coords, var = _standard_orientation((coords-origin), (var-origin), tol)
    coords, _ = _standard_orientation(coords, None, tol)

    U = get_moment_of_inertia(chgs, coords, True)
    coords = np.einsum('nx,xy->ny', coords, U)
    var = np.einsum('...x,xy->...y', var, U)

    return coords, var


def _standard_orientation(coords, var=None, tol=4):
    tol = np.power(10, -float(tol)) #1e-tol

    # var is a geometry-dependent object (vector/matrix)
    if isinstance(var, type(None)):
        var = np.zeros(coords.shape)

    def rotation(coords, var, a, b, axis):
        r = np.sqrt(a*a+b*b)
        if r < 1e-10:
            return coords

        theta = np.arccos(a/r)
        if b < 0.: theta *= -1.
        rot = get_rotation_matrix(theta, axis)
        coords = np.einsum('nx,xy->ny', coords, rot)
        var = np.einsum('...x,xy->...y', var, rot)
        return coords, var

    def kernel(coords, var, i, x, y, z, level):
        if level >= 3:
            coords, var = rotation(coords, var, x, y, 'z') # rotate to X axis

        if level >= 2:
            x, y, z = coords[i]
            coords, var = rotation(coords, var, z, x, 'y') # rotate to +Z axis

        if level >= 1:
            for j in range(i+1, natoms):
                x, y, z = coords[j]
                if np.sqrt(x*x+y*y) > tol: # rotate second atom to +X semi-plane
                    return rotation(coords, var, x, y, 'z')

        return coords, var


    natoms = coords.shape[0]
    for i in range(natoms):
        x, y, z = coords[i]
        if abs(x) > tol or abs(y) > tol:
            return kernel(coords, var, i, x, y, z, 3)

        if z < tol:
            return kernel(coords, var, i, x, y, z, 2)
        elif z > tol:
            return kernel(coords, var, i, x, y, z, 1)

    return coords, var
