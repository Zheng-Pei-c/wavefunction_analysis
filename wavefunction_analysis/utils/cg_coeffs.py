import os, sys
import numpy as np

from wavefunction_analysis.utils import print_matrix

def print_cg_coeff(j1=0, m1=0, j2=0, m2=0, j3=0, m3=0, cg=0,
        header=False, ic=0):
    if header:
        print('%9s %5s %5s %5s %5s %5s %7s' % ('j1', 'm1', 'j2', 'm2', 'j3', 'm3', 'CG'))

    if ic >= 0:
        print('%2d: %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %9.4f'
                % (ic, j1, m1, j2, m2, j3, m3, cg))


def ladder_coeff(j, m, operator):
    """
    j could be 0, \pm 1/2, \pm 1, \pm 3/2, \pm 2, \cdots
    m is \in {-j, -j+1, \dots, j-1, j}
    C_\pm = \sqrt(j (j + 1) - m (m \pm 1)) = \sqrt((j \mp m) (j \pm m + 1))
    j_\pm |jm> = C_\pm |j(m\pm1)>
    """
    if operator == '+': operator = np.add
    elif operator == '-': operator = np.subtract
    else: raise ValueError('operator undefined')

    if m < -j or m > j:
        #raise ValueError('m should in [-j, j]')
        return 0.

    return np.sqrt(j*(j+1) - m*(operator(m, 1)))


def clebsch_gordan_coeff_direct(j1, m1, j2, m2, j3, m3):
    """
    j3 is \in {|j_1 - j_2|, \dots, j_1 + j_2}
    m3 = m_1 + m_2 \in {-J, -J+1, \dots, J-1, J}
    """
    if (m1<-j1 or m1>j1) or (m2<-j2 or m2>j2) or (m3<-j3 or m3>j3) or (m1+m2 != m3):
        return 0.

    from scipy.special import factorial

    cg = (2.*j3+1)*factorial(j3+j1-j2) * factorial(j3-j1+j2) * factorial(j1+j2-j3) / factorial(j1+j2+j3+1)
    cg *= factorial(j3+m3) * factorial(j3-m3) * factorial(j1+m1) * factorial(j1-m1) * factorial(j2+m2) * factorial(j2-m2)
    cg = np.sqrt(cg)

    a = np.array([j1 + j2 - j3, j1 - m1, j2 + m2, j3 - j2 + m1, j3 - j1 - m2])

    kmin, kmax = max(0., -np.min(a[3:])), np.min(a[:3])
    if kmax < kmin: return 0.

    c = 0.
    for k in np.arange(kmin, kmax+1):
        c1 = a - k
        c1[3:] = a[3:] + k

        c1 = factorial(k) * np.prod(factorial(c1))
        c += (-1)**k / c1

    return cg*c


def clebsch_gordan_coeff_recur(j1, m1, j2, m2):
    """
    j3 is \in {|j_1 - j_2|, \dots, j_1 + j_2}
    m3 = m_1 + m_2 \in {-J, -J+1, \dots, J-1, J}
    """
    m3 = m1 + m2
    for j3 in range(abs(j1-j2), j1+j2+1):
        return


if __name__ == '__main__':
    #for j in range(3):
    #    for m in range(-j, j+1):
    #        cp = ladder_coeff(j, m, '+')
    #        cm = ladder_coeff(j, m, '-')
    #        print('j: %2d m: %2d cp: %8.4f cm: %8.4f' % (j, m, cp, cm))


    print_cg_coeff(header=True, ic=-1)

    ic = 0
    j1 = 1.
    j2 = 1.
    for m1 in np.arange(-j1, j1+1):
        for m2 in np.arange(-j2, j2+1):
            m3 = m1 + m2
            for j3 in np.arange(abs(j1-j2), j1+j2+1):
                cg = clebsch_gordan_coeff_direct(j1, m1, j2, m2, j3, m3)
                if abs(cg) > 1e-8:
                    print_cg_coeff(j1, m1, j2, m2, j3, m3, cg, False, ic)
                    ic += 1
