import sys

import numpy as np

def gradient_descent(func, gradf, retraction, x0,
                     method='armijo', nmax=50, thresh=1e-8,
                     *args, **kwargs):
    print('%s line search method is used in gradient descent' % method)

    xi = x0
    y0 = func(x0)

    xs, ys = [x0], [y0]

    ymin = kwargs.get('ymin', None)

    for i in range(nmax):
        step_size = line_search(func, gradf, retraction, xi, method, *args, **kwargs)

        dx = -step_size * gradf(xi)
        xi = retraction(xi, dx)
        yi = func(xi)

        xs.append(xi)
        ys.append(yi)

        if ymin:
            error = abs(yi - ymin)
        else:
            error = abs(yi - y0)

        print('i:%3d  value:%12.8f  error: %8.4e' % (i, yi, error), end=' ')
        if error > thresh:
            print('')
        else:
            print('optimization converged!')
            break

        y0 = yi

    return np.array(xs), np.array(ys)


def line_search(func, gradf, retraction, x0, method='armijo', *args, **kwargs):
    """
    method options: float, newton, backtracking, armijo, steepest
    """
    if isinstance(method, float): # step size is a given constant
        return method

    ls_method = eval('ls_'+method) # get the working function
    if callable(ls_method):
        #print('ls_method:', ls_method, callable(ls_method))
        return ls_method(func, gradf, retraction, x0, *args, **kwargs)
    else:
        raise NotImplementedError('line search method %s' % method)


def ls_newton_raphson(func, gradf, retraction, x0, **kwargs):
    hessf = kwargs.get('hessf')
    inv = np.linalg.inv(hessf)
    return


def ls_backtracking(func, gradf, retraction, x0,
                    tau=.5, r=1e-4, alpha=1.):
    """
    alpha is the step size we need to determine
    reduce it from a large value
    """
    y0 = func(x0)
    df = gradf(x0) # grad is also the negative descent direction
    #dot = np.dot(df, df) # -<grad, descent_direction>
    norm = np.linalg.norm(df)
    dot = norm**2

    alpha /= norm

    def kernel(alpha): # recursion
        dx = -alpha * df
        xi = retraction(x0, dx)
        dy = y0 - func(xi)

        if dy >= r * alpha * dot:
            return alpha

        alpha *= tau # reducing the step size
        return kernel(alpha)

    alpha = kernel(alpha)
    return alpha


ls_armijo = ls_backtracking


def ls_steepest(func, gradf, retraction, x0,
                tau=.5, r=1e-4, alpha=1000.):
    return 1.


def conjugate_gradient(func, gradf, retraction, x0,
                       method='cg_polak_ribiere', nmax=50, thresh=1e-8,
                       *args, **kwargs):
    xi = x0
    y0 = func(x0)

    xs, ys = [x0], [y0]

    for i in range(nmax):
        step_size = method(func, gradf, retraction, xi, method, *args, **kwargs)

        dx = -step_size * gradf(xi)
        xi = retraction(xi, dx)
        yi = func(xi)

        xs.append(xi)
        ys.append(yi)

        error = abs(yi - y0)
        print('i:%3d error: %8.4e  y:%10.5f' % (i, error, yi))
        if error < thresh: break

        y0 = yi

    return np.array(xs), np.array(ys)


def cg_exact(hessf, g1, d0):
    """
    f is the quadratic function xAx/2
    hessf is the hessian multiplies a vector as Ax
    """
    g0 = hessf(d0)
    return np.dot(g1, g0) / np.dot(d0, g0)


def cg_barzilai_borwein_1(g0, g1, t0):
    dg = g1 - g0
    return np.dot(t0, t0) / np.dot(dg, t0)


def cg_barzilai_borwein_2(g0, g1, t0):
    dg = g1 - g0
    return np.dot(dg, t0) / np.dot(dg, dg)


def cg_polak_ribiere(g0, g1):
    dg = g1 - g0
    return abs(np.dot(g1, dg)) / np.dot(g0, g0)


def cg_fletcher_reeves(g0, g1):
    return np.dot(g1, g1) / np.dot(g0, g0)


def cg_dai_yuan(g0, g1, t0):
    dg = g1 - g0
    return np.dot(g1, g1) / np.dot(t0, dg)


def cg_hestenes_stiefel(g0, g1, t0):
    dg = g1 - g0
    return np.dot(g1, dg) / np.dot(t0, dg)

