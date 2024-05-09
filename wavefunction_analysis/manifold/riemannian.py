import sys
import numpy as np

def gradient_descent(func, gradf, retraction, x0,
                     method='armijo', nmax=50, thresh=1e-8,
                     *args, **kwargs):
    print('method:', method)
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
        print('i:%3d error: %8.4e  y:%10.5f' % (i, error, yi))
        if error < thresh: break

        y0 = yi

    return np.array(xs), np.array(ys)


def line_search(func, gradf, retraction, x0, method='armijo', *args, **kwargs):
    if isinstance(method, float): # step size is a given constant
        return method

    elif method == 'newton':
        return ls_newton_raphson(gradf, x0, *args, **kwargs)
    elif method == 'backtracking':
        return ls_backtracking(func, gradf, retraction, x0, *args, **kwargs)
    elif method == 'armijo':
        return ls_armijo(func, gradf, retraction, x0, *args, **kwargs)
    elif method == '':
        return
    else:
        raise NotImplementedError('line search method %s' % method)


def ls_newton_raphson(gradf, x0, hessf):
    inv = np.linalg.inv(hessf)
    return


def ls_backtracking(func, gradf, retraction, x0,
                    tau=.5, r=1e-4, alpha=1000.):
    """
    alpha is the step size we need to determine
    reduce it from a large value
    """
    y0 = func(x0)
    df = gradf(x0) # grad is also the negative descent direction

    def kernel(alpha): # recursion
        dx = -alpha * df
        xi = retraction(x0, dx)
        dy = y0 - func(xi)

        if dy >= r * alpha * np.dot(df, df): # -<grad, descent_direction>
            return alpha

        alpha *= tau # reducing the step size
        return kernel(alpha)

    alpha = kernel(alpha)
    return alpha


ls_armijo = ls_backtracking


def conjugate_gradient(func, gradf, retraction, x0,
                       method=cg_polak_ribiere, nmax=50, thresh=1e-8,
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



class Sphere(): # S^(n-1) sphere manifold
    def __init__(self, x0, A=None, seed=12345):
        if not isinstance(A, np.ndarray): # get a random symmetric matrix
            ndim = x0.size
            rng = np.random.default_rng(seed)
            A = rng.random((ndim, ndim))
            A = .5 * (A + A.T)

        # x0 is 1d array
        self.x0 = self.retraction(x0, np.zeros(x0.shape)) # put it on sphere
        self.A = A


    @property
    def eigenvalue(self):
        e, v = np.linalg.eigh(self.A)

        arg = np.argsort(e)
        e, v = e[arg], v[:,arg]
        return e


    def func(self, x): # <x|A|x> / <x|x>
        """assume x is normalized"""
        A = self.A
        return np.einsum('m,mn,n', x, A, x) #/ np.dot(x, x)


    def func_grad(self, x): # 2 (I - |x><x|) A|x> = 2 (A - <x|A|x>I ) |x>
        # the returned grad has been projected on the tangent space
        """assume x is normalized"""
        #x /= np.linalg.norm(x) # put it on sphere
        Ax = np.einsum('mn,n->m', self.A, x)
        return 2. * (Ax - np.dot(x, Ax) * x)


    def retraction(self, x, v):
        """put (x+v) on the sphere"""
        x = x + v
        return x / np.linalg.norm(x)


    def gradient_descent(self, method=None, nmax=50, thresh=1e-8,
                         *args, **kwargs):
        if method is None: # assign a fixed suitable size by default
            method = 1. / (2. * np.linalg.norm(self.A))

        if kwargs.get('ymin') == 'eigenvalue':
            kwargs['ymin'] = self.eigenvalue[0]

        return gradient_descent(self.func, self.func_grad, self.retraction, self.x0,
                                method, nmax, thresh, *args, **kwargs)



class OrthogonalGroup(Sphere): # orthogonal group manifold
    # x0 is nd square matrix
    def func(self, x): # <x|A|x> / <x|x>
        """assume x is normalized"""
        A = self.A
        return np.einsum('mi,mn,ni->', x, A, x)


    def func_grad(self, x): # 2 (I - |x><x|) A|x> = 2 (A - <x|A|x>I ) |x>
        # the returned grad has been projected on the tangent space
        """assume x is normalized"""
        #x /= np.linalg.norm(x) # put it on sphere
        Ax = np.einsum('mn,ni->mi', self.A, x)
        return 2. * (Ax - np.einsum('mi,mi->', x, Ax) * x)


    def retraction(self, x, v):
        """put (x+v) on the orthogonal group"""
        x = x + v
        return x / np.linalg.norm(x, axis=0)



class Stiefel(OrthogonalGroup):
    def retraction(self, x, v):
        u, s, vt = np.linalg.svd(x+v)
        return np.einsum('mn,nl->ml', u, vt)



if __name__ == '__main__':
    ndim = 3
    rng = np.random.default_rng(seed=12345)
    x0 = rng.random(ndim)
    x0 /= np.linalg.norm(x0) # put it on sphere

    sphere = Sphere(x0)
    xs, ys = sphere.gradient_descent()
    xs, ys = sphere.gradient_descent(ymin='eigenvalue')
    xs, ys = sphere.gradient_descent(method='backtracking')


    # plot
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    f = sphere.func
    resolution = 51
    C = np.zeros((resolution, resolution))
    X, Y, Z = np.mgrid[-1:1:complex(resolution), -1:1:complex(resolution), -1:1:complex(resolution)]

    for i in range(resolution):
        for j in range(resolution):
            v = np.array([X[i, j, 0], Y[i, j, 0], Z[i, j, 0]])
            C[i, j] = f(v)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X[:, :, 0], Y[:, :, 0], Z[:, :, 0], facecolors=plt.cm.viridis(C), rstride=1, cstride=1, alpha=0.7, antialiased=True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Rayleigh Quotient on the Sphere')

    ax.plot(xs[:,0], xs[:,1], xs[:,2], '.', markersize=10, color='r')

    plt.show()
