import sys
import numpy as np

def line_search_backtracking(func, gradf, retraction, x0, tau=.5, r=1e-4, alpha=1000.):
    y0 = func(x0)
    df = gradf(x0)
    df = np.dot(df, df)

    def kernel(alpha):
        if dy >= r * alpha * df:
            return alpha

        dx = alpha * df
        xi = retraction(x0, dx)
        dy = y0 - func(xi)

        alpha *= tau
        return kernel(alpha)

    alpha = kernel(alpha)


def gradient_descent(func, gradf, retraction, x0,
                     nmax=50, step_size=None, thresh=1e-8):
    if step_size is None:
        step_size = 1.

    xi = x0
    y0 = func(x0)

    xs, ys = [], []
    xs.append(x0)
    ys.append(y0)

    for i in range(nmax):
        dx = -step_size * gradf(xi)
        xi = retraction(xi, dx)
        yi = func(xi)

        xs.append(xi)
        ys.append(yi)

        error = abs(yi - y0)
        print('i:', i, error, yi)
        if error < thresh: break

        y0 = yi

    return np.array(xs), np.array(ys)


def gradient_descent2(A, func_grad, retraction, x0,
                     nmax=50, step_size=None, thresh=1e-8):
    if step_size is None:
        step_size = 1. / (2. * np.linalg.norm(A)) # this step_size has fastest convergence
        print('step_size:', step_size)

    xi = x0
    eigen_min = np.min(np.linalg.eigvals(A))

    for i in range(nmax):
        #y = np.einsum('m,mn,n', xi, A, xi)
        dx = -step_size * func_grad(xi)
        xi = retraction(xi, dx)

        yi = np.einsum('m,mn,n', xi, A, xi)
        error = abs(yi - eigen_min)
        print('i:', i, error, yi)
        if error < thresh: break

    return xi, yi



class orthogonal_group(): # S^(n-1) manifold
    def __init__(self, x0, A=None, seed=12345):
        if not isinstance(A, np.ndarray):
            ndim = x0.size
            rng = np.random.default_rng(seed)
            A = rng.random((ndim, ndim))
            A = .5 * (A + A.T)

        self.x0 = self.retraction(x0, np.zeros(x0.shape)) # put it on sphere
        self.A = A


    @property
    def eigenvalue(self):
        e, v = np.linalg.eigh(self.A)
        return e


    def func(self, x): # <x|A|x> / <x|x>
        """assume x is normalized"""
        A = self.A
        return np.einsum('m,mn,n', x, A, x) #/ np.dot(x, x)


    def func_grad(self, x): # 2 (I - |x><x|) A|x> = 2 (A - <x|A|x>I ) |x>
        """assume x is normalized"""
        #x /= np.linalg.norm(x) # put it on sphere
        Ax = np.einsum('mn,n->m', self.A, x)
        return 2. * (Ax - np.dot(x, Ax) * x)


    def retraction(self, x, v):
        """put (x+v) on the sphere"""
        x = x + v
        return x / np.linalg.norm(x)


    def gradient_descent(self, nmax=50, step_size=None, thresh=1e-8):
        step_size = 1. / (2. * np.linalg.norm(self.A))
        return gradient_descent(self.func, self.func_grad, self.retraction,
                self.x0, nmax, step_size, thresh)



def Stiefel(orthogonal_group):
    def retraction(self, x, v):
        u, s, vt = np.linalg.svd(x+v)
        return np.einsum('mn,nl->ml', u, vt)



if __name__ == '__main__':
    ndim = 3
    rng = np.random.default_rng(seed=12345)
    x0 = rng.random(ndim)
    x0 /= np.linalg.norm(x0) # put it on sphere

    sphere = orthogonal_group(x0)
    xs, ys = sphere.gradient_descent()
    x, y = gradient_descent2(sphere.A, sphere.func_grad, sphere.retraction, sphere.x0)


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
