import sys
import numpy as np
import scipy

from wavefunction_analysis.manifold import gradient_descent, conjugate_gradient

import pymanopt

def get_random_matrix(ndim, seed=None, sym=True):
    rng = np.random.default_rng(seed)

    if isinstance(ndim, int): # square matrix
        m, n = ndim, ndim

    elif len(ndim) == 1: # vector
        return rng.random(ndim[0])

    elif len(ndim) == 2: # rectrangular matrix
        m, n = ndim

    A = rng.random((m, n))

    if sym and (m==n):
        A = .5 * (A + A.T)

    return A



class Riemannian():
    def __init__(self, ndim=None, x0=None, A=None, retraction='qr', **kwargs):
        if isinstance(ndim, float):
            self.ndim = np.array([ndim], dtype=int)
        elif isinstance(x0, np.ndarray):
            self.ndim = list(x0.shape)

        if self.ndim is None:
            raise ValueError('Riemannian dimension is not given')
        elif len(self.ndim) == 1:
            retraction = 'norm'
        elif len(self.ndim) == 2:
            if retraction == 'norm':
                retraction = 'qr'
                print('required retraction is not appliable and changed to qr')

        if isinstance(A, np.ndarray): # get a random symmetric matrix
            self.A = A
        else:
            seed = kwargs.get('seed', 12345)
            self.get_random_kernel(x0.shape[0], seed)


        try: # get the actural retraction method
            self.retraction = getattr(self, 'retraction_'+str(retraction))
        except AttributeError:
            raise ValueError('required retraction method is not implemented')

        if isinstance(x0, type(None)):
            x0 = self.get_random_matrix(self.ndim, seed=12345, sym=False)
        self.x0 = self.retraction(x0, np.zeros(x0.shape))

        self.check_sanity()


    def norm(self, x, v):
        """
        use euclidean norm by default
        """
        return np.dot(v, v)


    def retraction(self, x, v):
        """
        project new vector x+v at point x to the manifold
        """
        raise NotImplementedError('actural retraction is not implemented')


    def projection(self, x, v):
        """
        project vector v at point x to the manifold
        """
        raise NotImplementedError('actural projection is not implemented')


    to_tangent_space = projection


    def transport(self, x0, x1, v0):
        """
        transport tangent vector v0 at point x0 to point x1
        """
        return self.projection(x1, v0)


    def weingarten(self, x, v, normal):
        return


    def riemannian_gradient(self, x, grad):
        """
        project euclidean gradient to riemannian gradient
        """
        return self.projection(x, grad)


    def riemannian_hessian(self, x, v, grad, normal, hess):
        if isinstance(normal, type(None)):
            # euclidean_gradient - riemannian_gradient
            normal = grad - self.projection(x, grad)

        return self.projection(x, hess) + self.weingarten(x, v, normal)


    def gradient_descent(self, method=None, nmax=50, thresh=1e-8,
                         *args, **kwargs):
        if method is None: # assign a fixed suitable size by default
            method = 1. / (2. * np.linalg.norm(self.A))

        if kwargs.get('ymin') == 'eigenvalue':
            kwargs['ymin'] = self.eigenvalue[0]

        return gradient_descent(self.func, self.func_grad, self.retraction, self.x0,
                                method, nmax, thresh, *args, **kwargs)



# TODO: fix the N*N case
class OrthogonalGroup(Riemannian): # orthogonal group manifold
    def check_sanity(self):
        ndim = self.ndim
        if len(ndim)==2 and ndim[0]!=ndim[1]:
            raise ValueError('points on OrthogonalGroup should be square matrix')

    @property
    def eigenvalue(self):
        A = self.A
        e, v = np.linalg.eigh(A)

        arg = np.argsort(e)
        e, v = e[arg], v[:,arg]
        return e


    # x0 is nd square matrix
    def func(self, x): # <x|A|x> / <x|x>
        """assume x is normalized"""
        A = self.A
        return np.sum(np.einsum('m...,mn,n...->...', x, A, x))


    def func_grad(self, x): # 2 (I - |x><x|) A|x> = 2 (A - <x|A|x>I ) |x>
        # the returned grad has been projected on the tangent space
        """assume x is normalized"""
        A = self.A
        #x /= np.linalg.norm(x) # put it on sphere
        Ax = 2.* np.einsum('mn,n...->m...', A, x)
        return self.projection(x, Ax)


    def projection(self, x, v):
        return (v - x * np.sum(np.einsum('m...,m...->...', x, v)))


    def retraction_norm(self, x, v):
        """put (x+v) on the sphere"""
        x = x + v
        return x / np.linalg.norm(x, axis=0)


    def retraction_qr(self, x, v):
        x += x @ x
        q, r = np.linalg.qr(x)
        return q


    def retraction_polar(self, x, v):
        return x


    def retraction_exp(self, x, v):
        return scipy.linalg.expm(v) @ x


    def retraction_cayley(self, x, v):
        I = np.eye(x.shape[0])
        v *= .5
        m = np.linalg.inv((I - v))
        p = I + v
        return np.einsum('ij,jk,kl->il', m, p, x)



class Stiefel(OrthogonalGroup):
    def check_sanity(self):
        ndim = self.ndim
        if len(ndim)==1 or (len(ndim)==2 and ndim[0]<=ndim[1]):
            raise ValueError('points on Stiefel should be long rectagular matrix')

    def projection(self, x, v):
        tmp = x.T @ v
        tmp = .5 * (tmp + tmp.T)
        return (v - x @ tmp)


    def retraction_qr(self, x, v):
        x = x + v
        return pymanopt.tools.multi.multiqr(x)[0]


    def retraction_polar(self, x, v):
        x = x + v
        u, s, vt = np.linalg.svd(x+v, full_matrices=False)
        #return np.einsum('mn,nl->ml', u, vt)
        return u @ vt



class Grassmann(OrthogonalGroup):
    def retraction_qr(self, x, v):
        return



if __name__ == '__main__':
    ndim = 6
    A = get_random_matrix(ndim)

    nmax = 60

    itype = 'sphere'
    itype = 'stiefel'

    if itype == 'sphere':
        x0 = get_random_matrix([ndim])
        surf_obj = OrthogonalGroup(x0=x0, A=A)
        surf_obj2 = pymanopt.manifolds.Sphere(ndim)

        @pymanopt.function.autograd(surf_obj2)
        def cost(point):
            return point.T @ A @ point

    elif itype == 'stiefel':
        x0 = get_random_matrix((ndim, ndim-1))
        surf_obj = Stiefel(x0=x0, A=A)
        surf_obj2 = pymanopt.manifolds.Stiefel(ndim, ndim-1)

        @pymanopt.function.autograd(surf_obj2)
        def cost(point):
            #return np.trace(point.T @ A @ point)
            return np.sum(np.einsum('m...,mn,n...->...', point, A, point))

    #xs, ys = surf_obj.gradient_descent(nmax=nmax)
    #xs, ys = surf_obj.gradient_descent(ymin='eigenvalue')
    xs, ys = surf_obj.gradient_descent(method='backtracking', nmax=nmax)
    #print('x:', xs[-1])
    #print('norm:', np.einsum('ji,jk->ik', xs[-1], xs[-1]).diagonal())

    problem = pymanopt.Problem(surf_obj2, cost)

    optimizer_name = 'SteepestDescent'
    optimizer = getattr(pymanopt.optimizers, optimizer_name)(
                        max_iterations=nmax, verbosity=0)
    result = optimizer.run(problem, initial_point=surf_obj.x0)
    print('result:', result)
    #point = result.point
    #print('norm:', np.einsum('ji,jk->ik', point, point).diagonal())

    ## plot
    #import matplotlib.pyplot as plt
    #from mpl_toolkits.mplot3d import Axes3D

    #f = sphere.func
    #resolution = 51
    #C = np.zeros((resolution, resolution))
    #X, Y, Z = np.mgrid[-1:1:complex(resolution), -1:1:complex(resolution), -1:1:complex(resolution)]

    #for i in range(resolution):
    #    for j in range(resolution):
    #        v = np.array([X[i, j, 0], Y[i, j, 0], Z[i, j, 0]])
    #        C[i, j] = f(v)

    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.plot_surface(X[:, :, 0], Y[:, :, 0], Z[:, :, 0], facecolors=plt.cm.viridis(C), rstride=1, cstride=1, alpha=0.7, antialiased=True)
    #ax.set_xlabel('X')
    #ax.set_ylabel('Y')
    #ax.set_zlabel('Z')
    #ax.set_title('Rayleigh Quotient on the Sphere')

    #ax.plot(xs[:,0], xs[:,1], xs[:,2], '.', markersize=10, color='r')

    #plt.show()
