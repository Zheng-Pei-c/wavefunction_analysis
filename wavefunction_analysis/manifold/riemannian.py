import sys
from wavefunction_analysis import np, scipy

from wavefunction_analysis.manifold import gradient_descent, conjugate_gradient, newton_2nd
from wavefunction_analysis.utils import get_ortho_basis

"""
the manifold code refers to Alan Edelman, Tomas, Arias, and Steven Smith,
                            SIAM J. Matrix Anal. Appl. 1998
as well as other packages such as `pymanopt`, `manopt`, etc
"""

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


def solve_sylvester(A, C, B=None, solver='iter'):
    """
    solve X from `AX + XB = C` aka. Lyapunov, Stein equation
    return X
    """
    n = A.shape[0]
    I = np.eye(n)
    if not isinstance(B, np.ndarray):
        B = A.conj().T

    A = np.kron(I, A) + np.kron(B, I)

    def kernel(A, y):
        x = np.linalg.solve(A, y.flatten(order='F'))
        return x.reshape((n,-1), order='F')

    if C.ndim == 2:
        return kernel(A, C)

    elif C.ndim == 3:
        x = []
        for i in range(C.shape[0]):
            x.append(kernel(A, C[i]))

        return np.array(x)



class Riemannian():
    def __init__(self, ndim=None, x0=None, A=None, B=None,
                 retraction='qr', **kwargs):
        """
        A: the quadratic function kernel
        B: orthogonal metric (overlap), default is I_n
        x0: initial point on the manifold
        """
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

        if not isinstance(A, np.ndarray): # get a random symmetric matrix
            self._A = get_random_matrix(self.ndim[0], kwargs.get('seed', 12345))
        self._A = A

        if isinstance(B, np.ndarray):
            self._B = B
            # B^1/2, B^-1/2, B^-1
            self._L, self._Z, self._Binv = get_ortho_basis(self._B, method='lowdin')
        else: # identity matrix
            self._B = np.eye(self.ndim[0])
            self._L, self._Z, self._Binv = np.copy(self._B), np.copy(self._B), np.copy(self._B)


        try: # get the actual retraction method
            self.retraction = getattr(self, 'retraction_'+str(retraction))
        except AttributeError:
            raise ValueError('required retraction method is not implemented')

        if isinstance(x0, type(None)):
            x0 = self.get_random_matrix(self.ndim, seed=12345, sym=False)
        self.x0 = self.retraction(x0, np.zeros(x0.shape))

        self.check_sanity()


    @property
    def dimension(self):
        raise NotImplementedError('actual dimension is not implemented')


    @property
    def norm(self, x, v):
        """
        calculate the length of a tangent vector v of point x
        use euclidean norm by default
        """
        return np.dot(v, v)


    @property
    def dist(self, v1, v2):
        """
        calculate the distance between two tangent vectors at one point
        """
        #TODO: is this true? I just wrote it down in a second...
        dv = v1 - v2
        return np.dot(dv, dv)


    def exp(self, x, v, dt=1.):
        """
        geodesic mapping tangent vector v of x to other points on manifold
        exp_p: T_p M -> M
        """
        raise NotImplementedError('actual exp is not implemented')


    geodesic = exp


    def log(self, x1, x2, dt=1.):
        """
        geodesic mapping points on manifold to the tangent space of the first point
        log_p: M -> T_p M
        """
        raise NotImplementedError('actual log is not implemented')


    def retraction(self, x, v):
        """
        approximation to exponential geodesic mapping
        project new vector x+v from point x to the manifold
        """
        raise NotImplementedError('actual retraction is not implemented')


    def inverse_retraction(self):
        """
        approximation to logrithrim geodesic mapping
        project points to get the connecting tangent vector at first point
        """
        raise NotImplementedError('actual inverse_retraction is not implemented')


    def projection(self, x, v):
        """
        project a general matrix v to the tangent space of a point x on the manifold
        """
        raise NotImplementedError('actual projection is not implemented')


    to_tangent_space = projection


    #TODO: any difference between vector transport and parallel transport?
    def transport(self, x0, x1, v0):
        """
        vector or parallel transport moves tangent vector v0 at point x0 to point x1
        `T_{x0->x1} (v0): v0 \in T_{x0} M -> v1 \in T_{x1} M = Proj_M (x1, v0)`
        here x0 is a dummy variable
        project v0 to the tangent space of x1
        """
        return self.projection(x1, v0)


    #TODO: on same point or two points?
    def weingarten(self, x, v, *args, **kwargs):
        """
        connection between tangent vectors
        """
        raise NotImplementedError('actual weingarten is not implemented')


    def riemannian_gradient(self, x, grad):
        """
        project euclidean gradient to riemannian gradient of a function
        """
        return self.projection(x, grad)


    def riemannian_hessian(self, x, v, grad, hess, normal=None):
        """
        project euclidean hessian to riemannian hessian of a function
        """
        if normal is None:
            # euclidean_gradient - riemannian_gradient
            normal = grad - self.projection(x, grad)

        return self.projection(x, hess) - self.weingarten(x, v, grad, normal)


    def gradient_descent(self, method=None, nmax=50, thresh=1e-8,
                         *args, **kwargs):
        if method is None: # assign a fixed suitable size by default
            method = 1. / (2. * np.linalg.norm(self._A))

        if kwargs.get('ymin') == 'eigenvalue':
            kwargs['ymin'] = self.eigenvalue[0]

        return gradient_descent(self.func, self.func_grad, self.retraction,
                                self.x0, method, nmax, thresh, *args, **kwargs)


    def conjugate_gradient(self, method=None, cg_method='fletcher_reeves',
                           nmax=50, thresh=1e-8, *args, **kwargs):
        if method is None: # assign a fixed suitable size by default
            method = 1. / (2. * np.linalg.norm(self._A))

        if not hasattr(self, 'preconditioner'):
            self.preconditioner = None

        return conjugate_gradient(self.func, self.func_grad, self.retraction,
                                self.transport, self.preconditioner,
                                self.x0, method, cg_method, nmax, thresh,
                                *args, **kwargs)



# TODO: fix the N*N case
class OrthogonalGroup(Riemannian): # orthogonal group manifold
    """
    the points are n*n square matrix, n-dimensions and n-planes
    """
    def check_sanity(self):
        ndim = self.ndim
        if len(ndim)==2 and ndim[0]!=ndim[1]:
            raise ValueError('points on OrthogonalGroup should be square matrix')

    @property
    def eigenvalue(self):
        A = self._A
        e, v = np.linalg.eigh(A)

        arg = np.argsort(e)
        e, v = e[arg], v[:,arg]
        return e


    # x0 is nd square matrix
    def func(self, x): # <x|A|x> / <x|x>
        """assume x is normalized"""
        A = self._A
        return np.sum(np.einsum('m...,mn,n...->...', x, A, x))


    def func_grad(self, x): # 2 (I - |x><x|) A|x> = 2 (A - <x|A|x>I ) |x>
        # the returned grad has been projected on the tangent space
        """assume x is normalized"""
        A = self._A
        #x /= np.linalg.norm(x) # put it on sphere
        Ax = 2.* (A @ x) #np.einsum('mn,n...->m...', A, x)
        return self.projection(x, Ax)


    def projection(self, x, v):
        #return (v - x * np.sum(np.einsum('m...,m...->...', x, v)))
        return (v - x * np.dot(x, v))


    def retraction_norm(self, x, v, dt=1.):
        """put (x+v) on the sphere"""
        x = x + v*dt
        return x / np.linalg.norm(x, axis=0)


    def retraction_qr(self, x, v, dt=1.):
        #TODO: fix this bug!!!
        x += x @ x
        q, r = np.linalg.qr(x)
        return q


    def retraction_polar(self, x, v, dt=1.):
        return x


    def retraction_cayley(self, x, v, dt=1.):
        #TODO: where is the step size
        I = np.eye(x.shape[0])
        v *= .5
        m = np.linalg.inv((I - v))
        p = I + v
        return np.einsum('ij,jk,kl->il', m, p, x)


    def exp(self, x, v, dt=1.):
        return scipy.linalg.expm(v*dt) @ x


    #TODO: what about log?
    def log(self, x1, x2, dt=1.):
        return



class Stiefel(OrthogonalGroup):
    """
    `St(k,n) = {p \in F^{n*k} | p^\dagger B p = I_k}`
    the point p is a n*k matrix, n-dimensions and k-planes
    quotient space from OrthogonalGroup where k<n
    whose tangent vector V is on the tangent space of point p
    `T_p St(k,n) = {V \in F^{n*k} | p^\dagger B V + V^\dagger B p = 0_k}`
    """
    def check_sanity(self):
        ndim = self.ndim
        if len(ndim)==1 or (len(ndim)==2 and ndim[0]<=ndim[1]):
            raise ValueError('points on Stiefel should be long rectagular matrix')


    @property
    def dimension(self, dtype=float):
        n, k = self.ndim
        if dtype is float: # \mathbb{R}
            return int(n*k - k*(k+1)/2)
        elif dtype is complex: # \mathbb{C}
            return int((2*n - 1)*k)
        else: # quaternion \mathbb{H}
            return int((4*n - 2*k +1)*k)


    #TODO: find the distance for T_p St
    @property
    def dist(self, v1, v2):
        return


    def projection(self, x, v):
        """
        `Proj(p,V) = V - p Sym(p^\dagger B V)`
        """
        tmp = x.conj().T @ self._B @ v
        tmp = .5 * (tmp + tmp.conj().T)
        return (v - x @ tmp)


    def retraction_qr(self, x, v, dt=1.):
        """
        `Retr_p V = QD where QR = qr(p+V) and D=diag(sgn(diag(R)+.5))`
        """
        x = x + v*dt
        #TODO: compare this with pymanopt
        #return pymanopt.tools.multi.multiqr(x)[0]
        q, r = np.linalg.qr(x)
        d = np.sign(np.diag(r)+.5)
        return np.einsum('ij,j->ij', q, d)


    def retraction_polar(self, x, v, dt=1.):
        """
        `Retr_p V = U V^\dagger where UsV^\dagger = svd(p+V)`
        """
        x = x + v*dt
        u, s, vt = np.linalg.svd(x, full_matrices=False)
        return u @ vt


    #TODO: implement qr method
    def inverse_retraction_qr(self, x1, x2, dt=1.):
        return


    def inverse_retraction_polar(self, x1, x2, dt=1.):
        pt = x1.conj().T @ x2
        #TODO: it seems identity should be replaced by overlap metric
        v = solve_sylvester(pt, 2.*np.eye(x1.shape[1]))
        return (x2 @ (v*dt) - x1)


    def weingarten(self, x, v, grad, normal):
        # grad is not used
        tmp = v.T @ normal
        tmp = .5 * (tmp + tmp.T)

        w = v @ x.T @ normal
        w += x @ tmp
        return w


    #TODO: where is the time dt?
    def exp(self, x, v, dt=1.):
        """
        `exp_p V = (p \\ V) exp((p^\dagger V & - V^\dagger V \\ I_n & p^\dagger V)) (exp(-p^\dagger V) \\ 0_n)`
        """
        pt_tv = x.conj().T @ v
        identity = np.eye(self.ndim[1]) # column dimension

        a = np.block([x, v])
        b = np.block([[pt_tv, -v.conj().T @ v,],
                      [identity, pt_tv],])
        b = scipy.linalg.expm(b)[..., : self._p] #TODO: fix undefined self._p I guess it is k
        c = scipy.linalg.expm(-pt_tv)
        return a @ (b @ c)


    #TODO: can we find log for stiefel?
    def log(self, x1, x2, dt=1.):
        return



class Grassmann(Stiefel):
    """
    Grassmann is a quotient space from Stiefel
    with extra idempotency condition for density matrix P = p p^\dagger
    this class uses point p rather than P
    `Gr(k,n) = St(k,n) / O(k)
             = {p \in F^{n*k} | p^\dagger B p = I_k, PBP = P}
    T_p Gr(k,n) = {V \in F^{n*n} | PV + VP = V}`
    """
    @property
    def dimension(self, dtype=float):
        n, k = self.ndim
        d = (n-k)*k
        c = 1 if dtype is float else 2 if dtype is complex else 4
        return int(c*d)


    @property
    def dist(self, v1, v2):
        s = np.linalg.svd((v1.conj().T @ self._B @ v2), compute_uv=False)
        s[s > 1] = 1
        s = np.arccos(s)
        return np.linalg.norm(s)


    def projection(self, x, v):
        """
        the resulted tangent vector is perpendicular to BX
        `<\Delta, BX> = 0`
        """
        #p = x @ x.conj().T # projector, aka `density matrix`
        #return (self._Binv - p) @ v
        #TODO: which one is correct?
        p = x @ x.conj().T @ self._B
        return (v - p @ v)


    #TODO: should these be same?
    def retraction_qr(self, x, v, dt=1.): # same as stiefel
        return


    def retraction_polar(self, x, v, dt=1.): # same as stiefel
        x = x + v*dt
        u, s, vt = np.linalg.svd(x, full_matrices=False)
        return u @ vt


    def inverse_retraction_polar(self, x1, x2, dt=1.):
        pt = x1.conj().T @ x2
        pt = np.linalg.inv(pt) # get inverse
        #TODO: double check!
        return (x2 @ (pt*dt) - x1)


    def weingarten(self, x, v, grad, normal):
        # normal is not used
        # grad is euclidean gradient
        return v @ (x.T @ grad)


    def exp(self, x, v, dt=1.):
        """
        `exp_p (V) = p V cos(s) V^\dagger + U sin(s) V^\dagger`
        qr is needed for numerically stablity
        """
        #TODO: why v.conj().T @ self._B @ v
        u, s, vt = np.linalg.svd(v, full_matrices=False)

        s *= dt
        cos = np.expand_dims(np.cos(s), -2)
        sin = np.expand_dims(np.sin(s), -2)

        y = (x @ (vt.conj().T * cos) + (u * sin)) @ vt

        # it seems necessary to re-orthonormalize numerically
        # even though it is quite expensive.
        #TODO: compare it with pymanopt
        #q, _ = pymanopt.tools.multi.multiqr(y)
        q, _ = np.linalg.qr(y)
        return q


    def log(self, x1, x2, dt=1.):
        ytx = x2.conj().T @ self._B @ x1
        At = x2.conj().T - ytx @ x1.conj().T

        Bt = np.linalg.solve(ytx, At)
        u, s, vt = np.linalg.svd(Bt.conj().T, full_matrices=False)

        s *= dt
        arctan = np.expand_dims(np.arctan(s), -2)
        return (u * arctan) @ vt


    def tangent_solver(self, x, grad, hess, method='direct'):
        if method == 'direct':
            v = solve_sylvester(hess, (-x.T@grad), -self.projection(grad))
            return v


    def newton_2nd(self, gradf, hessf, x0, dt=1., nmax=50, thresh=1e-8):
        def gradf(x):
            return 2.* (self._A @ x)
        def hessf():
            return self._A

        return newton_2nd(self.func, gradf, hessf, self.tangent_solver,
                          self.geodesic, x0, dt, nmax, thresh)



if __name__ == '__main__':
    import pymanopt

    ndim = 6
    A = get_random_matrix(ndim)

    nmax = 60

    itype = 'sphere'
    #itype = 'stiefel'

    cg_method = 'fletcher_reeves'

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

    xs, ys = surf_obj.conjugate_gradient(method='backtracking', cg_method=cg_method, nmax=nmax)

    problem = pymanopt.Problem(surf_obj2, cost)

    #optimizer_name = 'SteepestDescent'
    optimizer_name = 'ConjugateGradient'
    optimizer = getattr(pymanopt.optimizers, optimizer_name)(
                        max_iterations=nmax, verbosity=0, beta_rule='FletcherReeves')
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
