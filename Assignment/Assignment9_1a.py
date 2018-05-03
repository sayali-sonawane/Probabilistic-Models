import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from bayes_opt import BayesianOptimization

# Our 2-dimensional distribution will be over variables X and Y
N = 100
X = np.linspace(-3, 3, N).reshape(-1, 1)
Y = np.linspace(-3, 4, N).reshape(-1, 1)
X, Y = np.meshgrid(X, Y)
# print(X.shape)



# Mean vector and covariance matrix
# mu1 = np.array([-1., 2.])
# Sigma1 = np.array([[ 1., -0.5], [-0.5,  1.5]])
#
# mu2 = np.array([0., -1.5])
# Sigma2 = np.array([[ 1., -0.5], [-0.5,  1.5]])
#
# mu3 = np.array([1., 0.])
# Sigma3 = np.array([[ 1., -0.5], [-0.5,  1.5]])


def multivariate_gaussian(x, y, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos.

    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.

    """
    # Pack X and Y into a single 3-dimensional array

    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma) # 1.25
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)
    out = np.exp(-fac / 2) / N
    return out


def target(x, y):
    x, y = np.meshgrid(x, y)
    return multivariate_gaussian(x, y, np.array([1., 0.]), np.array([[ 1., -0.5], [-0.5,  1.5]])) - \
           multivariate_gaussian(x, y, np.array([-1., 2.]), np.array([[ 1., -0.5], [-0.5,  1.5]])) - \
           1.4*multivariate_gaussian(x, y, np.array([0., -1.5]), np.array([[ 1., -0.5], [-0.5,  1.5]])) + \
           0.01*np.random.normal(loc=0, scale=1.0, size=(x.shape[0], y.shape[0]))


# The distribution on the variables X, Y packed into pos.
Z = target(X, Y)
print(Z.shape)
#
# Create a surface plot and projected filled contour plot under it.
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,
                cmap=cm.viridis)

cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap=cm.viridis)

# Adjust the limits, ticks and view angle
ax.set_zlim(-0.15,0.2)
ax.set_zticks(np.linspace(0,0.2,5))
ax.view_init(27, -21)

plt.show()

# bo = BayesianOptimization(target, {'x': (-3, 3), 'y': (-3, 4)})
# bo.maximize(init_points=2, n_iter=10, acq='ucb', kappa=5)
# print(bo.res['max'])