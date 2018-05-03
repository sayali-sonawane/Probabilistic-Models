from collections import namedtuple
import filterpy.stats as stats
import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt
from filterpy.kalman import predict, update
from scipy.linalg import solve
import math

gaussian = namedtuple('Gaussian', ['mean', 'var'])
gaussian.__repr__ = lambda s: '𝒩(μ={:.3f}, 𝜎²={:.3f})'.format(s[0], s[1])


# def predict(pos, movement, acceleration):
#     return gaussian(pos.mean + movement.mean + acceleration.mean, pos.var + movement.var + acceleration.var)


def gaussian_multiply(g1, g2):
    mean = (g1.var * g2.mean + g2.var * g1.mean) / (g1.var + g2.var)
    variance = (g1.var * g2.var) / (g1.var + g2.var)
    return gaussian(mean, variance)

#
# def update(likelihood, prior):
#     posterior = likelihood * prior
#     return posterior

a1 = 0.5
a2 = 0.1

# z = np.random.normal(loc=5, scale=3) # x
# y1 = np.random.normal(loc=10, scale=2) # x'
# y2 = np.random.normal(loc=5, scale=1) # x''

x = np.array([5., 10, 5])
P = np.diag([3, 2, 1])

F = np.array([[0, 1, 0],
              [0, a1, a2],
              [0, 1, 0]])

x, P = predict(x=x, P=P, F=F, Q=0)
from filterpy.common import Q_discrete_white_noise
Q = Q_discrete_white_noise(dim=3, dt=1., var=2.35) # 3 x 3
H = np.array([[1.0, 0, 0]])

R = np.diag([[1., 0, 0]])

from filterpy.kalman import update
z = 1.
# x, P = update(x, P, z, R, H)

from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise


def pos_vel_filter(x, P, R, Q=0., dt=1.0):
    """ Returns a KalmanFilter which implements a
    constant velocity model for a state [x dx].T
    """

    kf = KalmanFilter(dim_x=3, dim_z=1)
    kf.x = x  # location and velocity
    kf.F = F # state transition matrix
    kf.H = H # Measurement function
    kf.R *= R  # measurement uncertainty
    if np.isscalar(P):
        kf.P *= P  # covariance matrix
    else:
        kf.P[:] = P  # [:] makes deep copy
    if np.isscalar(Q):
        kf.Q = Q
    else:
        kf.Q[:] = Q
    return kf

iterations = 200

def compute_dog_data():
    "returns track, measurements 1D ndarrays"

    t = [i for i in range(iterations)]
    y1 = [0 for i in range(iterations)]
    y2 = [0 for i in range(iterations)]
    z = [0 for i in range(iterations)]

    y1[0] = np.random.normal(5, 1)
    y2[0] = np.random.normal(10, 2)
    z[0] = np.random.normal(10, 2)

    for i in range(1, iterations):
        y1[i] = a1*y1[i-1] + a2*y2[i-1] + np.random.normal(loc=0, scale=1)
        y2[i] = y1[i - 1] + np.random.normal(loc=0, scale=1)
        z[i] = y1[i] + np.random.normal(loc=0, scale=1)
    return z

def run(x0, P, R, Q, dt=1.0,track=None, zs=None, count=10, do_plot=True, **kwargs):
    """
    track is the actual position of the dog, zs are the
    corresponding measurements.
    """

    # Simulate dog if no data provided.
    if zs is None:
        zs = compute_dog_data()

    # create the Kalman filter
    kf = pos_vel_filter(x0, R=R, P=P, Q=Q, dt=dt)

    # run the kalman filter and store the results
    xs, cov = [], []
    for z in zs:
        kf.predict()
        kf.update(z)
        xs.append(kf.x[0])
        cov.append(kf.P[0])

    xs, cov = np.array(xs), np.array(cov)
    return xs, cov, zs

Ms, Ps, Zs = run(x0=(0., 0, 0),count=iterations, R=R, Q=Q, P=P, dt=1.0)
plt.plot(range(iterations), Zs, label='actual')
plt.plot(range(iterations), Ms, label='predicted')
plt.legend()
plt.show()