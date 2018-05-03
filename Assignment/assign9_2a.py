import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from collections import namedtuple

gaussian = namedtuple('Gaussian', ['mean', 'var'])
gaussian.__repr__ = lambda s: '𝒩(μ={:.3f}, 𝜎²={:.3f})'.format(s[0], s[1])


iterations = 200
t = [i for i in range(iterations)]
y1 = [0 for i in range(iterations)]
y2 = [0 for i in range(iterations)]
z = [0 for i in range(iterations)]

y1[0] = np.random.normal(5, 1)
y2[0] = np.random.normal(10, 2)
z[0] = np.random.normal(10, 2)

y1_t = [0 for i in range(iterations)]
y2_t = [0 for i in range(iterations)]
z_t = [0 for i in range(iterations)]

y1_t[0] = np.random.normal(5, 1)
y2_t[0] = np.random.normal(10, 2)
z_t[0] = np.random.normal(10, 2)
pos_mean = 5
pos_var = 1
vel_mean = 10
vel_var = 1
z_mean = 5
z_var = 1
current_time = 0

def LDS(a1, a2):
    # # Generate y1
    # for i in range(2,iterations):
    #     y1[i] = a1*y1[i-1] + a2*y1[i-2] + 2*np.random.normal(loc=0, scale=1)
    # # Generate y2
    # for i in range(1, iterations):
    #     y2[i] = y1[i-1] + np.random.normal(loc=0, scale=1)
    # # Generate z
    # for i in range(iterations):
    #     z[i] = y1[i] + np.random.normal(loc=0, scale=1)
    for i in range(1, iterations):
        y1[i] = a1*y1[i-1] + a2*y2[i-1] + np.random.normal(loc=0, scale=1)
        y2[i] = y1[i - 1] + np.random.normal(loc=0, scale=1)
        z[i] = y1[i] + np.random.normal(loc=0, scale=1)
    return y1, y2, z

def switchedLDS(a1, a2):
    i = current_time
    y1[i] = a1*y1[i-1] + a2*y2[i-1] + np.random.normal(loc=0, scale=1)
    y2[i] = y1[i - 1] + np.random.normal(loc=0, scale=1)
    z[i] = y1[i] + np.random.normal(loc=0, scale=1)
    return y1, y2, z


def plotLine(y1, y2, z):
    t = [i for i in range(iterations)]
    # plt.plot(t, y1, label='y1')
    # plt.plot(t, y2, label='y2')
    plt.plot(t, z, label='z')
    plt.legend()
    plt.show()


def plotScatter(y1, y2, z):
    t = [i for i in range(iterations)]
    # plt.scatter(t, y1, label='y1')
    # plt.scatter(t, y2, label='y2')
    plt.scatter(t, z, label='z')
    plt.legend()
    plt.show()


def predict(pos_mean, movement_mean, pos_var, movement_var):
    return pos_mean + movement_mean, pos_var + movement_var


def gaussian_multiply(g1_mean, g1_var, g2_mean, g2_var):
    mean = (g1_var * g2_mean + g2_var * g1_mean) / (g1_var + g2_var)
    variance = (g1_var * g2_var) / (g1_var + g2_var)
    return mean, variance


def update(prior_mean, prior_var, likelihood_mean, likelihood_var):
    posterior_mean, posterior_var = gaussian_multiply(likelihood_mean, likelihood_var, prior_mean, prior_var)
    return posterior_mean, posterior_var



# y1_t, y2_t = np.meshgrid(y1, y2)
# z_t = [[0 for i in range(iterations)] for j in range(iterations)]
# for i in range(iterations):
#     for j in range(iterations):
#         z_t[i][j] = y1_t[i][j] + np.random.normal(loc=0, scale=1)
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.plot_wireframe(X=y1_t,Y=y2_t,Z=z_t,rstride=3, cstride=3, linewidth=1, antialiased=True)
# ax.set_xlabel('y1')
# ax.set_ylabel('y2')
# ax.set_zlabel('z')
# plt.show()

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(y1_t, y2_t, z, rstride=3, cstride=3, linewidth=1, antialiased=True, cmap=cm.viridis)
# cset = ax.contourf(y1_t, y2_t, z, zdir='z', offset=-0.15, cmap=cm.viridis)
# cb = fig.colorbar(cset, shrink=0.5)
# plt.show()

"""
(2-a) Creating linear dynamical system LDS(a1, a2) 
"""
## plotting each signal
#
# y1, y2, z = LDS(a1=-0.5, a2=-0.5)
# plotLine(y1, y2, z)

# 0.5, 0.5 - range(-50, 50)
# 0.5, -0.5 - range(-5, 5)
# -1.5, 1.5 - range(0)

"""
(2-b) Switched linear dynamical system
"""
alphas = [[0.5, 0.1], [1.0, 0.1], [0.0, 0.1]]
choice = 0
colors=["#0000FF", "#00FF00", "#FF0066"] # state1 = blue, state2 = green, state3 = red
# use beta to decide the transition probability
transition_prob = [[0.8, 0.1, 0.1],
                   [0.1, 0.8, 0.1],
                   [0.1, 0.1, 0.8]]
for i in range(1, iterations):
    current_time = i
    # choice = np.random.multinomial(1, transition_prob[choice]).argmax()
    if i % 5 == 0:
        choice = np.random.choice([0,1,2])
    y1, y2, z = switchedLDS(alphas[choice][0], a2=alphas[choice][1])
    plt.subplot(2,1,1)
    plt.scatter(t[i], z[i], label='z', color=colors[choice])
plt.subplot(2,1,2)
plt.plot(t, z)
plt.title("3 state switched linear dynamical model - state1 = blue, state2 = green, state3 = red")
plt.xlabel("time")
plt.ylabel("observed values (z)")
plt.show()

"""
(2-c) Kalman Filter
"""
# for i in range(1, iterations):
#     y1[i] = -0.5 * y1[i - 1] + -0.5 * y2[i - 1] + np.random.normal(loc=0, scale=1)
#     y2[i] = y1[i - 1] + np.random.normal(loc=0, scale=1)
#     z[i] = y1[i] + np.random.normal(loc=0, scale=1)
#
#     prior_mean, prior_var = predict(pos_mean=pos_mean, pos_var=pos_var, movement_mean=vel_mean, movement_var=vel_var)
#     pos_mean = prior_mean
#     pos_var = prior_var
#     z_mean, z_var = update(pos_mean, pos_var, z_mean, z_var)
#     z_t[i] = np.random.normal(loc=z_mean, scale=z_var)
#
#     if i % 20 == 0:
#         print("\nactual positions")
#         print("y1 " + str(y1[i]))
#         print("y2 " + str(y2[i]))
#         print("z " + str(z[i]))
#
#         print("predicted positions")
#         print("y1 " + str(y1_t[i]))
#         print("y2 " + str(y2_t[i]))
#         print("z " + str(z_t[i]))