from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import norm


alpha = 0.5

# Task 1
stupid_prob = [1]
for n in range(1,500):

    # new table
    prob_new = alpha / float(n + alpha)
    stupid_prob.append(prob_new)
x = [i+1 for i in range(500)]
plt.plot(x, stupid_prob)
plt.show()

# Task 2
clusters = [1]
cluster_prob = [1]
points = []
cluster_points = {}


def sampleAxis():
    x = np.random.uniform(0,1)
    y = np.random.uniform(0,1)
    return (x, y)


def update_cluster_prob(clusters, n):
    cluster_prob = []
    for i in range(len(clusters)):
        cluster_prob.append(clusters[i] / float(n + 0.5))
    return cluster_prob



mean = sampleAxis()
points.append(mean)
cluster_points[0] = [mean]
covar = [[0.01, 0], [0, 0.01]]

for n in range(1,500):

    # new table
    prob_new = alpha / float(n + alpha)

    rand_int = np.random.rand()

    if rand_int < prob_new:
        # choose new table
        mean = sampleAxis()
        clusters.append(1)
        l = len(cluster_points)
        cluster_points[l] = list()
        cluster_points[l].append(mean)
    else:
        # choose old table
        k = np.random.multinomial(1, cluster_prob).argmax()
        mean = cluster_points[k][0]
        out = np.random.multivariate_normal([mean[0], mean[1]], covar)
        clusters[k] += 1
        cluster_points[k].append(out)

    cluster_prob = update_cluster_prob(clusters, n+1)

x_axis = []
y_axis = []
color = []
for key, value in cluster_points.items():

    for v in value:
        x_axis.append(v[0])
        y_axis.append(v[1])
        color.append(key)

plt.scatter(x_axis, y_axis, c=color)
plt.legend()
plt.show()