import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm



def p_x_given_y(y, mus, sigmas):
    mu = mus[0] + sigmas[1][0] / sigmas[1][1] * (y - mus[1])
    sigma = sigmas[0][0] * (1 - sigmas[0][1]**2/ (sigmas[0][0]*sigmas[1][1]))
    return np.random.normal(mu, np.sqrt(sigma))


def p_y_given_x(x, mus, sigmas):
    mu = mus[1] + (sigmas[0][1] / sigmas[0][0] * (x - mus[0]))
    sigma = sigmas[1][1] * (1 - sigmas[0][1]**2/ (sigmas[0][0]*sigmas[1][1]))
    return np.random.normal(mu, np.sqrt(sigma))


def gibbs_sampling(mus, sigmas, itera):
    samples = np.zeros((itera, 2))
    y = -4.0
    x = -3.0
    for i in range(itera):
        x = p_x_given_y(y, mus, sigmas)
        y = p_y_given_x(x, mus, sigmas)
        samples[i, :] = [x, y]

    return samples


mu = [1,0]
sigma = [[1, -0.5], [-0.5, 3]]
iteration = 15000
samples = gibbs_sampling(mu, sigmas=sigma, itera=iteration)
x = np.linspace(-10,10,iteration)
plt.subplot(1,2,1)
plt.hist(samples[:,1], normed=True, bins=40)
plt.plot(x, norm.pdf(x, loc=0, scale=np.sqrt(3)))

plt.subplot(1,2,2)
plt.hist(samples[:,0], normed=True, bins=40)
plt.plot(x, norm.pdf(x, loc=1, scale=1))

plt.show()



