import numpy as np
from matplotlib import pyplot as plt

def getProb(x):
    if x > 1 or x < 0:
        return 0
    else:
        return x**3

def getConditional(g,f):
    if g > 1 or g < 0:
        return 0
    else:
        ans = 1 - abs(g-f)
        return ans

def generateRandSample():
    x = np.random.rand()
    return x

def genUniform(x):
    s = 0.1
    n = 2
    while (n < 0 or n > 1):
        n = np.random.uniform(x-s, x+s)
    return n


def getAcceptance(f,g):
    f_prob = getProb(f)
    g_prob = getProb(g)
    div = (f_prob) / (g_prob)
    return min(1,div)

def acceptanceJoint(f, f_star, g, g_star):
    f_given_g = getConditional(g_star, f_star)
    g_given_f = getConditional(g, f)
    return min(1, f_given_g/g_given_f)

def calcMetro():
    iterations = 100000
    x = [0]*iterations
    x[0] = generateRandSample()
    for t in range(1,iterations):
        g = generateRandSample()
        f = genUniform(g)
        A = getAcceptance(f,g)
        n_rand = generateRandSample()
        if n_rand <= A:
            x[t] = getConditional(f,g)
        else:
            x[t] = x[t-1]
    plt.hist(x)
    plt.show()

def calcMH():
    iterations = 10000000
    x = [0] * iterations
    x_star_list = [0]*iterations
    x_star_list[0] = 0.9
    x[0] = 0.9
    for t in range(0, iterations-1):
        n_rand = generateRandSample()
        x_star = genUniform(x[t])
        x_star_list[t] = x_star
        A = getAcceptance(x_star,x[t])
        if n_rand <= A:
            x[t+1] = x_star
        else:
            x[t+1] = x[t]
    x_star_list[iterations-1] = x_star_list[iterations-2]
    return x, x_star_list

def calcJoint():
    f, f_star = calcMH()
    iterations = 10000000
    x = [0] * iterations
    x_star_list = [0] * iterations
    x_star_list[0] = 0.9
    x[0] = 0.9
    for t in range(0, iterations - 1):
        n_rand = generateRandSample()
        x_star = genUniform(x[t])
        x_star_list[t] = x_star
        A = acceptanceJoint(f[t], f_star[t], x[t], x_star)
        if n_rand <= A:
            x[t + 1] = x_star
        else:
            x[t + 1] = x[t]
    x_star_list[iterations-1] = x_star_list[iterations-2]
    sum = 0
    for i in range(iterations):
        sum += f[i]*x[i]
    expectation = sum/iterations
    print(expectation)
    plt.hist2d(x,f, bins=20)
    plt.show()



hist = calcJoint()
