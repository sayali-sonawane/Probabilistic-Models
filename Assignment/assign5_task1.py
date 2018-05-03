from scipy.stats import gamma
from scipy.stats import norm
import numpy as np

def getIntelligence():
    mean = 100
    sd = 15
    n = np.random.normal(loc=mean, scale=sd)
    return n

def getMajor(intelligence):
    rand_num = np.random.rand()
    m = 1.0/(1.0 + np.exp(-(intelligence - 110.0)/5.0))
    if rand_num < m:
        return 1 # comp sci
    else:
        return 0 # business

def getUniv(intelligence):
    rand_num = np.random.rand()
    m = 1.0 / (1.0 + np.exp(-(intelligence - 100.0) / 5.0))
    if rand_num < m:
        return 1  # cu
    else:
        return 0  # metro

def getSalary(I, M, U):
    s = np.random.gamma(.1 * I + M + 3 * U,5)
    return s

def drawSamples(size):
    samples = list()
    for i in range(size):
        I = getIntelligence()
        M = getMajor(I)
        U = getUniv(I)
        S = getSalary(I=I, M=M, U=U)
        samples.append((I, M, U, S))
    return samples

def getPosterior(samples,salary):
    p_sum = 0
    p_sum_ = 0
    p_su_m = 0
    p_su_m_ = 0
    tot_weight = 0
    for sam in samples:
        I, M, U, S = sam
        p_sal = gamma.pdf(salary, .1 * I + M + 3 * U, scale=5)
        tot_weight += p_sal
        if U == 1 and M == 1:
            p_sum += p_sal
        if U == 1 and M == 0:
            p_sum_ += p_sal
        if U == 0 and M == 1:
            p_su_m += p_sal
        if U == 0 and M == 0:
            p_su_m_ += p_sal

    print("Comp Sci + CU = " + str(p_sum/tot_weight))
    print("Comp Sci + Metro = " + str(p_su_m/tot_weight))
    print("Business + CU = " + str(p_sum_/tot_weight))
    print("Business + Metro = " + str(p_su_m_/tot_weight))

size = 100000
samples = drawSamples(size)
getPosterior(samples,120)
print("\n")
getPosterior(samples,60)
print("\n")
getPosterior(samples, 20)