from scipy.stats import beta
import numpy as np
import matplotlib.pyplot as plt

def gen_post(h,t,prior):
    # Likelihood distribution for coin toss follows the pattern of beta function.
    # Returning continuous posterior using beta function
    x = np.linspace(0.01, 0.99, 100)
    out = beta.pdf(x, a=h+1, b=t+1)
    return out*prior

# Hypotheses are discrete varying from 0.0 to 1.0. We are considering uniform
# prior. So prior probability will be 1/11 for all priors.
models = [i for i in range(0,1,11)]
prior = 1/len(models)

h = 0
t = 0
x = np.linspace(0.01, 0.99, 100)
seq = [1, 0, 0, 1, 0, 0, 0, 1 ]
c = 1
plt.subplot(3, 3, c)
plt.plot(x, [prior]*len(x), label='priors')
plt.legend()

for i in seq:
    c += 1
    if i == 1:
        h += 1
    else:
        t += 1
    posterior = gen_post(h, t, prior)
    plt.subplot(3, 3, c)
    plt.plot(x, posterior, label='trial %d' %(c-1) + ': h=%d' %h +' t=%d'%t)
    plt.legend()
    plt.xlabel('Hypothesis')
    plt.ylabel('Beta posterior probability')
plt.show()
