# import pymc3 as pm
# import numpy as np
# from matplotlib import pyplot as plt
#
# # created a model with Bernoulli samples of g1, g2, g3 and
# # Normal samples of x1, x2, x3
# with pm.Model() as model:
#     # g1 = 1 with probability 0.5
#     g1 = pm.Bernoulli('g1', 0.5)
#     #g2 and g3 are conditioned on g1 as given
#     g2 = pm.Bernoulli('g2', pm.math.switch(g1, 0.1, 0.9))
#     g3 = pm.Bernoulli('g3', pm.math.switch(g1, 0.1, 0.9))
#
#     # 3 means are created conditioned over respective Gs
#     mu1 = pm.math.switch(g1, 50, 60)
#     mu2 = pm.math.switch(g2, 50, 60)
#     mu3 = pm.math.switch(g3, 50, 60)
#
#     #Xs are sampled conditioned over respective Gs
#     x1 = pm.Normal('x1 given g1=1', mu=mu1, sd=np.sqrt(10))
#     x2 = pm.Normal('x2 given g2=1', mu=mu2, sd=np.sqrt(10), observed=50)
#     x3 = pm.Normal('x3 given g3=1', mu=mu3, sd=np.sqrt(10))
#
# with model:
#     # with created model extract Metropolis 20000 samples
#     step = pm.Metropolis()
#     samples = pm.sample(20000, tune=1000, step=step, cores=2)
#     print(pm.summary(samples))
#
# count_g1 = 0
# # All of G1 was sampled, out of it G1 = 2 values are extracted
# for i in samples.get_values(model.g1):
#     if i == 1:
#         count_g1 += 1
# # printing conditional probability
# print('P(G1 = 2 | X2 = 50) = '+str(float(count_g1)/float(len(samples.get_values(model.g1)))))
# # traceplot to see the sampling of all the variables
# pm.traceplot(samples)
# plt.show()
#
#

import pymc3 as pm
import numpy as np
from matplotlib import pyplot as plt

# created a model with Bernoulli samples of g1, g2, g3 and
# Normal samples of x1, x2, x3
with pm.Model() as model:
    # g1 = 1 with probability 0.5
    g1 = pm.Bernoulli('g1', 0.5)
    #g2 and g3 are conditioned on g1 as given
    g2 = pm.Bernoulli('g2', pm.math.switch(g1, 0.1, 0.9))
    g3 = pm.Bernoulli('g3', pm.math.switch(g1, 0.1, 0.9))

    # 3 means are created conditioned over respective Gs
    mu1 = pm.math.switch(g1, 50, 60)
    mu2 = pm.math.switch(g2, 50, 60)
    mu3 = pm.math.switch(g3, 50, 60)

    #Xs are sampled conditioned over respective Gs
    x1 = pm.Normal('x1', mu=mu1, sd=np.sqrt(10))
    x2 = pm.Normal('x2', mu=mu2, sd=np.sqrt(10), observed=50)
    x3 = pm.Normal('x3', mu=mu3, sd=np.sqrt(10))

with model:
    # with created model extract Metropolis 20000 samples
    step = pm.Metropolis()
    samples = pm.sample(2000, tune=1000, step=step, cores=2)
    print(pm.summary(samples))

count_g1 = 0
# All of X3 was sampled, out of it X3 = (49.5, 50.5) values are extracted
for i in samples['x3']:
    if i > 49.5 and i < 50.5:
        count_g1 += 1
# printing conditional probability
print('P(X3 = 50 | X2 = 50) = '+str(float(count_g1)/float(len(samples.get_values(model.x3)))))
# traceplot to see the sampling of all the variables
pm.traceplot(samples)
plt.show()


