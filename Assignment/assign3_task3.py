import numpy as np

def gaussian(x, sig, mu):
    const = 1/ np.sqrt(2 * np.pi * (sig))
    power = -1 * ((x - mu)**2) / (2 * (sig))
    gaus = const * (np.e**power)
    return gaus

def get_dir(r,sig, prior):
    x_dir = [1, -1, 0, 0]
    y_dir = [0, 0, 1, -1]
    directions = list()
    for d in range(4):
        prod = 1
        for rs in range(len(r)):
            if rs%2 == 0:
                prod = prod * gaussian(r[rs], sig=sig, mu=x_dir[d])
            else:
                prod = prod * gaussian(r[rs], sig=sig, mu=y_dir[d])
        directions.append(prod * prior[d])
    return directions

print(gaussian(50, 10, 50))
print(gaussian(50, 10, 60))
# rx = 0.75
# ry = -0.6
# bx = 1.4
# by = -0.2
# r = [rx,ry,bx,by]
# sig = 1
# prior = [0.25, 0.25, 0.25, 0.25]
# # prior = [0.125, 0.125, 0.125, 0.625]
# dir = get_dir(r, sig, prior)
# norm_dir = list()
# for d in dir:
#     norm_dir.append(d/sum(dir))
# print(norm_dir)
#
# """
# Answers
# sig = 1, prior = [0.25, 0.25, 0.25, 0.25]
# [0.7546323907671381, 0.01023927412664219, 0.039497237431914536, 0.19563109767430517]
# Maximum probability is 0.75 for 'right' direction.
#
# sig = 5, prior = [0.25, 0.25, 0.25, 0.25]
# [0.27187896101410924, 0.22891642279657057, 0.24161775944220235, 0.2575868567471178]
# Maximum probability is 0.27 for 'right' direction.
#
# sig = 1, prior = [0.125, 0.125, 0.125, 0.625]
# [0.42335038707210587, 0.0057442547098259765, 0.022158034772509057, 0.548747323445559]
# Maximum probability is 0.54 for 'down' direction.
#
# sig = 5, prior = [0.125, 0.125, 0.125, 0.625]
# [0.1339076048759674, 0.11274741443444124, 0.11900315986835013, 0.6343418208212412]
# Maximum probability is 0.63 for 'down' direction.
# """
#
