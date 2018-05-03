import matplotlib.pyplot as plt
import numpy as np
import math


exampleA1 = [
    [1,0,0,0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,0],
    [0,0,0,0,1,0,0,0,0,0],
    [0,0,0,0,1,0,0,0,0,0],
]
exampleA2 = [
    [0,1,0,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,0],
    [0,0,0,0,1,0,0,0,0,0],
    [0,0,0,0,1,0,0,0,0,0],
    [0,0,0,0,0,1,0,0,0,0],
    [0,0,0,0,0,1,0,0,0,0],
]
exampleB1 = [
    [1,0,0,0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,0],
    [0,0,0,0,1,0,0,0,0,0],
    [0,0,0,0,1,0,0,0,0,0],
]
exampleB2 = [
    [1,0,0,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,0],
    [0,0,0,0,1,0,0,0,0,0],
    [0,0,0,0,1,0,0,0,0,0],
    [0,0,0,0,0,1,0,0,0,0],
]
exampleC1 = [
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [1,1,1,1,1,1,0,0,0,0],
    [0,0,0,0,0,1,0,0,0,0],
    [0,0,0,0,0,1,0,0,0,0],
    [0,0,0,0,0,1,0,0,0,0],
    [0,0,0,0,0,1,0,0,0,0],
    [0,0,0,0,0,1,0,0,0,0],
    [0,0,0,0,0,1,0,0,0,0],
]
exampleC2 = [
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [1,1,1,1,1,1,1,1,0,0],
    [0,0,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,1,0,0],
]

def drawImage(image, cmap):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap=cmap)
    # draw gridlines
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    ax.set_xticks(np.arange(-0.5, 10, 1))
    ax.set_yticks(np.arange(-0.5, 10, 1))
    plt.show()

# im1 = np.array(exampleA1)*5 + np.array(exampleA2)*10
# im2 = np.array(exampleB1)*5 + np.array(exampleB2)*10
# im3 = np.array(exampleC1)*5 + np.array(exampleC2)*10
# cmap1 = colors.ListedColormap(['black', 'red', 'green'])
# cmap2 = colors.ListedColormap(['black', 'red', 'green', 'yellow'])
# image = im1
# cmap = cmap2
# if np.array_equal(np.array(image), np.array(im1)):
#     cmap = cmap1
# drawImage(image, cmap=cmap)

def velVector(exampleA1, exampleA2):
    velocity = [-2, -1, 0, 1, 2]

    outIm1 = [[0 for x in range(5)] for y in range(5)]

    for vx in velocity:
        for vy in velocity:
            sum = 0
            for i in range(10):
                for j in range(10):
                    if i + vy >= 0 and i + vy <= 9 and j + vx >= 0 and j + vx <= 9:
                        sum -= (exampleA1[i][j] - exampleA2[i+vy][j+vx])**2
            outIm1[vy + 2][vx + 2] = sum

    print(outIm1)
    plt.imshow(outIm1, cmap='hot',extent=[-2.5,2.5,-2.5,2.5])
    plt.colorbar()
    plt.show()

def velVectorWithPrior(exampleA1, exampleA2):
    velocity = [-2, -1, 0, 1, 2]

    outIm1 = [[0 for x in range(5)] for y in range(5)]

    for vx in velocity:
        for vy in velocity:
            sum = 0
            for i in range(10):
                for j in range(10):
                    if i + vy >= 0 and i + vy <= 9 and j + vx >= 0 and j + vx <= 9:
                        sum -= (exampleA1[i][j] - exampleA2[i+vy][j+vx])**2
            outIm1[vy + 2][vx + 2] = sum - 0.5*(vx**2 + vy**2)

    print(outIm1)
    plt.imshow(outIm1, cmap='hot',extent=[-2.5,2.5,-2.5,2.5])
    plt.colorbar()
    plt.show()

def velVectorWithVelocities(exampleA1, exampleA2):
    velocity = [-2, -1, 0, 1, 2]

    outIm1 = [[0 for x in range(5)] for y in range(5)]

    for vx in velocity:
        for vy in velocity:
            sum = 0
            for i in range(10):
                for j in range(10):
                    if i + vy >= 0 and i + vy <= 9 and j + vx >= 0 and j + vx <= 9:
                        sum -= ((exampleA1[i][j] - exampleA2[i+vy][j+vx])**2)
            outIm1[vy + 2][vx + 2] = math.exp(sum - 0.5*(vx**2 + vy**2))
    tot_sum = np.sum(outIm1)
    outIm2 = [[0 for x in range(5)] for y in range(5)]
    for vx in velocity:
        for vy in velocity:
            outIm2[vy + 2][vx + 2] = math.log(float(outIm1[vy + 2][vx + 2]) / float(tot_sum))

    print(outIm2)
    plt.imshow(outI                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 m2, cmap='hot',extent=[-2.5,2.5,-2.5,2.5])
    plt.colorbar()
    plt.show()

"""
Task1 Likelihood
"""
velVector(exampleA1=exampleA1, exampleA2=exampleA2)
velVector(exampleA1=exampleB1, exampleA2=exampleB2)
velVector(exampleA1=exampleC1, exampleA2=exampleC2)

"""
Task2 Posterior
"""
velVectorWithPrior(exampleA1=exampleA1, exampleA2=exampleA2)
velVectorWithPrior(exampleA1=exampleB1, exampleA2=exampleB2)
velVectorWithPrior(exampleA1=exampleC1, exampleA2=exampleC2)

"""
Task3 Normalized velocity prior
"""
velVectorWithVelocities(exampleA1=exampleA1, exampleA2=exampleA2)
velVectorWithVelocities(exampleA1=exampleB1, exampleA2=exampleB2)
velVectorWithVelocities(exampleA1=exampleC1, exampleA2=exampleC2)