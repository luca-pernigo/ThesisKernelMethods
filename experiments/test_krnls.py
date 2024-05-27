# test comparing kernels from two different sklearn libraries

import numpy as np

from sklearn.metrics.pairwise import rbf_kernel, laplacian_kernel, linear_kernel
from sklearn.gaussian_process.kernels  import Matern, RBF
# find when they are the same

g=1/10

# matern = 1.0 * Matern(length_scale=1/(np.sqrt(2*g)), nu=np.inf)

# X = [[3, 1, 2,4], [1, 2, 1,5],[1,5,32,2]]
# Y = [[1, 12,24 ,0], [1, 1,12 ,0], [1,5,0,6]]

# print("rbf kernel","\n", rbf_kernel(X,Y, gamma=g))

# print("matern_inf kernel","\n", matern(X,Y))


# # print("*******")
matern_05 = 1.0 * Matern(length_scale=1/g, nu=0.5)


# print("laplacian kernel","\n", laplacian_kernel(X,Y, gamma=g))

# print("matern_05 kernel","\n", matern_05(X,Y))


# when X is 1d a_laplacian and matern are the same
X = [[3], [1],[1]]
Y = [[1], [12], [16]]
print("laplacian kernel","\n", laplacian_kernel(X,Y, gamma=g))

print("matern_05 kernel","\n", matern_05(X,Y))