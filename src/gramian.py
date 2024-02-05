from sklearn.metrics.pairwise import pairwise_distances
import numpy as np

# kernel is just a kernel object from sklearn.gaussian_process.kernels

def eval_kernel(x,x0, kernel):
    # evaluates k(x,.)
    # the call to kernel returns the kernel matrix of k(x, x0)
    # therefore only the off diagonal element is of interest for us
    # for this reason the [0,1] slicing
    
    return kernel(np.array([[x],[x0]]))[0,1]


# function to compute gaussian Kernel matrix
# inputs:two numpy arrays
def gram(x,y):
    gram= np.exp(-pairwise_distances(x.reshape(-1,1), y.reshape(-1,1), metric='euclidean')**2/2)
    return gram