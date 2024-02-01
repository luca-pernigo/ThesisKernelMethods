from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
from src.PivCholesky import piv_chol
from src.Gramian import eval_kernel, gram

# kde computes approximate the probability density function at a specified x point
# suppose x=2, the kde function will return the kernel mean embedding estimate of p(2)
# x_sample is the dataset you have, whose underlying distribution is unknown
# kernel is just a kernel object from sklearn.gaussian_process.kernels
def kde(x_sample, x,kernel):
    ans=[eval_kernel(x_sample[i], x, kernel) for i in range(len(x_sample))]
    ans=np.sum(ans)/len(x_sample)
    return ans




def nyström_kde(x_sample, x_tilde, x,kernel, Kmt, Kmn):
    # compute kernel feature map, that is k(x_tilde, .) where the dot in this notebook is the variable x
    ans=[eval_kernel(x_tilde[i], x, kernel) for i in range(len(x_tilde))]

    # compute alphas https://proceedings.mlr.press/v162/chatalic22a/chatalic22a.pdf
    alphas=1/len(x_sample)*Kmt@Kmn@np.ones(len(x_sample))

    # compute summation
    ans=alphas.T@ans
    return ans


# K_x_sample=gramian(x_sample, x_sample)
# tol=piv cholesky tolerance
# otherwise you can define a maximum number of steps=piv cholesky steps

# K_x_sample is the kernel matrix computed on x_sample array of data
def piv_cholesky_kde(x_sample, x,kernel, K_x_sample, tol, steps):
    B,L,m, indices= piv_chol(np.matrix(K_x_sample), tol,steps)
    # get the pivots xs in the pivoted cholesky decomposition
    x_pivots=x_sample[indices]

    Km=gram(x_pivots, x_pivots)
    
    Kmn=gram(x_pivots, x_sample)
    ans=[eval_kernel(x_pivots[i], x, kernel) for i in range(len(x_pivots))]

    # compute alphas https://proceedings.mlr.press/v162/chatalic22a/chatalic22a.pdf
    alphas=1/len(x_sample)*np.linalg.pinv(Km)@Kmn@np.ones(len(x_sample))

    # compute summation
    
    ans=alphas.T@ans
    return ans



# 2d Kde
def kde2(x1,x2,x_sample, kernel):
    ans=[kernel(np.array([x_sample[i],[x1,x2]])) for i in range(len(x_sample))]
    ans=np.sum(ans)/len(x_sample)
    return ans


#2d EvalKernel
def eval_kernel2(x,x0, kernel):
    return kernel(np.array([x,x0]))
