import numpy as np
import pandas as pd

from scipy.optimize import minimize
import time

def exponential_ker(x,x_sample,gamma):
    # Computes first part of equation to be maximized, eq 8 paper https://arxiv.org/pdf/1203.3472.pdf
    
    n = x_sample.shape[0]
    k=np.zeros(n)
    
    for i in range(n):
        # gaussian kernel between x and x'
        k[i] = np.exp(-np.linalg.norm(x-x_sample[i,:])/gamma**2)

    # take sample mean to approximate the expectation of the kernel
    expectation_kernel = sum(k)/n
    return expectation_kernel

def summation_ker(x,xss,t,gamma):
    # Computes second part of equation to be maximized, eq 8 paper https://arxiv.org/pdf/1203.3472.pdf
    
    #initialize
    summation=0
    
    k=np.zeros(t)
    
    for i in range(t):
        k[i] = np.exp(-np.linalg.norm(x-xss[i,:])/gamma**2)
    summation = np.sum(k)
    
    summation = summation/(t+1)
    return summation


def kernel_herding(x_sample,n,gamma):
    # n is how much samples we want to generate
    dim = x_sample.shape[1] #dimension of the sample 
    n_sample = x_sample.shape[0] #number of samples
    
    xss = np.zeros((n, dim))
    i=1

    # in scipy.optimize.minimize the second variable corresponds to an initial guess
    seed_best = np.zeros(dim)
    
    while i<n:
        print (f"step.{i}", end=" ")
        # note it is the same
        # from the paper except that the sign is inverted, therefore
        # instead of maximizing we have to minimize
        # Minimize with scipy
        f = lambda x: -exponential_ker(x,x_sample,gamma)+summation_ker(x,xss,i,gamma)
        res = minimize(f,
                           seed_best,
                           method='nelder-mead',
                           options={'xtol': 1e-4, 'disp': False})

        
        # next best super sample is set as the next best seed
        seed=np.array([-exponential_ker(xss[j,:],x_sample,gamma)+summation_ker(xss[j,:],xss,i,gamma) for j in range(i)])
        # start from where error is minimum
        # this is criterion to set new seed
        seed_best = np.argmin(seed)
        seed_best=xss[seed_best,:]
        
        #append ith value to super sample list
        xss[i,:]=res.x
        
        i=i+1
       
    return xss
