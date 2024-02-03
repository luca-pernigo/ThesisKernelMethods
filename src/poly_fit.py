import numpy as np

import math

def beta_rat_polyfit2(x, y, order):
    n=y.size
    A=np.zeros([n,order*2+1])
    A[:,0]=np.ones(y.size)
    for i in range(order):
        A[:,i+1]=x**(i+1)
    for i in range(order):
        A[:,order+i+1]=-x**(i+1)*y

    b=np.linalg.inv(A.T@A)@A.T@y
    return b


def polyfit_evaluate2(x, b):
    m=b.size
    x_vec=np.ones(math.ceil(m/2))
    print(x_vec.size)
    for i in range(len(x_vec)):
        x_vec[i]=x**i
    print(x_vec)
    print(b[math.ceil(m/2):])
    # x_vec=np.array([x0,x1,x2,x3])
    
    ans= x_vec.T@b[0:math.ceil(m/2)]/(1+x_vec[1:].T@b[math.ceil(m/2):])
    return ans

def polyfit2(xvec,b):
    ans=np.array([polyfit_evaluate2(i,b) for i in xvec])
    return ans




# naive implementation
# def beta_rat_polyfit(x, y):
#     n=y.size
#     A=np.zeros([n,5])
#     A[:,0]=np.ones(y.size)
#     A[:,1]=x
#     A[:,2]=x**2
#     A[:,3]=-x*y
#     A[:,4]=-x**2*y

#     b=np.linalg.inv(A.T@A)@A.T@y
#     return b


# def polyfit_evaluate(x, b):
#     x0=1
#     x1=x
#     x2=x**2
#     x_vec=np.array([x0,x1,x2])
#     print(x_vec.size)
#     print(x_vec)
#     print(b[3:])
#     ans= x_vec.T@b[0:3]/(1+x_vec[1:].T@b[3:])
#     return ans

# def polyfit(xvec,b):
#     ans=np.array([polyfit_evaluate(i,b) for i in xvec])
#     return ans
