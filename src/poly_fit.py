import numpy as np
import math
from src.cdf import cdf_funct
from src.miscellaneous import find
from scipy.optimize import curve_fit


def spl_polyfit(x,y):
    s=3
    # take subset to make interpolation smoother
    x=np.sort(x)
    x_sub=x[0:-1:s]
    # append the max as end point to be interpolated
    if x_sub[-1]!=x.max:
        x_sub=np.append([x_sub], [x.max()])

    while x_sub.size%s!=0:
        # print("*")
        i=np.random.random_integers(0,x_sub.size-1)
        x_sub=np.delete(x_sub, i)

    y_sub=cdf_funct(x_sub)

    fit_spl_rpol=np.zeros(y_sub.size)
    for i in range(int(y_sub.size/s)):
        fit_spl_rpol[i*s:(i+1)*s]=polyfit(y_sub[i*s:(i+1)*s],beta_rat_polyfit(y_sub[i*s:(i+1)*s], np.sort(x_sub)[i*s:(i+1)*s], order=1))
    
    return [x_sub, y_sub,fit_spl_rpol]

def simulate_spl_polyfit(unif,y, fit_spl_rpol):
    
    sim_spl_rpol=np.zeros(unif.size)

    for i in range(len(unif)):
        idx=find(unif[i], y)
        t1=(y[idx+1]-unif[i])/(y[idx+1]-y[idx])
        t2=(unif[i]-y[idx])/(y[idx+1]-y[idx])
        sim_spl_rpol[i]=fit_spl_rpol[idx]*t1+fit_spl_rpol[idx+1]*t2
    
    return sim_spl_rpol


def beta_rat_polyfit(x, y, order):
    n=y.size
    A=np.zeros([n, order*2+1])
    A[:,0]=np.ones(y.size)
    for i in range(order):
        A[:,i+1]=x**(i+1)
    for i in range(order):
        A[:,order+i+1]=-x**(i+1)*y
    
    # b=np.linalg.inv(A.T@A)@A.T@y
    
    # # solve with QR decomposition
    Q, R = np.linalg.qr(A)
    p = np.dot(Q.T, y)
    b=np.dot(np.linalg.inv(R), p)
    

    return b


def polyfit_evaluate(x, b):
    m=b.size
    x_vec=np.ones(math.ceil(m/2))

    for i in range(len(x_vec)):
        x_vec[i]=x**i

    
    ans= x_vec.T@b[0:math.ceil(m/2)]/(1+x_vec[1:].T@b[math.ceil(m/2):])
    return ans

def polyfit(xvec,b):
    ans=np.array([polyfit_evaluate(i,b) for i in xvec])
    return ans



# optimization by Gauss Newton method
def A(x, y, order):
    n=y.size
    A=np.zeros([n, order*2+1])
    A[:,0]=np.ones(y.size)
    for i in range(order):
        A[:,i+1]=x**(i+1)
    for i in range(order):
        A[:,order+i+1]=-x**(i+1)*y
    
    
    return A

def rf(A,beta,y):
    r=y-A@beta
    return r

def GaussNewton(x,y, order):
    beta=np.zeros(order*2+1)
    M=A(x,y,order=2)
    i=0
    while i<100:
        M=-A(x,y,order)
        r=rf(-M, beta, y)
        q=np.linalg.solve(M.T@M, M.T@r)
        beta=beta-q
        i+=1
    return beta


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


def rational(x, p, q):
    return np.polyval(p, x) / ([1.0]+np.polyval(q, x))

def rational_2(x, p0, p1, p2, q1, q2):
    return rational(x, [p0, p1, p2], [q1, q2])


def rational_3(x, p0, p1, p2,p3, q1, q2,q3):
    return rational(x, [p0, p1, p2, p3], [q1, q2,q3])

def rational_4(x, p0, p1, p2,p3,p4, q1, q2,q3,q4):
    return rational(x, [p0, p1, p2, p3,p4], [q1, q2,q3,q4])


def rational_5(x, p0, p1, p2,p3,p4,p5, q1, q2,q3,q4,q5):
    return rational(x, [p0, p1, p2, p3,p4,p5], [q1, q2,q3,q4,q5])



