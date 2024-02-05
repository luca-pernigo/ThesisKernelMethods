import numpy as np
from scipy import interpolate


def spline_evaluate(t, tck):

    t_vec=tck[0]
    cntr_points=tck[1]
    n=tck[2]

    e_vec=np.zeros(cntr_points.size)
    k=0
    while t_vec[n+k+1]<t:
        k+=1
    
    for i in range(n+1):
        e_vec[i]=cntr_points[i+k]


    for j in range(1,n+1,1):
        for i in range(0,n-j+1,1):
            e_vec[i]=(t_vec[n+i+k+1]-t)/(t_vec[n+i+k+1]-t_vec[i+k+j])*e_vec[i]+(t-t_vec[i+k+j])/(t_vec[n+i+k+1]-t_vec[i+k+j])*e_vec[i+1]
    
    return e_vec[0]


def control_points(x,y,n):
    ans=interpolate.splrep(x,y, k=n)[1]
    return ans