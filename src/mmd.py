import numpy as np
import FMCA

def mmd(x,y, kernel_type, kernel_prm):
    m=x.size
    n=y.size
    cov=FMCA.CovarianceKernel(f"{kernel_type}", kernel_prm)
    Kx=cov.eval(x,x)
    Ky=cov.eval(y,y)
    Kxy=cov.eval(x,y)

    A=(np.sum(Kx))/(m*m)
    B= (np.sum(Kxy))/(m*n)
    C=(np.sum(Ky))/(n*n)

    mmd2=A-2*B+C

    return mmd2