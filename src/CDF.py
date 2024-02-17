import numpy as np
from src.miscellaneous import find

def cdf_funct(x):
    ans=np.arange(len(x)) / float(len(x)-1)
    return ans


def inv_cdf_funct(y,x):
    # x has to be in format np([])
    x=x.reshape(x.size,)

    sorted_x_axis=np.sort(x)
    idx_pos= y*float(len(sorted_x_axis))
    # print(idx_pos)
    # -1 because when calculating cdf_funct
    # the function appends an additional value to
    # the sorted array. However, when it comes to the inverse
    # we deal with the original sorted array. That is the one
    # without the additional x
    ans=  sorted_x_axis[int(idx_pos)-1]
    return ans




def Phi(xn, xcdf, ycdf):
    i=find(xn, xcdf)
    if(xn<xcdf.min()):
        ans=0
        return ans
    if(xn>xcdf.max()):
        ans=1
        return ans
    

    ans=ycdf[i]*(xcdf[i+1]-xn)/(xcdf[i+1]-xcdf[i])+ycdf[i+1]*(xn-xcdf[i])/(xcdf[i+1]-xcdf[i])
    return ans




def phi(xn, xpdf, ypdf):
    i=find(xn, xpdf)
    if(xn<xpdf.min()):
        ans=ypdf[0]
        return ans
    if(xn>xpdf.max()):
        ans=ypdf[-1]
        return ans
    

    ans=ypdf[i]*(xpdf[i+1]-xn)/(xpdf[i+1]-xpdf[i])+ypdf[i+1]*(xn-xpdf[i])/(xpdf[i+1]-xpdf[i])
    
    return ans




def phi_inv(u, xpdf,ypdf, xcdf, ycdf, tol=1e-3):
    xn=0
    bol=True
    
    if(u==1):
        return xpdf[-1]
    
    while(bol):
        xnew=xn-(Phi(xn,xcdf, ycdf)-u)/phi(xn, xpdf, ypdf)
        if(abs(xn-xnew)<tol):
            bol=False
    
        xn=xnew
        # print(xn)
    
    return xn