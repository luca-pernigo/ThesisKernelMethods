from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
from src.PivCholesky import piv_chol
from src.Gramian import eval_kernel, gram
import FMCA
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

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



def standardize(data):
    return (data-np.mean(data))/np.std(data)



class Kde:
    def __init__(self, kernel_type, kernel_prm, grid_size=1000):
        self.kernel_type=kernel_type
        self.kernel_prm=kernel_prm
        
        # empty array to store the estimated kme of pdf
        self.grid_size=grid_size
        self.y_axis=np.zeros(grid_size)
        self.cov=FMCA.CovarianceKernel(self.kernel_type, self.kernel_prm)


    def Kme(self, data):
        
        
        self.data=data
        self.data=self.data.reshape(self.data.size,)

        self.mu=np.mean(data)
        self.sigma=np.std(data)
        # reshape data in format (n,)
        data_size=data.size
        data=data.reshape(data.size,)

        # standardize
        self.data_standardized=standardize(data)

        self.x_axis=np.linspace(self.data_standardized.min(), self.data_standardized.max(),self.grid_size)

        m=self.x_axis.size
        

        for k in range (m):
            self.y_axis[k]=np.sum([self.cov.eval(np.array([np.array([self.data_standardized[i]])]),np.array([[self.x_axis[k]]])) for i in range(data_size)])/data_size
        
    def PlotKde(self):
        # plot kme and hist plot of original data
        # note that we have to scale back the data
        # the shift in the direction of mu does not affect the density
        # however we have to take into account the multiplcation by sigma
        # so the estimated density has to be divided by sigma
        
        try:
            self.x_axis
        except:
            print("In order to plot the Kde first you have to compute it with self.Kme")
            return
            
        m=self.x_axis.size
        self.data=self.data.reshape(self.data.size,)
        plt.plot(self.mu+self.sigma*self.x_axis.reshape(m,), self.y_axis/self.sigma)

        # hist of original data
        plt.hist(self.data, bins=100, density=True)

        plt.xlabel("Price range")
        plt.ylabel("PDF")
        plt.title("KME Estimation")



    def PlotKmatrix(self, data):
        data=data.reshape(data.size,)
        data=standardize(data)

        a=np.array([data])
        covarianceFull=self.cov.eval(a,a)

        # plot kernel covariance
        fig, ax=plt.subplots(figsize=(4, 4))
        im = ax.imshow(covarianceFull)
        ax.set_xlabel("days")
        ax.set_ylabel("days")
        ax.set_title(f"KernelMatrix")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        plt.colorbar(im, cax=cax)
