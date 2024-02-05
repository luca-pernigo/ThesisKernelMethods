import numpy as np

def piv_chol(K, tol, steps=0):
# K=square symmetric kernel matrix
    m=1
    d=np.diag(K)
    # e=np.linalg.eig(K)[1]
    e=np.matrix(np.zeros(shape=(K.shape[0],1)))

    L=np.matrix(np.zeros(shape=(K.shape[0],K.shape[0])))
    B=np.matrix(np.zeros(shape=(K.shape[0],K.shape[0])))
    err=np.sum(np.abs(d))
    i=0
    indices=[]
    N=K.shape[0]-1
# steps=0 by default, if nothing is specified the maximum number of steps will be K.shape[0]-1
# otherwise if any steps variable is specified, the number of maximum steps can be set to be smaller
    if steps!=0:
        N=steps-1
    while err>tol and i<=N:
            
        j=np.argmax(d)
        indices+=[j]
        l_m_hat=K[:,j]-L@L.T[:,j]


        ej=e.copy() 
        ej[j]=1
        
        b_m_hat=ej-B@L.T[:,j]
    
        
        lm=l_m_hat/np.sqrt(d[j]+1e-10)
        
        bm = b_m_hat/np.sqrt(d[j]+1e-10)
                  
        L[:,i]=lm
        B[:,i]=bm

        
        d=d-np.squeeze(np.asarray(lm))*np.squeeze(np.asarray(lm))

        err=np.sum(np.abs(d))
        
        m+=1
        i+=1

    L=L[:,0:(m+1)]
    B=B[:,0:(m+1)]

    # print(err)
    ans=[B, L, m, indices]
    return ans