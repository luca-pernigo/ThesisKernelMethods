import numpy as np
import pandas as pd

from qr import qr_tests

if __name__=="__main__":
    # load data
    df=pd.read_csv("/Users/luca/Desktop/ThesisKernelMethods/dataset_applications/abalone.csv", sep=",", decimal=".")

    # map categorical regressor to integer form
    key = {"I":0, "M":1, "F":2}
    df['Sex'] = df['Sex'].map(key)

    X=df.iloc[:,0:-1]
    y=df.iloc[:,-1]
    
    qr_tests(X,y,"qr_abalone")