import numpy as np
import pandas as pd

from qr import qr_tests

if __name__=="__main__":
    # load data
    df=pd.read_csv("/Users/luca/Desktop/ThesisKernelMethods/extradata/BostonHousing.csv", sep=",", decimal=".")

    X=df.iloc[:,0:-1]
    y=df.iloc[:,-1]
    qr_tests(X,y, "qr_boston")