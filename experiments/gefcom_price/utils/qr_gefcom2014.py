import numpy as np
import pandas as pd
import pickle
from pprint import pprint
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
from tqdm import tqdm

from kqr import KQR

from cvxopt import matrix, spmatrix, sparse

if __name__=="__main__":
    # load data
    df=pd.read_csv("/Users/luca/Desktop/ThesisKernelMethods/dataset_applications/gefcom_price/Price/Task 4/Task4_P_train.csv")
    
    # define train
    X_train=df[["MONTH","DAY","HOUR","Forecasted Total Load","Forecasted Zonal Load"]][-1000:]
    y_train=df["Zonal Price"][-1000:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # 99 quantiles
    quantiles = [i/100 for i in range(1,100)]

    # kernel quantile regression
    qr_krn_models=[]
    y_test_pred_qr_krn=[]
    
    for i,q in enumerate(tqdm(quantiles)):

        # fit data for specific quantile
        qr_krn_models+=[KQR(alpha=q, C=100, gamma=0.5).fit(X_train_scaled, y_train)]

        # save models to pickle
        pickle.dump(qr_krn_models[i], open(f'/Users/luca/Desktop/ThesisKernelMethods/dataset_applications/gefcom_price/models/krn_qr_{i}.pkl', 'wb'))
        