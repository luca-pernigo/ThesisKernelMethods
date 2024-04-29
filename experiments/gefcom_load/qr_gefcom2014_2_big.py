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
    # load data start 2008- start 2010
    df=pd.read_csv("/Users/luca/Desktop/GEFCom2014 Data/Load/Task 1/L-train_clean_full_2.csv")
    
    X_train=df[[f"w{i}" for i in range(1,26)]+["DAY",  "MONTH",  "HOUR",  "DAY_OF_WEEK",  "IS_HOLIDAY",  "w_avg"]]
    y_train=df["LOAD"]

    # 99 quantiles
    quantiles = [i/100 for i in range(0,10)]


    # kernel quantile regression
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    qr_krn_models=[]
    y_test_pred_qr_krn=[]
    # cross validated on smaller subset=> C=100, gamma=0.5=> I use these on the full dataset

    for i,q in enumerate(tqdm(quantiles)):

        # fit data for specific quantile
        qr_krn_models+=[KQR(alpha=q, C=100, gamma=0.5).fit(X_train_scaled, y_train)]

        # save models to pickle
        pickle.dump(qr_krn_models[i], open(f'/Users/luca/Desktop/ThesisKernelMethods/dataset_applications/models_full_2/krn_qr_{i}.pkl', 'wb'))
    