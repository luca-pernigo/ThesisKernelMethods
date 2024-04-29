import numpy as np

import pandas as pd
import pickle

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn import preprocessing

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV,TimeSeriesSplit,HalvingGridSearchCV
from sklearn.svm import SVR

from point_utility import load_train, load_test, df_fill

df_train, X_train, y_train=load_train()

scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)



svr=pickle.load(open('/Users/luca/Desktop/ThesisKernelMethods/gefcom2012_load/svr_load.pkl', 'rb'))


for i in range(1,16):
    df_test, X_test, y_test=load_test(i)
    X_test_scaled = scaler.transform(X_test)

    y_hat=svr.predict(X_test_scaled)
    df_fill(y_test, y_hat, "SVR",i)


    rmse=np.sqrt(mean_squared_error(y_test, y_hat))
    mae=mean_absolute_error(y_test, y_hat)
    mape=mean_absolute_percentage_error(y_test, y_hat)
    
    print(f"Task {i}, RMSE: ", rmse)
    print(f"Task {i}, MAE: ", mae)
    print(f"Task {i}, MAPE: ", mape)
    