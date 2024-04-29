import holidays

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import pickle
from prophet import Prophet

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

from point_utility import load_train, load_test, df_fill

df_train, X_train, y_train=load_train()

# rename for prophet API
df_train=df_train.rename(columns={'DS': 'ds'})
df_train=df_train.rename(columns={'LOAD': 'y'})
df_train["y"]=np.log(df_train["y"])
df_train["w_avg2"]=df_train["w_avg"]**2


proph=pickle.load(open("/Users/luca/Desktop/ThesisKernelMethods/experiments/point/models/prophet_v2_load.pkl", "rb"))



for i in range(1,16):
    future=pd.DataFrame(columns=["ds", "w_avg","w_avg2"])
    df_test, X_test, y_test=load_test(i)
    
    df_test=df_test.rename(columns={'DS': 'ds'})
    df_test["w_avg2"]=df_test["w_avg"]**2
    

    future["ds"]=df_test["ds"]
    regressors=df_test[["w_avg","w_avg2"]]
    future["w_avg"]=regressors["w_avg"].values
    future["w_avg2"]=regressors["w_avg2"].values

    forecast = proph.predict(future)
    y_hat=np.exp(forecast["yhat"].values)

    df_fill(y_test,y_hat,"Prophet",i)

    

    rmse=np.sqrt(mean_squared_error(y_test, y_hat))
    mae=mean_absolute_error(y_test, y_hat)
    mape=mean_absolute_percentage_error(y_test, y_hat)
    
    print(f"Task {i}, RMSE: ", rmse)
    print(f"Task {i}, MAE: ", mae)
    print(f"Task {i}, MAPE: ", mape)
    