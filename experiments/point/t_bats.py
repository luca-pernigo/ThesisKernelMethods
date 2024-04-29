import numpy as np
import pandas as pd
import pickle

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

from tbats import BATS, TBATS

from point_utility import load_train, load_test, df_fill


df_train, X_train, y_train=load_train()

tbats=pickle.load(open("/Users/luca/Desktop/ThesisKernelMethods/experiments/point/models/tbats_load.pkl", "rb"))

for i in range(1,16):
    df_test, X_test, y_test=load_test(i)

    y_hat=tbats.forecast(steps=len(y_test))

    df_fill(y_test,y_hat,"TBATS",i)


    rmse=np.sqrt(mean_squared_error(y_test, y_hat))
    mae=mean_absolute_error(y_test, y_hat)
    mape=mean_absolute_percentage_error(y_test, y_hat)
    
    print(f"Task {i}, RMSE: ", rmse)
    print(f"Task {i}, MAE: ", mae)
    print(f"Task {i}, MAPE: ", mape)
    