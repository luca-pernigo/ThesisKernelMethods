import datetime 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

from point_utility import load_train, load_test, df_fill

# get train data
df_train, X_train, y_train=load_train()

reg = LinearRegression().fit(X_train, y_train)

for i in range(1,16):
    # get test data
    df_test, X_test, y_test=load_test(i)

    # model
    y_hat=reg.predict(X_test)

    # populate df of results with score metrics
    df_fill(y_test,y_hat,"MLR",i)

    # print score metrics
    rmse=np.sqrt(mean_squared_error(y_test, y_hat))
    mae=mean_absolute_error(y_test, y_hat)
    mape=mean_absolute_percentage_error(y_test, y_hat)
    
    print(f"Task {i}, RMSE: ", rmse)
    print(f"Task {i}, MAE: ", mae)
    print(f"Task {i}, MAPE: ", mape)
    