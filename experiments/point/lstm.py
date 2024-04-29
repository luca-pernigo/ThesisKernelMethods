import matplotlib.pyplot as plt 
import matplotlib.dates as mdates 

import numpy as np

import pandas as pd
import pickle

import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

from point_utility import load_train, load_test, df_fill, LSTMModel, df_to_seq

df_train, X_train, y_train=load_train()

w=24
X,y=df_to_seq(df_train, w)

# scale data for better performance
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).reshape(-1)


lstm=pickle.load(open('/Users/luca/Desktop/ThesisKernelMethods/dataset_applications/lstm_load.pkl', 'rb'))

for i in range(1,16):
    df_test,X_test, y_test=load_test(i)
    

    if i==1:
        df_test_prev=pd.read_csv("/Users/luca/Desktop/GEFCom2014 Data/Load/L-train.csv")
        df_test=pd.read_csv("/Users/luca/Desktop/GEFCom2014 Data/Load/Task 1/L1-test_clean.csv")
    else:
        df_test_prev=pd.read_csv(f"/Users/luca/Desktop/GEFCom2014 Data/Load/Task {i}/L{i}-test_clean.csv")
        df_test=pd.read_csv(f"/Users/luca/Desktop/GEFCom2014 Data/Load/Task {i}/L{i}-test_clean.csv")

    df_task=pd.concat([df_test_prev[-w:],df_test], ignore_index=True)[["DAY",  "MONTH",  "HOUR",  "DAY_OF_WEEK",  "IS_HOLIDAY",  "w_avg"]]

    X_task,_=df_to_seq(df_task, w)
    X_task_scaled = scaler_X.transform(X_task.reshape(-1, X_task.shape[-1])).reshape(X_task.shape)
    X_task_tensor = torch.tensor(X_task_scaled, dtype=torch.float32)

    lstm.eval()
    with torch.no_grad():
        y_hat = lstm(X_task_tensor).squeeze().numpy()
        y_hat = scaler_y.inverse_transform(y_hat.reshape(-1, 1)).reshape(-1)


    df_fill(y_test, y_hat, "LSTM", i)

    rmse=np.sqrt(mean_squared_error(y_test, y_hat))
    mae=mean_absolute_error(y_test, y_hat)
    mape=mean_absolute_percentage_error(y_test, y_hat)
    
    print(f"Task {i}, RMSE: ", rmse)
    print(f"Task {i}, MAE: ", mae)
    print(f"Task {i}, MAPE: ", mape)
    