import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

import sys

import torch
import torch.nn as nn



def load_train():
    df_train=pd.read_csv("/Users/luca/Desktop/GEFCom2014 Data/Load/L-train.csv")
    y_train=df_train["LOAD"]
    X_train=df_train[["DAY","MONTH","HOUR","DAY_OF_WEEK","IS_HOLIDAY","w_avg"]]

    return df_train, X_train, y_train



def load_test(i):
    df_test=pd.read_csv(f"/Users/luca/Desktop/GEFCom2014 Data/Load/Task {i}/L{i}-test_clean.csv")
    X_test=df_test[["DAY",  "MONTH",  "HOUR",  "DAY_OF_WEEK",  "IS_HOLIDAY",  "w_avg"]]
    y_test=df_test["LOAD"]

    return df_test, X_test, y_test


def df_fill(x,y, name, i):
    df_rmse=pd.read_csv("res_rmse.csv", index_col=0)
    df_rmse.loc[name,f"Task {i}"]=np.sqrt(mean_squared_error(x, y))
    df_rmse.to_csv("res_rmse.csv")

    df_mae=pd.read_csv("res_mae.csv", index_col=0)
    df_mae.loc[name,f"Task {i}"]= mean_absolute_error(x, y)
    df_mae.to_csv("res_mae.csv")

    df_mape=pd.read_csv("res_mape.csv", index_col=0)
    df_mape.loc[name,f"Task {i}"]=mean_absolute_percentage_error(x, y)
    df_mape.to_csv("res_mape.csv")




class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out



def df_to_seq(df, w):
    # window size for historical data, we consider the last 24 data, hourly frequency
    window_size = w

    # create sequences of data for lstm
    sequences = []
    for i in range(window_size, len(df)):
        # get last w observations of data
        sequence = df.iloc[i - window_size:i]
        sequences.append(sequence)

    # split sequences into input features and targets
    X = []
    y = []
    for sequence in sequences:
        X.append(sequence[["DAY","MONTH","HOUR","DAY_OF_WEEK","IS_HOLIDAY","w_avg"]].values)
        
        if "LOAD" in df.columns:
            y.append(sequence['LOAD'].values[-1])

    X = np.array(X)
    y = np.array(y)
    
    return X,y



def df_to_tex(df,name):

    if name!=None:
        original_stdout=sys.stdout

        with open(f"/Users/luca/Desktop/ThesisKernelMethods/thesis/tables/{name}.txt", "w") as f:
            sys.stdout=f
            print(df.to_latex(index=True,
                    formatters={"name": str.upper},
                    float_format="{:.4f}".format))
            
            sys.stdout=original_stdout