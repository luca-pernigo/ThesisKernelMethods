import pandas as pd


df=pd.DataFrame(columns=[f"Task {i}" for i in range(1,16)], index=["MLR", "TBATS", "Prophet", "KNN","SVR","LSTM", "KrnRidge", "KrnSVR"])

df.to_csv("res_rmse.csv")
df.to_csv("res_mae.csv")
df.to_csv("res_mape.csv")

