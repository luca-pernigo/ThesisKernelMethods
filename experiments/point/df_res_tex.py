import pandas as pd

from point_utility import df_to_tex

df_rmse=pd.read_csv("res_rmse.csv", index_col=0)
df_mae=pd.read_csv("res_mae.csv", index_col=0)
df_mape=pd.read_csv("res_mape.csv", index_col=0)


df_to_tex(df_rmse, "point_RMSE")
df_to_tex(df_mae, "point_MAE")
df_to_tex(df_mape, "point_MAPE")

