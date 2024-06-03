import matplotlib.pyplot as plt

import pandas as pd

# script for eda gefcom load and gefcom price

# load
df=pd.read_csv("/Users/luca/Desktop/GEFCom2014 Data/Load/L-train_clean.csv")

# load starts from 2005
df=df[df["YEAR"]>=2005][1:]
df["TIME"]=pd.to_datetime(df[['YEAR', 'MONTH', 'DAY', 'HOUR']])


# # plot load
plt.figure(figsize=(15,5))
plt.plot(df["TIME"], df["LOAD"], color="black")

plt.xlabel("Time")
plt.ylabel("Load (MW)")

plt.savefig("/Users/luca/Desktop/ThesisKernelMethods/thesis/images/gefcom_load_historical.png")

plt.show()


# # plot w_avg
plt.figure(figsize=(15,5))
plt.plot(df["TIME"], df["w_avg"], color="black")

plt.xlabel("Time")
plt.ylabel("Average weather temperature")

plt.savefig("/Users/luca/Desktop/ThesisKernelMethods/thesis/images/gefcom_w_avg_historical.png")

plt.show()

# # load vs w_avg
plt.plot(df["w_avg"], df["LOAD"], "o", alpha=0.4, markersize=3)
plt.xlabel("Average weather temperature")
plt.ylabel("Load (MW)")

plt.savefig("/Users/luca/Desktop/ThesisKernelMethods/thesis/images/gefcom_load_vs_w_avg.png")
plt.show()


# price

df=pd.read_csv("/Users/luca/Desktop/ThesisKernelMethods/experiments/Data/Price/Task 1/Task1_P_train.csv")

# # zonal price vs total load
plt.plot(df["Forecasted Total Load"], df["Zonal Price"], "o", alpha=0.4, markersize=3)
plt.xlabel("Forecasted total load")
plt.ylabel("Price $/MW")
plt.savefig("/Users/luca/Desktop/ThesisKernelMethods/thesis/images/gefcom_zonal_price_vs_total_load.png")
plt.show()


# # zonal price vs zonal load
plt.plot(df["Forecasted Zonal Load"], df["Zonal Price"], "o", alpha=0.4, markersize=3)
plt.xlabel("Forecasted zonal load")
plt.ylabel("Price $/MW")
plt.savefig("/Users/luca/Desktop/ThesisKernelMethods/thesis/images/gefcom_zonal_price_vs_zonal_load.png")
plt.show()