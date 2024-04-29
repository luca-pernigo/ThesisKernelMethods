# same as weather_prediction_avg.py but selecting only weather stations "w6", "w10", "w22", "w25"
import numpy as np
import pandas as pd
from sklearn.metrics import mean_pinball_loss

from qr import qr_tests

import holidays


df=pd.read_csv("/Users/luca/Desktop/GEFCom2014 Data/Load/Task 1/L1-train.csv", sep=",", decimal=".")
    
# clean data dates
dates = pd.date_range(start='2001-01-01 01:00:00', periods=len(df), freq='H')
df['DAY'] = dates.day
df['MONTH'] = dates.month
df['HOUR'] = dates.hour
df['YEAR'] = dates.year

# add info weekday; saturday, sunday
df['DAY_OF_WEEK'] = dates.dayofweek

# holidays
hol=holidays.US(years=range(2001, 2011))
df["IS_HOLIDAY"]=pd.Series(dates.date).isin(hol)
# convert it to categorical variable
df["IS_HOLIDAY"]=pd.Categorical(df["IS_HOLIDAY"].astype(int))


# drop not useful columns
# zoneid is dropped because are all 1, so it is not a meaningful independent variable
# the information in timestamp has been decomposed in four columns so we can drop it
df.drop(columns=['ZONEID', 'TIMESTAMP'], inplace=True)


# average temperatures
w_avg_sub = df[["w6", "w10", "w22", "w25"]].mean(axis=1)

weather_stat=df.filter(regex=("w.*")).columns.to_list()
df.drop(columns=weather_stat, inplace=True)
df['w_avg']=w_avg_sub

df.drop(columns=["YEAR"], inplace=True)

df.to_csv("/Users/luca/Desktop/GEFCom2014 Data/Load/L-weather-train2.csv",index=False)



df_w=pd.read_csv(f"/Users/luca/Desktop/GEFCom2014 Data/Load/L-weather-train2.csv")

# drop leap year
df_w=df_w[~((df_w["MONTH"]==2) & (df_w["DAY"]==29))]

w_avg=df_w.groupby(["MONTH","DAY","HOUR"]).mean("w_avg").reset_index()


# print(len(w_avg))
task_month={1:10,2:11,3:12,4:1,5:2,6:3,7:4,8:5,9:6,10:7,11:8,12:9,13:10,14:11,15:12}

for t,m in task_month.items():
    
    w_avg_month=w_avg[w_avg["MONTH"]==m]
    
    if m!=12:
        idx_next=w_avg_month.index[-1]
    else:
        idx_next=0
    # print(end)
    # print(w_avg.iloc[end-1])
    
    # add last row because test data ends at 00:00 of the next month
    w_avg_month.loc[len(w_avg_month)] = w_avg.iloc[idx_next+1]
    
    # drop first row because test data starts from hour 01:00, thus drop data for hour 00:00
    w_avg_month.drop(index=w_avg_month.index[0], axis=0, inplace=True)
    # print(w_avg_month)

    df_test=pd.read_csv(f"/Users/luca/Desktop/GEFCom2014 Data/Load/Task {t}/L{t}-test_clean.csv")
    
    df_test["w_avg"]=w_avg_month["w_avg"].values

    df_test.to_csv(f"/Users/luca/Desktop/GEFCom2014 Data/Load/Task {t}/L{t}-avg_test_clean2.csv", index=False)


