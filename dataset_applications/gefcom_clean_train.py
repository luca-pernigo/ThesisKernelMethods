import numpy as np
import pandas as pd
from sklearn.metrics import mean_pinball_loss

from qr import qr_tests

import holidays


if __name__=="__main__":
    # load data
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


    # slice for training, from when we have load informations 2005
    # slice from 2009 so it is less computations heavy
    idx_start = df.loc[(df['DAY'] == 1) & (df["MONTH"]==1) & (df["YEAR"]==2009) & (df["HOUR"])==1].index[0]
    
    
    idx_end = df.loc[(df['DAY'] == 1) & (df["MONTH"]==1) & (df["YEAR"]==2010) & (df["HOUR"])==1].index[0]


    df = df.loc[idx_start-1:idx_end-1]

    # average temperatures
    # select weather stations data using regex
    weather_stat=df.filter(regex=("w.*")).columns.to_list()
    # average them 
    df['w_avg'] = df[weather_stat].mean(axis=1)


    df.drop(columns=weather_stat, inplace=True)
    df.drop(columns=["YEAR"], inplace=True)

    df.to_csv("/Users/luca/Desktop/GEFCom2014 Data/Load/L2009-train.csv",index=False)

    
    