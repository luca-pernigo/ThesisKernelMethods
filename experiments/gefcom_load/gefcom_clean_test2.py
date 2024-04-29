import pandas as pd
import sys

import holidays

def clean_test(ith):
    if ith==15:
        df=pd.read_csv(f"/Users/luca/Desktop/GEFCom2014 Data/Load/Task {ith}/L{ith}-test.csv")
        # dataset 15 does not stick to convention used so far in the competition, thus we have an if conditional to handle it
        df['TIMESTAMP'] = pd.to_datetime(df['date'], format='%m/%d/%Y')

        # to be coherent hours go from 0 to 23
        df['HOUR'] =df["hour"]-1

        df['DAY'] = df['TIMESTAMP'].dt.day
        df['MONTH'] = df['TIMESTAMP'].dt.month
        df['HOUR'] = df['TIMESTAMP'].dt.hour
        df['YEAR'] = df['TIMESTAMP'].dt.year

        # add info weekday; saturday, sunday
        df['DAY_OF_WEEK'] = df['TIMESTAMP'].dt.dayofweek

        # holidays
        hol=holidays.US(years=range(2001, 2011))
        df["IS_HOLIDAY"]=df['TIMESTAMP'].isin(hol)
        # convert it to categorical variable
        df["IS_HOLIDAY"]=pd.Categorical(df["IS_HOLIDAY"].astype(int))


        # drop not useful columns
        df.drop(columns=['TIMESTAMP', 'date', 'hour'], inplace=True)


        # average temperatures
        weather_stat=df.filter(regex=("w.*")).columns.to_list()
        # average them 
        df['w_avg'] = df[weather_stat].mean(axis=1)

        # drop columns
        df.drop(columns=["YEAR"], inplace=True)

        df=df[["LOAD","DAY","MONTH","HOUR","DAY_OF_WEEK","IS_HOLIDAY"]+[[f"w{i}"] for i in range(1,25)]]
        
        # save to csv
        df.to_csv(f"/Users/luca/Desktop/GEFCom2014 Data/Load/Task {ith}/L{ith}-test_clean_2.csv", index=False)

        # print(df)

    else:    
        df=pd.read_csv(f"/Users/luca/Desktop/GEFCom2014 Data/Load/Task {ith}/L{ith}-test.csv")

        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format='%m%d%Y %H:%M')

        df['DAY'] = df['TIMESTAMP'].dt.day
        df['MONTH'] = df['TIMESTAMP'].dt.month
        df['HOUR'] = df['TIMESTAMP'].dt.hour
        df['YEAR'] = df['TIMESTAMP'].dt.year

        # add info weekday; saturday, sunday
        df['DAY_OF_WEEK'] = df['TIMESTAMP'].dt.dayofweek

        # holidays
        hol=holidays.US(years=range(2001, 2011))
        df["IS_HOLIDAY"]=df['TIMESTAMP'].isin(hol)
        # convert it to categorical variable
        df["IS_HOLIDAY"]=pd.Categorical(df["IS_HOLIDAY"].astype(int))


        # drop not useful columns
        df.drop(columns=['ZONEID', 'TIMESTAMP'], inplace=True)


        # average temperatures
        weather_stat=df.filter(regex=("w.*")).columns.to_list()
        # average them 
        df['w_avg'] = df[weather_stat].mean(axis=1)

        # drop columns
        df.drop(columns=["YEAR"], inplace=True)

        # save to csv
        df.to_csv(f"/Users/luca/Desktop/GEFCom2014 Data/Load/Task {ith}/L{ith}-test_clean_2.csv", index=False)

    # print(df)


if __name__=="__main__":
    clean_test(int(sys.argv[1]))