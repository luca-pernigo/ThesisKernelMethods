import pandas as pd
import re

import holidays

def clean_time(file):
    # read file as pandas df
    df=pd.read_csv(file, sep=",", decimal=".")
    
    ## clean data dates
    dates = pd.date_range(start='2011-01-01 00:00:00', periods=len(df), freq='H')
    df['DAY'] = dates.day
    df['MONTH'] = dates.month
    df['HOUR'] = dates.hour
    df['YEAR'] = dates.year
    
    # add info weekday; saturday, sunday
    df['DAY_OF_WEEK'] = dates.dayofweek

    # holidays
    hol=holidays.US(years=range(2011, 2014))
    df["IS_HOLIDAY"]=pd.Series(dates.date).isin(hol).astype(int)  
    # print(df)
    return df

def order_columns(df):
    df=df[["ZONEID","timestamp","MONTH","DAY","HOUR","Forecasted Total Load","Forecasted Zonal Load","Zonal Price"]]
    return df

def get_task_number(file):
    pattern = r'(\d+)'

    matches = re.findall(pattern, file)

    if matches:
        n = int(matches[0])
        
    return n



def get_test(file):
    pattern = r'(\d+)'

    matches = re.findall(pattern, file)

    if matches:
        n = int(matches[0])
        n_new = n + 1

        file_new = re.sub(pattern, str(n_new), file)
    
    return file_new