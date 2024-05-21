import numpy as np
import pandas as pd

from qr import qr_tests

if __name__=="__main__":
    # load data
    df=pd.read_csv("/Users/luca/Desktop/ThesisKernelMethods/extradata/car_data.csv", sep=",", decimal=".")

    # map categorical regressor to integer form
    key6 = {"Petrol":0, "Diesel":1, "CNG":2}
    df['Fuel_Type'] = df['Fuel_Type'].map(key6)

    key7 = {"Dealer":0, "Individual":1}
    df['Seller_Type'] = df['Seller_Type'].map(key7)

    key8 = {"Manual":0, "Automatic":1}
    df['Transmission'] = df['Transmission'].map(key8)

    X=df.iloc[:,[1,3,4,5,6,7,8]]
    y=df.iloc[:,2]
    print(X)
    
    qr_tests(X,y,"qr_cars")