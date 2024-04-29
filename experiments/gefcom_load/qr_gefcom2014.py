import numpy as np
import pandas as pd
import pickle
from pprint import pprint
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
from tqdm import tqdm

from kqr import KQR

from cvxopt import matrix, spmatrix, sparse




if __name__=="__main__":
    # load data 2009
    df=pd.read_csv("/Users/luca/Desktop/GEFCom2014 Data/Load/L2009-train.csv")
    
    X_train=df[["DAY",  "MONTH",  "HOUR",  "DAY_OF_WEEK",  "IS_HOLIDAY",  "w_avg"]]
    y_train=df["LOAD"]

    # 99 quantiles
    quantiles = [i/100 for i in range(1,100)]

    # define loss to tune
    neg_mean_pinball_loss_scorer_05 = make_scorer(
    mean_pinball_loss,
    alpha=0.5,
    greater_is_better=False,  # maximize the negative of the loss
    )

    # kernel quantile regression
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    qr_krn_models=[]
    y_test_pred_qr_krn=[]

    param_grid_krn = dict(
    C=[1e-1,1e-2,1, 5, 10,1e2,1e4],
    gamma=[1e-1,1e-2,0.5,1, 5, 10, 20]   
    )
    
    krn_blueprint=KQR(alpha=0.5)
    best_hyperparameters_krn=HalvingRandomSearchCV(
            krn_blueprint,
            param_grid_krn,
            scoring=neg_mean_pinball_loss_scorer_05,
            n_jobs=2,
            random_state=0,
        ).fit(X_train_scaled, y_train).best_params_
    
    for i,q in enumerate(tqdm(quantiles)):

        # fit data for specific quantile
        qr_krn_models+=[KQR(alpha=q, **best_hyperparameters_krn).fit(X_train_scaled, y_train)]

        # save models to pickle
        pickle.dump(qr_krn_models[i], open(f'/Users/luca/Desktop/ThesisKernelMethods/dataset_applications/models/krn_qr_{i}.pkl', 'wb'))
        
        
    
   










    # # load data
    # df=pd.read_csv("/Users/luca/Desktop/GEFCom2014 Data/Load/Task 1/L1-train.csv", sep=",", decimal=".")
    
    # # clean data dates
    # dates = pd.date_range(start='2001-01-01 01:00:00', periods=len(df), freq='H')
    # df['DAY'] = dates.day
    # df['MONTH'] = dates.month
    # df['HOUR'] = dates.hour
    # df['YEAR'] = dates.year
    
    # # add info weekday; saturday, sunday
    # df['DAY_OF_WEEK'] = dates.dayofweek

    # # holidays
    # hol=holidays.US(years=range(2001, 2011))
    # df["IS_HOLIDAY"]=pd.Series(dates.date).isin(hol)
    # # convert it to categorical variable
    # df["IS_HOLIDAY"]=pd.Categorical(df["IS_HOLIDAY"].astype(int))
    

    # # drop not useful columns
    # # zoneid is dropped because are all 1, so it is not a meaningful independent variable
    # # the information in timestamp has been decomposed in four columns so we can drop it
    # df.drop(columns=['ZONEID', 'TIMESTAMP'], inplace=True)


    # # slice for training, from when we have load informations 2005
    # # slice from 2009 so it is less computations heavy
    # idx_start = df.loc[(df['DAY'] == 1) & (df["MONTH"]==1) & (df["YEAR"]==2009) & (df["HOUR"])==1].index[0]
    # df = df.loc[idx_start:]

    # # average temperatures
    # # select weather stations data using regex
    # weather_stat=df.filter(regex=("w.*")).columns.to_list()
    # # average them 
    # df['w_avg'] = df[weather_stat].mean(axis=1)

    # print(df)
    # # prova veloce
    # X=df[["DAY",  "MONTH",  "HOUR",  "DAY_OF_WEEK",  "IS_HOLIDAY",  "w_avg"]]
    # y=df["LOAD"]
    # # qr_tests(X,y)
    # # train data


    # # test data

    # # prediction for next month


    # # pinball loss


    
    