from kernel_quantile_regression.kqr import KQR
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pprint import pprint

from quantile_forest import RandomForestQuantileRegressor as rfr

from sklearn.ensemble import GradientBoostingRegressor as gbr

from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import train_test_split

import statsmodels.regression.quantile_regression as qr 
import sys

from tqdm import tqdm



if __name__=="__main__":
    # load data
    df=pd.read_csv("Data/temperatures_melbourne.csv", sep=";", decimal=".")

    # quantiles
    quantiles = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    # table for output results
    
    # train, test split
    X_train, X_test, y_train, y_test = train_test_split(df["Yt-1"], df["Yt"], test_size=0.20, random_state=0)

    # equally space dataset for plotting
    eval_set=np.linspace(df["Yt-1"].min(), df["Yt-1"].max(), 100).T

    # reshape data to be predicted
    X_train= X_train.values.reshape(-1, 1)
    X_test = X_test.values.reshape(-1, 1)
    eval_set=eval_set.reshape(-1,1)


    # define loss to tune
    neg_mean_pinball_loss_scorer_05 = make_scorer(
    mean_pinball_loss,
    alpha=0.5,
    greater_is_better=False,  # maximize the negative of the loss
    )
    
    # kernel quantile regression
    qr_krn_models=[]
    y_test_pred_qr_krn=[]
    ktype="matern_2.5"

    # gamma=[1e-1,1e-2,1,5,10,20]
    # sigma=[1e-1,1e-2,1,5,10,20]

    # polynomial c=[0, 1, 10, 100],
    # d=[2,3,4,5,8]

    # sigmoid
    # c=[0, 1, 10, 100],
    # gamma=[1e-1,1e-2,1,5,10,20]

    # periodic
    # p=[0,1,10,0.5]

    # matern
    # nu=[0.5, 1.5, 2.5, float('inf')]
    # gamma=[1e-1,1e-2,1,5,10,20]

    param_grid_krn = dict(
    C=[0.1,1, 5, 10],
    gamma=[1/(1e-1),1/(1e-2),1/(1),1/(5),1/(10),1/(20)]
    )
    krn_blueprint=KQR(alpha=0.5, kernel_type=ktype)
    best_hyperparameters_krn=HalvingRandomSearchCV(
            krn_blueprint,
            param_grid_krn,
            scoring=neg_mean_pinball_loss_scorer_05,
            n_jobs=2,
            random_state=0,
        ).fit(X_train, y_train).best_params_
    
    for i,q in enumerate(tqdm(quantiles)):

        # fit data for specific quantile
        qr_krn_models+=[KQR(alpha=q, kernel_type=ktype, **best_hyperparameters_krn).fit(X_train, y_train)]
        
        # list of prediction for each quantile
        y_test_pred_qr_krn+=[qr_krn_models[i].predict(X_test)]
      


    # mae score
    print("mae", f"{mean_absolute_error(y_test, y_test_pred_qr_krn[5]):.6f}")

    pinball=0
    for i,q in enumerate(quantiles):
        print(f"{mean_pinball_loss(y_test,qr_krn_models[i].predict(X_test), alpha=q):.6f}", "&")
        pinball+=mean_pinball_loss(y_test,qr_krn_models[i].predict(X_test), alpha=q)

print("total pinball loss: ", pinball)
print("best hyperparameters: ", best_hyperparameters_krn)
#    Linear qr     Gbm qr      Quantile forest  Kernel qr rbf gaussian   Laplacian   Rbf gaussian x laplacian
#    11.278895     10.317612   10.370558        10.031708                10.056884   10.150826       

#    Cosine       Linear       Polynomial       Sigmoid                 Chi Squared  Matern     
#    16.253973    10.463867    11.238393        16.253973               10.023732       10.021369  

#    Periodic
#    15.946272         