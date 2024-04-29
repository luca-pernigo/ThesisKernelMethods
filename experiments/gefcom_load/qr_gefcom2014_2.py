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
    df=pd.read_csv("/Users/luca/Desktop/GEFCom2014 Data/Load/L2009-train_2.csv")
    
    X_train=df[[f"w{i}" for i in range(1,26)]+["DAY",  "MONTH",  "HOUR",  "DAY_OF_WEEK",  "IS_HOLIDAY",  "w_avg"]]
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
        pickle.dump(qr_krn_models[i], open(f'/Users/luca/Desktop/ThesisKernelMethods/dataset_applications/models_2/krn_qr_{i}.pkl', 'wb'))
    