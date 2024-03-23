from kqr import KQR

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
from sklearn.preprocessing import StandardScaler

import statsmodels.regression.quantile_regression as qr 
import sys

from tqdm import tqdm

from cvxopt import matrix, spmatrix, sparse
def qr_tests(X,y,write_to_path):
    # quantiles
    quantiles = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    # table for output results
    pinball_scores=pd.DataFrame(columns=["Linear qr", "Gbm qr", "Quantile forest", "Kernel qr"],index=quantiles)
    mae_scores=pd.DataFrame(columns=["Linear qr", "Gbm qr", "Quantile forest", "Kernel qr"])

    # train, test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


    # # quantile regression: linear, random forest, gbm

    # linear quantile regression
    qr_models = [qr.QuantReg(y_train, X_train).fit(q=q) for q in quantiles]
    
    y_test_pred_qr=[qr_model.predict(X_test) for qr_model in qr_models]

    # compute pinball loss scores on test data
    for i, q in enumerate(quantiles):
        pinball_scores.loc[q,"Linear qr"]=mean_pinball_loss(y_test,qr_models[i].predict(X_test), alpha=q)

    
    # gbm quantile regressor
    # set grid parameters for hypertuning
    qr_gbr_models=[]
    y_test_pred_qr_gbr=[]

    param_grid_gbr = dict(
    learning_rate=[0.05, 0.1, 0.2],
    max_depth=[2, 5, 10],
    min_samples_leaf=[1, 5, 10, 20],
    min_samples_split=[5, 10, 20, 30, 50],
    )
    # define loss to tune
    neg_mean_pinball_loss_scorer_05 = make_scorer(
    mean_pinball_loss,
    alpha=0.5,
    greater_is_better=False,  # maximize the negative of the loss
    )
    gbr_blueprint=gbr(loss="quantile", alpha=0.5, random_state=0)
    # tune hyperparameters
    # and fit data
    best_hyperparameters_gbm=HalvingRandomSearchCV(
            gbr_blueprint,
            param_grid_gbr,
            resource="n_estimators",
            max_resources=250,
            min_resources=50,
            scoring=neg_mean_pinball_loss_scorer_05,
            n_jobs=2,
            random_state=0,
        ).fit(X_train, y_train).best_params_
    
    for i,q in enumerate(tqdm(quantiles)):

        # fit data for specific quantile
        qr_gbr_models+=[gbr(loss="quantile", alpha=q, random_state=0,**best_hyperparameters_gbm).fit(X_train, y_train)]
        
        # list of prediction for each quantile
        y_test_pred_qr_gbr+=[qr_gbr_models[i].predict(X_test)]

        pinball_scores.loc[q,"Gbm qr"]=mean_pinball_loss(y_test,qr_gbr_models[i].predict(X_test), alpha=q)

    
    # ranform forest quantile regression
    qr_rfr_models=[]
    y_test_pred_qr_rfr=[]

    param_grid_rfr = dict(
    max_depth=[2, 5, 10],
    min_samples_leaf=[1, 5, 10, 20],
    min_samples_split=[5, 10, 20, 30, 50],
    )
    rfr_blueprint=rfr(default_quantiles=0.5)
    best_hyperparameters_rff=HalvingRandomSearchCV(
            rfr_blueprint,
            param_grid_rfr,
            resource="n_estimators",
            max_resources=250,
            min_resources=50,
            scoring=neg_mean_pinball_loss_scorer_05,
            n_jobs=2,
            random_state=0,
        ).fit(X_train, y_train).best_params_
    
    for i,q in enumerate(tqdm(quantiles)):

        # fit data for specific quantile
        qr_rfr_models+=[rfr(default_quantiles=q, **best_hyperparameters_rff).fit(X_train, y_train)]
        
        # list of prediction for each quantile
        y_test_pred_qr_rfr+=[qr_rfr_models[i].predict(X_test)]
      
        pinball_scores.loc[q,"Quantile forest"]=mean_pinball_loss(y_test,qr_rfr_models[i].predict(X_test), alpha=q)


    # kernel quantile regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

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
        
        # list of prediction for each quantile
        y_test_pred_qr_krn+=[qr_krn_models[i].predict(X_test_scaled)]
      
        pinball_scores.loc[q,"Kernel qr"]=mean_pinball_loss(y_test,qr_krn_models[i].predict(X_test_scaled), alpha=q)


    # create table with pinball loss    
    # pinball loss score
    avg_pinball_scores=pinball_scores.sum(axis=0).to_frame().T
    print(avg_pinball_scores)
    # mae score
    mae_scores.loc[0,"Linear qr"]=mean_absolute_error(y_test, y_test_pred_qr[5])
    mae_scores.loc[0,"Gbm qr"]=mean_absolute_error(y_test, y_test_pred_qr_gbr[5])
    mae_scores.loc[0,"Quantile forest"]=mean_absolute_error(y_test, y_test_pred_qr_rfr[5])
    mae_scores.loc[0,"Kernel qr"]=mean_absolute_error(y_test, y_test_pred_qr_krn[5])

    
    original_stdout=sys.stdout
    with open(f"/Users/luca/Desktop/ThesisKernelMethods/thesis/tables/{write_to_path}.txt", "w") as f:
        sys.stdout=f
        print(avg_pinball_scores.to_latex(index=True,
                  formatters={"name": str.upper},
                  float_format="{:.4f}".format))
    
        print(pinball_scores.to_latex(index=True,
                  formatters={"name": str.upper},
                  float_format="{:.4f}".format))
        
        print(mae_scores.to_latex(index=True,
                  formatters={"name": str.upper},
                  float_format="{:.4f}".format))
            
        sys.stdout=original_stdout