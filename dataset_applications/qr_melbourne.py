import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pprint import pprint

from quantile_forest import RandomForestQuantileRegressor as rfr

from sklearn.ensemble import GradientBoostingRegressor as gbr

from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import train_test_split

import statsmodels.regression.quantile_regression as qr 
import sys

from tqdm import tqdm



if __name__=="__main__":
    # load data
    df=pd.read_csv("/Users/luca/Desktop/ThesisKernelMethods/dataset_applications/temperatures_melbourne.csv", sep=";", decimal=".")

    quantiles = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    pinball_scores=pd.DataFrame(columns=["Linear qr", "Gbm qr", "Quantile forest"],index=quantiles)

    # plot data
    plt.plot(df["Yt-1"],df["Yt"],"o", alpha=0.2)
    plt.show()

    # train, test split
    X_train, X_test, y_train, y_test = train_test_split(df["Yt-1"], df["Yt"], test_size=0.20, random_state=4)

    eval_set=np.linspace(df["Yt"].min(), df["Yt"].max(), 100).T
    eval_set=eval_set.reshape(-1,1)
    X_train= X_train.values.reshape(-1, 1)
    X_test = X_test.values.reshape(-1, 1)


    # quantile regression: linear, random forest, gbm

    # linear quantile regression
    qr_models = [qr.QuantReg(y_train, X_train).fit(q=q) for q in quantiles]
    
    y_test_pred_qr=[qr_model.predict(X_test) for qr_model in qr_models]

    for i, q in enumerate(quantiles):
        pinball_scores.loc[q,"Linear qr"]=mean_pinball_loss(y_train,qr_models[i].predict(X_train), alpha=q)

    # plot model fit
    # linear quantile regression
    plt.plot(df["Yt-1"],df["Yt"],"o", alpha=0.2)
    for i,q in enumerate(quantiles):
        plt.plot(X_test,y_test_pred_qr[i], alpha=0.4, label=f"q={quantiles[i]}",color="black", linestyle="dashed")
    plt.legend()
    plt.title("Linear quantile regression")
    plt.show()
    
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
    for i,q in enumerate(tqdm(quantiles)):
        # define loss to tune
        neg_mean_pinball_loss_scorer = make_scorer(
        mean_pinball_loss,
        alpha=q,
        greater_is_better=False,  # maximize the negative of the loss
        )

        gbr_blueprint=gbr(loss="quantile", alpha=q, random_state=0)
        # tune hyperparameters
        # and fit data
        qr_gbr_models+=[HalvingRandomSearchCV(
            gbr_blueprint,
            param_grid_gbr,
            resource="n_estimators",
            max_resources=250,
            min_resources=50,
            scoring=neg_mean_pinball_loss_scorer,
            n_jobs=2,
            random_state=0,
        ).fit(X_train, y_train)]
        # list of prediction for each quantile
        y_test_pred_qr_gbr+=[qr_gbr_models[i].predict(eval_set)]

        pinball_scores.loc[q,"Gbm qr"]=mean_pinball_loss(y_train,qr_gbr_models[i].predict(X_train), alpha=q)

    # gradient boosting quantile regression
    plt.plot(df["Yt-1"],df["Yt"],"o", alpha=0.2)
    for i,q in enumerate(quantiles):
        plt.plot(eval_set,y_test_pred_qr_gbr[i], alpha=0.4, label=f"q={quantiles[i]}",color="black", linestyle="dashed")
    plt.legend()
    plt.title("Gradient boosting quantile regression")
    plt.show()


    # ranform forest quantile regression
    qr_rfr_models=[]
    y_test_pred_qr_rfr=[]

    param_grid_rfr = dict(
    max_depth=[2, 5, 10],
    min_samples_leaf=[1, 5, 10, 20],
    min_samples_split=[5, 10, 20, 30, 50],
)
    for i,q in enumerate(tqdm(quantiles)):
        # define loss to tune
        neg_mean_pinball_loss_scorer = make_scorer(
        mean_pinball_loss,
        alpha=q,
        greater_is_better=False,  # maximize the negative of the loss
        )

        rfr_blueprint=rfr(default_quantiles=q)
        # tune hyperparameters
        # and fit data
        qr_rfr_models+=[HalvingRandomSearchCV(
            rfr_blueprint,
            param_grid_rfr,
            resource="n_estimators",
            max_resources=250,
            min_resources=50,
            scoring=neg_mean_pinball_loss_scorer,
            n_jobs=2,
            random_state=0,
        ).fit(X_train, y_train)]
        # list of prediction for each quantile
        y_test_pred_qr_rfr+=[qr_rfr_models[i].predict(eval_set)]

        pinball_scores.loc[q,"Quantile forest"]=mean_pinball_loss(y_train,qr_rfr_models[i].predict(X_train), alpha=q)


    plt.plot(df["Yt-1"],df["Yt"],"o", alpha=0.2)

    for i,q in enumerate(quantiles):
        plt.plot(eval_set,y_test_pred_qr_rfr[i], alpha=0.4, label=f"q={quantiles[i]}",color="black", linestyle="dashed")
    plt.legend()
    plt.title("Quantile forest")
    plt.show()

    # create table with pinball loss    
    # pinball loss score
    avg_pinball_scores=pinball_scores.sum(axis=0).to_frame().T
    print(avg_pinball_scores)

    
    original_stdout=sys.stdout
    with open("/Users/luca/Desktop/ThesisKernelMethods/thesis/tables/qr_tables.txt", "w") as f:
        sys.stdout=f
        print(avg_pinball_scores.to_latex(index=True,
                  formatters={"name": str.upper},
                  float_format="{:.4f}".format))
    
        print(pinball_scores.to_latex(index=True,
                  formatters={"name": str.upper},
                  float_format="{:.4f}".format))
            
        sys.stdout=original_stdout