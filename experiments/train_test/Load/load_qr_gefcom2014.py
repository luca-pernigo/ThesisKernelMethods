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

# from kernel_quantile_regression.kqr import KQR
sys.path.append("/Users/luca/Desktop/ThesisKernelMethods/experiments/")
from kqr import KQR




if __name__=="__main__":
    ith=int(sys.argv[1])
    # load data
    df=pd.read_csv("experiments/Data/Load/L-train.csv")
    # in the load track each task predicts the month=task number -3
    df=df[df["MONTH"]==(ith-3)][-1440:]
    X_train=df[["DAY",  "HOUR",  "DAY_OF_WEEK",  "IS_HOLIDAY",  "w_avg"]]
    y_train=df["LOAD"]

    # 99 quantiles
    quantiles = [i/100 for i in range(1,100)]

    # kernel quantile regression
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    qr_krn_models=[]
    y_test_pred_qr_krn=[]

    m=len(y_train)
    
    param_grid_krn = dict(
        C=[1e-1,1e-2,1, 5, 10,1e2,1e4],
        gamma=[1e-1,1e-2,0.5,1, 5, 10, 20]
    )
    # C=[1/(m*10e-6),1/(m*10e-5),1/(m*10e-4),1/(m*10e-3),1/(m*10e-2),1/(m*10e-1),1/(m*10e-0),1/(m*10e1)],
    # gamma=[2**-6,2**-5,2**-4, 2**-3,2**-2,2**-1,1, 2, 2**2, 2**3,2**4,2**5,2**6]   
    
    # nu=[0.5, 1.5, 2.5] 

    ktype="a_laplacian"
    # folder per kernel type
    
    
    for i,q in enumerate(tqdm(quantiles)):
        
        # define loss to tune
        neg_mean_pinball_loss_scorer = make_scorer(
        mean_pinball_loss,
        alpha=q,
        greater_is_better=False,
        )

        krn_blueprint=KQR(alpha=q, kernel_type=ktype)
        best_hyperparameters_krn=HalvingRandomSearchCV(
                krn_blueprint,
                param_grid_krn,
                scoring=neg_mean_pinball_loss_scorer,
                n_jobs=2,
                random_state=0,
            ).fit(X_train_scaled, y_train).best_params_

        # fit data for specific quantile
        qr_krn_models+=[KQR(alpha=q, **best_hyperparameters_krn, kernel_type=ktype).fit(X_train_scaled, y_train)]

        # save models to pickle
        pickle.dump(qr_krn_models[i], open(f'experiments/train_test/Load/{ktype}/task {ith}/krn_qr_{i}.pkl', 'wb'))
        
