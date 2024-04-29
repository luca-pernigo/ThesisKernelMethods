
import pandas as pd
import pickle

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_pinball_loss
from sklearn.preprocessing import StandardScaler
import sys

from tqdm import tqdm


def test(ith):

    task_month={4:7,5:7,6:7,7:7,8:7,9:7,10:7,11:7,12:7,13:12,14:12,15:12}

    df=pd.read_csv(f"/Users/luca/Desktop/ThesisKernelMethods/dataset_applications/gefcom_price/Price/Task {ith}/Task{ith}_P_test.csv")

    X_test=df[["MONTH","DAY","HOUR","Forecasted Total Load","Forecasted Zonal Load"]]
    y_test=df["Zonal Price"]

    quantiles = [i/100 for i in range(1,100)]

    # we need to scale X_test
    df_train=pd.read_csv(f"/Users/luca/Desktop/ThesisKernelMethods/dataset_applications/gefcom_price/Price/Task {ith}/Task{ith}_P_train.csv")
    df_train=df_train[(df_train["MONTH"]==task_month[ith])]

    X_train=df_train[["MONTH","DAY","HOUR","Forecasted Total Load","Forecasted Zonal Load"]]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pinball_tot=0

    # predict
    df_template_submission=pd.read_csv(f"/Users/luca/Desktop/ThesisKernelMethods/dataset_applications/gefcom_price/Price/Task {ith}/Benchmark{ith}_P.csv")

    df_predict=df_template_submission[["ZONEID", "timestamp"]].copy()

    for i,q in enumerate(quantiles):
        krn_q=pickle.load(open(f"/Users/luca/Desktop/ThesisKernelMethods/dataset_applications/gefcom_price/models cv/task {ith}/krn_qr_{i}.pkl", "rb"))
        y_predict_q=krn_q.predict(X_test_scaled)
        pinball_q=mean_pinball_loss(y_test,y_predict_q, alpha=q)
        print(f"pinball loss quantile {q}: ", pinball_q)
        pinball_tot+=pinball_q

        df_predict[f"{q}"]=pd.Series(y_predict_q)

    a = df_predict[[f"{i/100}" for i in range(1,100)]].values
    a.sort(axis=1)
    res=pd.DataFrame(a, df_predict.index)

    print("total quantile: ", pinball_tot/len(quantiles))

    pinball_tot=0
    for i,q in enumerate(quantiles):
        predict=res.iloc[:,i]
        pinball_q=mean_pinball_loss(y_test,predict, alpha=q)
        print(f"pinball loss quantile {q}: ", pinball_q)
        pinball_tot+=pinball_q

    print("total quantile: ", pinball_tot/len(quantiles))
    ans=pinball_tot/len(quantiles)
    
    return ans

if __name__=="__main__":
    i=int(sys.argv[1])
    ans=test(i)