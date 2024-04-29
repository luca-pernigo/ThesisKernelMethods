
import pandas as pd
import pickle

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_pinball_loss
from sklearn.preprocessing import StandardScaler
import sys

from tqdm import tqdm


def test(ith):
    df=pd.read_csv(f"/Users/luca/Desktop/GEFCom2014 Data/Load/Task {ith}/L{ith}-avg_test_clean.csv")

    X_test=df[["DAY",  "MONTH",  "HOUR",  "DAY_OF_WEEK",  "IS_HOLIDAY",  "w_avg"]]
    y_test=df["LOAD"]

    quantiles = [i/100 for i in range(1,100)]

    # we need to scale X_test
    df_train=pd.read_csv("/Users/luca/Desktop/GEFCom2014 Data/Load/L2009-train.csv")
    X_train=df_train[["DAY",  "MONTH",  "HOUR",  "DAY_OF_WEEK",  "IS_HOLIDAY",  "w_avg"]]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pinball_tot=0

    for i,q in enumerate(quantiles):
        krn_q=pickle.load(open(f"/Users/luca/Desktop/ThesisKernelMethods/dataset_applications/models/krn_qr_{i}.pkl", 'rb'))
        pinball_q=mean_pinball_loss(y_test,krn_q.predict(X_test_scaled), alpha=q)
        print(f"pinball loss quantile {q}: ", pinball_q)
        pinball_tot+=pinball_q
    
    print("total quantile: ", pinball_tot/len(quantiles))
    ans=pinball_tot/len(quantiles)
    return ans

if __name__=="__main__":
    i=int(sys.argv[1])
    ans=test(i)

    original_stdout=sys.stdout
    with open(f"/Users/luca/Desktop/ThesisKernelMethods/thesis/tables/qr_gefcom2014_w_avg.txt", "a") as f:
        sys.stdout=f
        # print(f"Average pinball loss, task n. {i}: ")
        print(f"{ans:.4f}")
        print("&")

        sys.stdout=original_stdout
