# code to save predictions in the same format of GEFCom challenge
# ZONEID| TIMESTAMP| 0.01|...|0.99

# save data as csv, each task in its folder
# from pd dataframe with rows timestamp
# columns quantile



import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler
import sys

from tqdm import tqdm


def predict_save(ith):
    # df=pd.read_csv(f"/Users/luca/Desktop/GEFCom2014 Data/Load/Task {ith}/L{ith}-test_clean.csv")
    df=df=pd.read_csv(f"/Users/luca/Desktop/GEFCom2014 Data/Load/Task {ith}/L{ith}-avg_test_clean.csv")
    df["w_avg"]=pd.read_csv(f"/Users/luca/Desktop/GEFCom2014 Data/Load/Task {ith}/temp.csv")

    X_test=df[["DAY",  "MONTH",  "HOUR",  "DAY_OF_WEEK",  "IS_HOLIDAY",  "w_avg"]]
    y_test=df["LOAD"]

    quantiles = [i/100 for i in range(1,100)]

    # we need to scale X_test
    df_train=pd.read_csv("/Users/luca/Desktop/GEFCom2014 Data/Load/L2009-train.csv")
    X_train=df_train[["DAY",  "MONTH",  "HOUR",  "DAY_OF_WEEK",  "IS_HOLIDAY",  "w_avg"]]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    
    df_template_submission=pd.read_csv(f"/Users/luca/Desktop/GEFCom2014 Data/Load/Task {ith}/L{ith}-test.csv")
    
    if ith==15:
        df_template_submission=pd.read_csv(f"/Users/luca/Desktop/GEFCom2014 Data/Load/Task {ith}/L15-benchmark.csv")
        output_df=df_template_submission[["ZONEID", "TIMESTAMP"]].copy()
        
    else:
        output_df=df_template_submission[["ZONEID", "TIMESTAMP"]].copy()
    
    for i,q in enumerate(quantiles):
        krn_q=pickle.load(open(f"/Users/luca/Desktop/ThesisKernelMethods/dataset_applications/models/krn_qr_{i}.pkl", 'rb'))
        y_predict_q=krn_q.predict(X_test_scaled)

        # save prediction for qth quantile
        output_df[f"{q}"]=pd.Series(y_predict_q)

    output_df.to_csv(f"/Users/luca/Desktop/GEFCom2014 Data/Load/Task {ith}/L{ith}-our_model_prediction.csv", index=False)
    return

if __name__=="__main__":
    i=int(sys.argv[1])
    predict_save(i)
