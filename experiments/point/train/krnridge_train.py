import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import pickle

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn import preprocessing

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV,TimeSeriesSplit,HalvingGridSearchCV
from sklearn.kernel_ridge import KernelRidge


df_train=pd.read_csv("/Users/luca/Desktop/GEFCom2014 Data/Load/L-train.csv")

y_train=df_train["LOAD"]
X_train=df_train[["DAY","MONTH","HOUR","DAY_OF_WEEK","IS_HOLIDAY","w_avg"]]

df_test=pd.read_csv(f"/Users/luca/Desktop/GEFCom2014 Data/Load/Task 1/L1-test_clean.csv")

X_test=df_test[["DAY",  "MONTH",  "HOUR",  "DAY_OF_WEEK",  "IS_HOLIDAY",  "w_avg"]]
y_test=df_test["LOAD"]


scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


print("train model")
krn_ridge = KernelRidge(alpha=1, gamma=1, kernel="rbf")
krn_ridge.fit(X_train_scaled, y_train)


y_hat=krn_ridge.predict(X_test_scaled)


pickle.dump(krn_ridge, open('/Users/luca/Desktop/ThesisKernelMethods/experiments/point/models/krn_ridge_load.pkl', 'wb'))


krn_ridge=pickle.load(open('/Users/luca/Desktop/ThesisKernelMethods/experiments/point/models/krn_ridge_load.pkl', 'rb'))


plt.figure(figsize=(15,5))
plt.plot(y_hat)
plt.plot(y_test)

plt.plot()


mean_squared_error(y_test, y_hat)