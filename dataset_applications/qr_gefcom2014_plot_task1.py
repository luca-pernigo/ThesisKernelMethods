# code to create plot of 90% interval

# take data saved as predictions 5, 50, 95 quantile from each task,
# vertical stack data
# plot 

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

if __name__=="__main__":
    
    # load test task 1
    y_test=pd.read_csv("/Users/luca/Desktop/GEFCom2014 Data/Load/Task 1/L1-test_clean.csv")
    
    
    # load prediction for task 1
    y_predict_q=pd.read_csv("/Users/luca/Desktop/GEFCom2014 Data/Load/Task 1/L1-our_model_prediction.csv")

    # select quantiles
    y_predict_5=y_predict_q["0.05"]
    y_predict_50=y_predict_q["0.5"]
    y_predict_95=y_predict_q["0.95"]

    # dummy x for plotting
    x=np.arange(len(y_test))
    # ticks for the days, (including also hours adds clutter)
    tick_array = [i for i in range(24,744+1,24)]
    # label ticks
    labels=[int(i/24) for i in tick_array]
    
    # plot quantile range
    plt.plot(y_test["LOAD"],color="black")
    
    plt.fill_between(x,y_predict_q["0.05"],y_test["LOAD"], alpha=0.4, color="green", edgecolor="red")
    plt.fill_between(x,y_test["LOAD"],y_predict_q["0.95"], alpha=0.4, color="green", edgecolor="red")
    
    plt.xticks(ticks=tick_array, labels=labels, rotation=45)
    
    plt.xlabel("Days")
    plt.ylabel("Load in MW")
    plt.show()


    # plot true vs median
    plt.plot(x,y_test["LOAD"],color="black", label="y_test")
    plt.plot(x,y_predict_50, "red", alpha=0.4, label="0.5 prediction")
    plt.xticks(ticks=tick_array, labels=labels, rotation=45)
    
    plt.xlabel("Days")
    plt.ylabel("Load in MW")
    plt.legend()
    plt.show()

