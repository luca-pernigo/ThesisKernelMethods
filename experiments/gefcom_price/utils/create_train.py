

import pandas as pd
import sys

import holidays

import miscellaneous
# bash script
# for ((i=1;i<=15;i++)); do python create_train.py "/Users/luca/Desktop/ThesisKernelMethods/dataset_applications/gefcom_price/Price/Task $i/Task${i}_P.csv"; done

def create_train(file):
    df=miscellaneous.clean_time(file)

    df_train=df[0:-24]

    # order columns
    df_train=miscellaneous.order_columns(df_train)
    # save
    n=miscellaneous.get_task_number(file)
    df_train.to_csv(f"/Users/luca/Desktop/ThesisKernelMethods/dataset_applications/gefcom_price/Price/Task {n}/Task{n}_P_train.csv", index=False)

if __name__=="__main__":
    file=sys.argv[1]
    # clean passed file
    create_train(file)