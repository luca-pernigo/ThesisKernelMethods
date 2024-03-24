#!/bin/bash
# code for moving test data in respective folder
for((i=2;i<=15;i++))
do
    i_minus_1=$((i - 1))
    i_minus_1=$((i-1))
    # get test data
    cp "/Users/luca/Desktop/GEFCom2014 Data/Load/Task ${i}/L${i}-train.csv" "/Users/luca/Desktop/GEFCom2014 Data/Load/Task ${i_minus_1}/L${i_minus_1}-test.csv"
done

cp "/Users/luca/Desktop/GEFCom2014 Data/Load/Solution to Task 15/solution15_L_temperature.csv" "/Users/luca/Desktop/GEFCom2014 Data/Load/Task 15/L15-test.csv"
