#!/bin/bash

# code to extract the timestamp and zoneid from test data so that the prediction is in the same format of guidelines


for((i=1;i<=15;++i))
do
    cut -d , -f 1-2 /Users/luca/Desktop/GEFCom2014\ Data/Load/Task\ $i/L$i-test.csv >> /Users/luca/Desktop/GEFCom2014\ Data/Load/Task\ $i/L$i-timestamp.csv
done