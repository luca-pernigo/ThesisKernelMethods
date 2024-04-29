#!/bin/bash

# for each task clean test dataset
for ((i=1;i<=15;i++))
do
    python /Users/luca/Desktop/ThesisKernelMethods/dataset_applications/gefcom_clean_test.py $i

done