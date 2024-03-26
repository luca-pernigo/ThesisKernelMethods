#!/bin/bash

# clean
rm /Users/luca/Desktop/ThesisKernelMethods/thesis/tables/qr_gefcom2014.txt

for((i=1;i<=15;++i))
do
    python /Users/luca/Desktop/ThesisKernelMethods/dataset_applications/qr_gefcom2014_test.py $i

done