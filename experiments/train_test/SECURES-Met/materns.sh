#!/usr/bin/env bash
# script for rerunning materns
python experiments/train_test/SECURES-Met/secures_qr.py "DE" "a_laplacian"
python experiments/train_test/SECURES-Met/secures_qr.py "DE" "gaussian_rbf"
python experiments/train_test/SECURES-Met/secures_qr.py "DE" "matern_0.5"
python experiments/train_test/SECURES-Met/secures_qr.py "DE" "matern_1.5"
python experiments/train_test/SECURES-Met/secures_qr.py "DE" "matern_2.5"


python experiments/train_test/SECURES-Met/secures_qr.py "CH" "a_laplacian"
python experiments/train_test/SECURES-Met/secures_qr.py "CH" "gaussian_rbf"
python experiments/train_test/SECURES-Met/secures_qr.py "CH" "matern_0.5"
python experiments/train_test/SECURES-Met/secures_qr.py "CH" "matern_1.5"
python experiments/train_test/SECURES-Met/secures_qr.py "CH" "matern_2.5"


python experiments/train_test/SECURES-Met/secures_qr.py "AT" "a_laplacian"
python experiments/train_test/SECURES-Met/secures_qr.py "AT" "gaussian_rbf"
python experiments/train_test/SECURES-Met/secures_qr.py "AT" "matern_0.5"
python experiments/train_test/SECURES-Met/secures_qr.py "AT" "matern_1.5"
python experiments/train_test/SECURES-Met/secures_qr.py "AT" "matern_2.5"