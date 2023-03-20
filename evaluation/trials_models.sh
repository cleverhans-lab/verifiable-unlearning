set -x
set -e

source ~/.bashrc

/usr/bin/python3 /root/verifiable-unlearning/src/run.py \
    --trial_name models/linear_regression \
    --technique retraining --mode train \
    --no_samples_D_prev 0 \
    --no_samples_D_plus 100 \
    --no_samples_U_prev 0 \
    --no_samples_U_plus 0 \
    --dataset_name synthetic_10 \
    --classifier linear_regression \
    --proof_system nizk

/usr/bin/python3 /root/verifiable-unlearning/src/run.py \
    --trial_name models/logistic_regression \
    --technique retraining --mode train \
    --no_samples_D_prev 0 \
    --no_samples_D_plus 100 \
    --no_samples_U_prev 0 \
    --no_samples_U_plus 0 \
    --dataset_name synthetic_10 \
    --classifier logistic_regression \
    --proof_system nizk

/usr/bin/python3 /root/verifiable-unlearning/src/run.py \
    --trial_name models/neural_network_2 \
    --technique retraining --mode train \
    --no_samples_D_prev 0 \
    --no_samples_D_plus 100 \
    --no_samples_U_prev 0 \
    --no_samples_U_plus 0 \
    --dataset_name synthetic_10 \
    --classifier neural_network_2 \
    --proof_system nizk

/usr/bin/python3 /root/verifiable-unlearning/src/run.py \
    --trial_name models/neural_network_4 \
    --technique retraining --mode train \
    --no_samples_D_prev 0 \
    --no_samples_D_plus 100 \
    --no_samples_U_prev 0 \
    --no_samples_U_plus 0 \
    --dataset_name synthetic_10 \
    --classifier neural_network_4 \
    --proof_system nizk
