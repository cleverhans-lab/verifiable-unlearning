set -x
set -e

source ~/.bashrc

/usr/bin/python3 /root/verifiable-unlearning/src/run.py \
    --trial_name datasets/analcatdata_creditscore \
    --technique retraining --mode train \
    --dataset_name analcatdata_creditscore \
    --classifier linear_regression \
    --proof_system nizk

/usr/bin/python3 /root/verifiable-unlearning/src/run.py \
    --trial_name datasets/postoperative_patient_data \
    --technique retraining --mode train \
    --dataset_name postoperative_patient_data \
    --classifier linear_regression \
    --proof_system nizk

/usr/bin/python3 /root/verifiable-unlearning/src/run.py \
    --trial_name datasets/analcatdata_cyyoung9302 \
    --technique retraining --mode train \
    --dataset_name analcatdata_cyyoung9302 \
    --classifier linear_regression \
    --proof_system nizk

/usr/bin/python3 /root/verifiable-unlearning/src/run.py \
    --trial_name datasets/corral \
    --technique retraining --mode train \
    --dataset_name corral \
    --classifier linear_regression \
    --proof_system nizk

/usr/bin/python3 /root/verifiable-unlearning/src/run.py \
    --trial_name datasets/analcatdata_lawsuit \
    --technique retraining --mode train \
    --dataset_name analcatdata_lawsuit \
    --classifier linear_regression \
    --proof_system nizk

/usr/bin/python3 /root/verifiable-unlearning/src/run.py \
    --trial_name datasets/breast_cancer \
    --technique retraining --mode train \
    --dataset_name breast_cancer \
    --classifier linear_regression \
    --proof_system nizk

/usr/bin/python3 /root/verifiable-unlearning/src/run.py \
    --trial_name datasets/monk3 \
    --technique retraining --mode train \
    --dataset_name monk3 \
    --classifier linear_regression \
    --proof_system nizk
