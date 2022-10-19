set -x
set -e

source ~/.bashrc
clear

/usr/bin/python3 /root/verifiable-unlearning/src/proof_model.py --trial_name models/linear_regression --dataset_name synthetic_100 --linear_regression
/usr/bin/python3 /root/verifiable-unlearning/src/proof_model.py --trial_name models/logistic_regression --dataset_name synthetic_100 --logistic_regression
/usr/bin/python3 /root/verifiable-unlearning/src/proof_model.py --trial_name models/neural_network_2 --dataset_name synthetic_100 --neural_network_2
/usr/bin/python3 /root/verifiable-unlearning/src/proof_model.py --trial_name models/neural_network_4 --dataset_name synthetic_100 --neural_network_4

/usr/bin/python3 /root/evaluation/scripts/models.py