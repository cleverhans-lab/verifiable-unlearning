set -x
set -e

source ~/.bashrc
clear

python3 /root/verifiable-unlearning/src/proof_model.py --trial_name classification/analcatdata_creditscore/verification --dataset_name analcatdata_creditscore --linear_regression --skip_regression
python3 /root/verifiable-unlearning/src/proof_model.py --trial_name classification/analcatdata_creditscore/linear_regression --dataset_name analcatdata_creditscore --linear_regression
python3 /root/verifiable-unlearning/src/proof_model.py --trial_name classification/analcatdata_creditscore/logistic_regression --dataset_name analcatdata_creditscore --logistic_regression
python3 /root/verifiable-unlearning/src/proof_model.py --trial_name classification/analcatdata_creditscore/neural_network_2 --dataset_name analcatdata_creditscore --neural_network_2
python3 /root/verifiable-unlearning/src/proof_model.py --trial_name classification/analcatdata_creditscore/neural_network_4 --dataset_name analcatdata_creditscore --neural_network_4

python3 /root/verifiable-unlearning/src/proof_model.py --trial_name classification/postoperative_patient_data/verification --dataset_name postoperative_patient_data --linear_regression --skip_regression
python3 /root/verifiable-unlearning/src/proof_model.py --trial_name classification/postoperative_patient_data/linear_regression --dataset_name postoperative_patient_data --linear_regression
python3 /root/verifiable-unlearning/src/proof_model.py --trial_name classification/postoperative_patient_data/logistic_regression --dataset_name postoperative_patient_data --logistic_regression
python3 /root/verifiable-unlearning/src/proof_model.py --trial_name classification/postoperative_patient_data/neural_network_2 --dataset_name postoperative_patient_data --neural_network_2
python3 /root/verifiable-unlearning/src/proof_model.py --trial_name classification/postoperative_patient_data/neural_network_4 --dataset_name postoperative_patient_data --neural_network_4

python3 /root/verifiable-unlearning/src/proof_model.py --trial_name classification/analcatdata_cyyoung9302/verification --dataset_name analcatdata_cyyoung9302 --linear_regression --skip_regression
python3 /root/verifiable-unlearning/src/proof_model.py --trial_name classification/analcatdata_cyyoung9302/linear_regression --dataset_name analcatdata_cyyoung9302 --linear_regression
python3 /root/verifiable-unlearning/src/proof_model.py --trial_name classification/analcatdata_cyyoung9302/logistic_regression --dataset_name analcatdata_cyyoung9302 --logistic_regression
python3 /root/verifiable-unlearning/src/proof_model.py --trial_name classification/analcatdata_cyyoung9302/neural_network_2 --dataset_name analcatdata_cyyoung9302 --neural_network_2
python3 /root/verifiable-unlearning/src/proof_model.py --trial_name classification/analcatdata_cyyoung9302/neural_network_4 --dataset_name analcatdata_cyyoung9302 --neural_network_4

python3 /root/verifiable-unlearning/src/proof_model.py --trial_name classification/corral/verification --dataset_name corral --linear_regression --skip_regression
python3 /root/verifiable-unlearning/src/proof_model.py --trial_name classification/corral/linear_regression --dataset_name corral --linear_regression
python3 /root/verifiable-unlearning/src/proof_model.py --trial_name classification/corral/logistic_regression --dataset_name corral --logistic_regression
python3 /root/verifiable-unlearning/src/proof_model.py --trial_name classification/corral/neural_network_2 --dataset_name corral --neural_network_2
python3 /root/verifiable-unlearning/src/proof_model.py --trial_name classification/corral/neural_network_4 --dataset_name corral --neural_network_4

/usr/bin/python3 /root/evaluation/scripts/datasets.py