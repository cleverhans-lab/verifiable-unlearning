set -x
set -e

source ~/.bashrc
clear

/usr/bin/python3 /root/verifiable-unlearning/src/proof_membership.py --trial_name benchmarks/membership/10 --no_samples_D 10 --no_samples_U_prev 1 --no_samples_U_plus 1
/usr/bin/python3 /root/verifiable-unlearning/src/proof_membership.py --trial_name benchmarks/membership/100 --no_samples_D 100 --no_samples_U_prev 10 --no_samples_U_plus 1
/usr/bin/python3 /root/verifiable-unlearning/src/proof_membership.py --trial_name benchmarks/membership/1000 --no_samples_D 1000 --no_samples_U_prev 100 --no_samples_U_plus 1

/usr/bin/python3 /root/verifiable-unlearning/src/proof_dataset.py --trial_name benchmarks/dataset/10 --no_samples_D 10 --no_samples_U_prev 1 --no_samples_U_plus 1
/usr/bin/python3 /root/verifiable-unlearning/src/proof_dataset.py --trial_name benchmarks/dataset/100 --no_samples_D 100 --no_samples_U_prev 10 --no_samples_U_plus 1
/usr/bin/python3 /root/verifiable-unlearning/src/proof_dataset.py --trial_name benchmarks/dataset/1000 --no_samples_D 1000 --no_samples_U_prev 100 --no_samples_U_plus 1

/usr/bin/python3 /root/verifiable-unlearning/src/proof_model.py --trial_name benchmarks/model/10 --dataset_name synthetic_10 --linear_regression
/usr/bin/python3 /root/verifiable-unlearning/src/proof_model.py --trial_name benchmarks/model/100 --dataset_name synthetic_100 --linear_regression
/usr/bin/python3 /root/verifiable-unlearning/src/proof_model.py --trial_name benchmarks/model/1000 --dataset_name synthetic_1000 --linear_regression

/usr/bin/python3 /root/evaluation/scripts/benchmarks.py