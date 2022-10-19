# Verifiable and Provably Secure Machine Unlearning

This is the code repository accompaning our paper [Verifiable and Provably Secure Machine Unlearning](https://arxiv.org/abs/2210.09126).

> Machine unlearning aims to remove points from the training dataset of a machine learning model after training; for example when a user requests their data to be deleted. While many machine unlearning methods have been proposed, none of them enable users to audit the unlearning procedure and verify that their data was indeed unlearned. To address this, we define the first cryptographic framework to formally capture the security of verifiable machine unlearning. While our framework is generally applicable to different approaches, its advantages are perhaps best illustrated by our instantiation for the canonical approach to unlearning: retraining without the data to be unlearned. In our protocol, the server first computes a proof that the model was trained on a dataset $D$. Given a user data point $d$, the server then computes a proof of unlearning that shows that $d \notin D$. We realize our protocol using a SNARK and Merkle trees to obtain proofs of update and unlearning on the data. Based on cryptographic assumptions, we then present a game-based proof that our instantiation is secure. Finally, we validate the practicality of our constructions for unlearning in linear regression, logistic regression, and neural networks.

## Prerequisites

We implemented our framework based on [ZoKrates](https://zokrates.github.io) (Version 0.8.3) and created a Dockerfile with all necessary tools to reproduce the results from the paper. It can be build via

```
git clone git@github.com:cleverhans-lab/verifiable-unlearning.git ~/verifiable-unlearning
docker build -t verifiable-unlearning .
```

## Experiments

To reproduce the experiments from the paper, we prepared various convenience scripts to run within the container. You can run the container with

```
docker run --rm -it verifiable-unlearning
```

Note: the `--rm` flags automatically deletes the container when stopped (adjust as appropiate).

The scripts can be found in directory `evaluation` and run as follows:
```
/root/verifiable-unlearning/evaluation/trials_benchmarks.sh
/root/verifiable-unlearning/evaluation/trials_models.sh
/root/verifiable-unlearning/evaluation/trials_datasets.sh
```

Results and plots are saved to the `evaluation` directory.
