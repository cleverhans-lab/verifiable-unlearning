# Verifiable and Provably Secure Machine Unlearning

This is the code repository accompaning our paper [Verifiable and Provably Secure Machine Unlearning](https://arxiv.org/abs/2210.09126).

> Machine unlearning aims to remove points from the training dataset of a machine learning model after training; for example when a user requests their data to be deleted. While many machine unlearning methods have been proposed, none of them enable users to audit the procedure. Furthermore, recent work shows a user is unable to verify if their data was unlearned from an inspection of the model alone. Rather than reasoning in model space, our key insight is thus to address verifiable unlearning through algorithmic guarantees. We identify the necessary requirements, and based on these, define the first cryptographic framework to formally capture the syntax and security for verifiable machine unlearning. Our framework is generally applicable to different unlearning techniques. We instantiate the framework using SNARKs and hash chains. More specifically, the server first computes a proof that the model was trained on a dataset $D$. Given a user data point $d$ requested to be deleted, the server updates the model using an unlearning algorithm. It then provides a proof of the correct execution of unlearning and that $d \notin D'$, where $D'$ is the new training dataset. Based on cryptographic assumptions, we then present a game-based proof that our instantiation is secure. Finally, we implement the protocol for three different unlearning techniques and validate the feasibility for linear regression, logistic regression, and neural networks.

## Evaluation

We implemented our framework based on [CirC](https://github.com/circify/circ/) and [Spartan](https://github.com/microsoft/Spartan). For ease of use, we included a Dockerfile with all necessary tools to reproduce the results from the paper. It can be build via

```
git clone https://github.com/cleverhans-lab/verifiable-unlearning verifiable-unlearning
cd verifiable-unlearning; ./docker.sh build
```

Beside building, the `docker.sh` script allows to spawn a shell in the container:

```
./docker.sh shell 
```

or run the evaluation:

```
./docker.sh eval 
```
