FROM ubuntu:22.04

RUN apt update &&\
    apt upgrade -y

RUN apt update &&\
    apt install -y curl python3-pip git tmux htop

# rust
WORKDIR /root
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > rustup.sh \
    && sh rustup.sh -y \
    && echo 'source "$HOME/.cargo/env"' >> ~/.bashrc \
    && /root/.cargo/bin/rustup default nightly

# Spartan
RUN    git clone https://github.com/microsoft/Spartan.git \
    && cd /root/Spartan \
    && git checkout 633a6cc16b4c766185991feaa75fad71b1a26358

ADD patches/lib.rs /root/Spartan/src/lib.rs

# CirC
RUN    git clone https://github.com/circify/circ.git \
    && cd circ \
    && git checkout 78c5d10fb2addb37e62a7b3486d23377c60dcc4b

RUN apt install -y cvc4 coinor-cbc coinor-libcbc-dev m4

WORKDIR /root/circ

ADD patches/Cargo.toml /root/circ/Cargo.toml
ADD patches/unlearning.rs /root/circ/examples/unlearning.rs
ADD patches/spartan.rs /root/circ/src/target/r1cs/spartan.rs
RUN /root/.cargo/bin/cargo build --release --example unlearning --features "zok,smt,r1cs,ristretto255" --no-default-features

WORKDIR /root
ADD . /root/verifiable-unlearning

WORKDIR /root/verifiable-unlearning
RUN pip install -r requirements.txt

RUN ln -s /root/circ/third_party /root/verifiable-unlearning/third_party