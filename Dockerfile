FROM ubuntu:22.04

RUN apt update &&\
    apt upgrade -y

RUN apt update &&\
    apt install -y curl python3-pip git tmux htop

RUN curl -LSfs get.zokrat.es | sh &&\
    echo 'export PATH=$PATH:/root/.zokrates/bin' >> ~/.bashrc

WORKDIR /root
RUN git clone https://github.com/cleverhans-lab/verifiable-unlearning

WORKDIR /root/verifiable-unlearning
RUN pip install -r requirements.txt

RUN git clone https://github.com/Zokrates/pycrypto.git &&\
    cd pycrypto &&\
    pip install -r requirements.txt

# only needed for plots
ENV DEBIAN_FRONTEND='noninteractive'
RUN    apt update \
    && apt install -y texlive-full