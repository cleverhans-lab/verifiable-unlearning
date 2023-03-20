#!/bin/bash

build()
{
    docker build -t verifiable-unlearning .
}

shell() 
{
    docker run -it verifiable-unlearning
}

evaluation() 
{
    docker run -it verifiable-unlearning bash /root/verifiable-unlearning/evaluation/trials.sh
}

print_usage() 
{
    echo "Choose: docker.sh {build|shell|eval}"
    echo "    build - Build the container"
    echo "    shell - Spawn a shell inside the container"
    echo "    eval  - Run evaluation in the container"
}

if [[ $1 == "" ]]; then
    echo "No argument provided"
    print_usage
elif [[ $1 == "build" ]]; then
    build
elif [[ $1 == "shell" ]]; then
    shell 
elif [[ $1 == "eval" ]]; then
    evaluation
else 
    echo "Argument not recognized!"
    print_usage
fi