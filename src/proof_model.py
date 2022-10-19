import sys
from pathlib import Path
sys.path.append(Path.home().joinpath('verifiable-unlearning/pycrypto').as_posix())  

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pydng

from classifier.linear_regression import LinearRegression
from classifier.logistic_regression import LogisticRegression
from classifier.neural_network import NeuralNetwork
from dataset import Dataset
from snarks import *
from utils import parse_and_group_arguments, set_seeds, setup_working_dir

def main(dataset_dir, dataset_name, trials_dir, trial_name, proof_config, train_config):

    print("[+] Arguments")
    print(f"    - {'dataset_dir':<20}: {dataset_dir}")
    print(f"    - {'dataset_name':<20}: {dataset_name}")
    print(f"    - {'trials_dir':<20}: {trials_dir}")
    print(f"    - {'trial_name':<20}: {trial_name}")
    print(f"    - {'trial_dir':<20}: {trials_dir/trial_name}")
    print(f'    - {"train_config":<25}')
    for name, value in train_config.items():
        print(f'      - {name:<25}: {value}')
    print(f'    - {"proof_config":<25}')
    for name, value in proof_config.items():
        print(f'      - {name:<25}: {value}')

    set_seeds(2022)
    working_dir = setup_working_dir(trials_dir, trial_name)

    if "synthetic" in dataset_name:
        no_samples = int(dataset_name.split('_')[1])
        no_features = 1
        dataset = Dataset.make_classification(no_samples, no_features)

    else:
        dataset = Dataset.from_pmlb(dataset_name)

    dataset = dataset.shift(train_config['precision'])

    # model setup
    if train_config['linear_regression'] is True:
        print('linear')
        model = LinearRegression()
    elif train_config['logistic_regression'] is True:
        print('logistic')
        model = LogisticRegression()
    elif train_config['neural_network_2'] is True:
        print('neural_network_2')
        model = NeuralNetwork(neurons=2)
    elif train_config['neural_network_4'] is True:
        print('neural_network_4')
        model = NeuralNetwork(neurons=4)
    else:
        raise ValueError("Need to sleect a classifier")

    # train model
    weights_init = (np.random.rand(model.no_weights(dataset, bias=False))-0.5).tolist()
    weights_init_str = model.format_weights_init(train_config, dataset, weights_init)
    print(weights_init_str)

    model.train(train_config, dataset, weights_init)
    acc = model.score(dataset, train_config)
    working_dir.joinpath('model.json').write_text(json.dumps({'acc': acc}, indent=4))

    # create proof source
    proof_src = create_circuit(no_samples=len(dataset), 
                                no_features=dataset.no_features, 
                                no_weights=len(model.weights), 
                                weights_init_str=weights_init_str,
                                proof_config=proof_config, 
                                train_config=train_config)

    # witness
    X = [ [ twos_complement(x_i) for x_i in x ] for x in dataset.X ]
    Y = [ twos_complement(y_i) for y_i in dataset.Y ]
    weights = [ twos_complement(w) for w in model.weights]

    h_m = hash_int(weights[0])
    for w_i in weights[1:]:
        h_m = hash_hex(h_m+hash_int(w_i))

    if proof_config['skip_verification']:
        accumulator = b'\x00'*64 
    else:
        leafs = [ [ Y[idx] ] + X[idx] for idx in range(len(X)) ]
        accumulator = build_merkle_tree(leafs)

    witness_args = " ".join([
        " ".join([str(int(c, 16)) for c in to_u32(accumulator)]),
        " ".join([str(int(c, 16)) for c in to_u32(h_m)]),
        " ".join([ " ".join(map(str, x)) for x in X ]),
        " ".join(map(str, Y)),
    ])

    # run zokrates
    zokrates = ZoKrates(debug=proof_config["debug"], backend=proof_config["backend"])
    stats = {}

    print(f'\n[+] Setup')
    tic = time.time()
    no_constraints, verification_key, proving_key, circuit = zokrates.setup(working_dir.joinpath('setup'), proof_src)
    running_time = time.time() - tic
    stats["setup"] = running_time
    stats["no_constraints"] = no_constraints
    print(f'    took {format_running_time(running_time)}')

    print(f'\n[+] Prove')
    tic = time.time()
    proof = zokrates.create_proof(working_dir.joinpath('proof'), circuit, witness_args, proving_key)
    running_time = time.time() - tic
    stats["prove"] = running_time
    print(f'    took {format_running_time(running_time)}')

    print(f'\n[+] Verify')
    tic = time.time()
    zokrates.verify_proof(working_dir.joinpath('verify'), verification_key, proof)
    running_time = time.time() - tic
    stats["verify"] = running_time
    print(f'    took {format_running_time(running_time)}')

    working_dir.joinpath('stats.json').write_text(json.dumps(stats, indent=4))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--trials_dir', type=Path, default=Path.home().joinpath('verifiable-unlearning/evaluation/trials'))
    parser.add_argument('--trial_name', type=str, default=f'unsorted/{time.strftime("%Y-%m-%d")}_{pydng.generate_name()}', help='')

    parser.add_argument('--dataset_name', type=str, default='synthetic_1')
    parser.add_argument('--dataset_dir', type=Path, default=Path.home().joinpath('verifiable-unlearning/evaluation/data'))

    train_config_parser = parser.add_argument_group('train_config')
    train_config_parser.add_argument('--epochs', type=int, default=10)
    train_config_parser.add_argument('--lr', type=float, default=0.01)
    train_config_parser.add_argument('--linear_regression', action="store_true")
    train_config_parser.add_argument('--logistic_regression', action="store_true")
    train_config_parser.add_argument('--neural_network_2', action="store_true")
    train_config_parser.add_argument('--neural_network_4', action="store_true")
    train_config_parser.add_argument('--precision', type=int, default=1e5)
    
    proof_config_parser = parser.add_argument_group('proof_config')
    proof_config_parser.add_argument('--proof_template', type=Path, default=Path.home().joinpath("verifiable-unlearning/templates/model.zk.template"))
    proof_config_parser.add_argument('--skip_verification', action="store_true")
    proof_config_parser.add_argument('--skip_regression', action="store_true")
    proof_config_parser.add_argument('--backend', type=str, default='ark')
    proof_config_parser.add_argument('--debug', action="store_true")

    args = parser.parse_args()
    main(**parse_and_group_arguments(parser))
