import argparse
import random
import shutil
import time
from pathlib import Path

import pydng

from circ import CirC
from classifier.linear_regression import LinearRegression
from classifier.logistic_regression import LogisticRegression
from classifier.neural_network import NeuralNetwork
from dataset import Dataset
from hashs.utils import *
from techniques.amnesiac import *
from techniques.optimization import *
from techniques.retraining import *
from utils import (format_running_time, parse_and_group_arguments, set_seeds,
                   setup_working_dir)


def main(trials_dir, trial_name, technique, mode, dataset_config, proof_config):

    print("[+] Arguments")
    print(f"    - {'trials_dir':<20}: {trials_dir}")
    print(f"    - {'trial_name':<20}: {trial_name}")
    print(f"    - {'trial_dir':<20}: {trials_dir/trial_name}")
    print(f"    - {'mode':<20}: {mode}")
    print(f"    - {'technique':<20}: {technique}")
    print(f'    - {"dataset_config":<25}')
    for name, value in dataset_config.items():
        print(f'      - {name:<25}: {value}')
    print(f'    - {"proof_config":<25}')
    for name, value in proof_config.items():
        print(f'      - {name:<25}: {value}')

    set_seeds(2022)
    working_dir = setup_working_dir(trials_dir, trial_name, overwrite=False)
    proof_config['working_dir'] = working_dir

    #
    # Init dataset
    #

    if 'synthetic' in dataset_config['dataset_name']:
        no_features = int(dataset_config['dataset_name'].split('_')[1])
        data_points = Dataset.make_classification(no_features=no_features)

        D_prev = Dataset([ next(data_points) for _ in range(dataset_config['no_samples_D_prev'])]).shift(proof_config['precision'])
        D_plus = Dataset([ next(data_points) for _ in range(dataset_config['no_samples_D_plus'])]).shift(proof_config['precision'])
        U_prev = Dataset([ next(data_points) for _ in range(dataset_config['no_samples_U_prev'])]).shift(proof_config['precision'])
        I = random.sample(range(dataset_config['no_samples_D_prev']), k=dataset_config['no_samples_U_plus'])
        U_plus = Dataset([D_prev[idx] for idx in I])

    else:
        D_prev = Dataset([]).shift(proof_config['precision'])
        U_prev = Dataset([]).shift(proof_config['precision'])
        D_plus = Dataset.from_pmlb(dataset_config['dataset_name']).shift(proof_config['precision'])
        print(f'[+] {D_plus}')

    #
    # Non-membership
    #

    if mode == 'non-membership':
        H_U, h_U = hash_dataset((U_prev+U_plus).data)
        # compute tree path
        tic = time.time()
        paths = []
        for d in U_plus.data:
            paths += [compute_tree_path(d, H_U)]
        running_time = time.time() - tic
        print(f"[+] Compute tree path")
        print(f"    {int(running_time*1000)} ms")

        # verify tree path
        tic = time.time()
        for d, path in zip(U_plus.data, paths):
            verify_tree_path(d, h_U, path)
        running_time = time.time() - tic
        print(f"[+] Verify tree path")
        print(f"    {int(running_time*1000/U_plus.size)} ms")

        return

    #
    # Init model
    #

    model_classes = {
        'linear_regression' : LinearRegression(),
        'logistic_regression' : LogisticRegression(),
        'neural_network_2' : NeuralNetwork(neurons=2),
        'neural_network_4' : NeuralNetwork(neurons=4)
    }
    model = model_classes[proof_config['classifier']]
    
    #
    # Init circuits
    #

    if technique == 'retraining' and mode == 'train':
        proof_src, params = circuit_train_retraining(proof_config, model, D_prev, U_prev, D_plus)

    elif technique == 'retraining' and mode == 'unlearn':
        proof_src, params = circuit_unlearn_retraining(proof_config, model, D_prev, U_prev, U_plus, I)

    elif technique == 'amnesiac' and mode == 'train':
        proof_src, params = circuit_train_amnesiac(proof_config, model, D_prev, U_prev, D_plus)

    elif technique == 'amnesiac' and mode == 'unlearn':
        proof_src, params = circuit_unlearn_amnesiac(proof_config, model, D_prev, U_prev, U_plus, I)

    elif technique == 'optimization' and mode == 'train':
        proof_src, params = circuit_train_optimization(proof_config, model, D_prev, U_prev, D_plus)

    elif technique == 'optimization' and mode == 'unlearn':
        proof_src, params = circuit_unlearn_optimization(proof_config, model, D_prev, U_prev, U_plus, I)

    else:
        raise ValueError()

    #
    # Prove and verify
    #

    working_dir.joinpath('circuit.zok').write_text(proof_src)

    circ = CirC(proof_config['circ_path'], debug=proof_config['debug'])
    shutil.copytree('/root/verifiable-unlearning/templates/poseidon', working_dir.joinpath('poseidon'))

    print(f'\n[+] CirC')
    if proof_config['proof_system'] == 'nizk':
        circ.spartan_nizk(params, working_dir)
    elif  proof_config['proof_system'] == 'snark':
        circ.spartan_snark(params, working_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials_dir', type=Path, default=Path('/root/verifiable-unlearning/evaluation/trials'))
    parser.add_argument('--trial_name', type=str, default=f'unsorted/{time.strftime("%Y-%m-%d")}_{pydng.generate_name()}', help='')
    
    parser.add_argument('--technique', type=str, default='optimization')
    parser.add_argument('--mode', type=str, default='train')

    dataset_config = parser.add_argument_group('dataset_config')
    dataset_config.add_argument('--dataset_dir', type=Path, default=Path.home().joinpath('verifiable-unlearning/evaluation/data'))
    dataset_config.add_argument('--dataset_name', type=str, default='synthetic_1')

    dataset_config.add_argument('--no_samples_D_prev', type=int, default=0)
    dataset_config.add_argument('--no_samples_D_plus', type=int, default=1)
    dataset_config.add_argument('--no_samples_U_prev', type=int, default=0)
    dataset_config.add_argument('--no_samples_U_plus', type=int, default=0)

    proof_config = parser.add_argument_group('proof_config')
    proof_config.add_argument('--circ_path', type=Path, default=Path('/root/circ'))
    proof_config.add_argument('--proof_system', type=str, default="nizk")
    proof_config.add_argument('--circuit_dir', type=Path, default=Path('/root/verifiable-unlearning/templates'))
    proof_config.add_argument('--epochs', type=int, default=3)
    proof_config.add_argument('--lr', type=float, default=0.01)
    proof_config.add_argument('--classifier', type=str, default='linear_regression')
    proof_config.add_argument('--precision', type=int, default=1e5)
    proof_config.add_argument('--debug', action="store_false")
    proof_config.add_argument('--model_seed', type=int, default=2023)

    proof_config.add_argument('--unlearning_epochs', type=int, default=3)
    proof_config.add_argument('--unlearining_lr', type=float, default=0.01)

    args = parse_and_group_arguments(parser)
    main(**args)
