import sys
from pathlib import Path
sys.path.append(Path.home().joinpath('verifiable-unlearning/pycrypto').as_posix())  
import argparse
import json
import time

import pydng
from tqdm import tqdm

from dataset import Dataset
from snarks import *
from utils import parse_and_group_arguments, set_seeds, setup_working_dir

def hash_data_point(x):
    h = hash_int(x[0])
    for x_i in x[1:]:
        lhs = h
        rhs = bytes.fromhex(("0"*64 + hex(x_i)[2:])[-64:])
        h = hash_hex(lhs+rhs)
    return h

def hash_unlearnt(H_U):
    tree = [hash_int(0)]
    for leaf_idx in range(len(H_U)):
        tree += [ H_U[leaf_idx] ]
        tree += [ hash_hex(tree[-2]+tree[-1]) ]
    return tree[-1]

def compute_tree_path(d, H_U):
    h_d = hash_data_point(d)
    idx = H_U.index(h_d)
    psi = hash_unlearnt(H_U[:idx])
    path = [psi]
    for h in H_U[idx+1:]:
        path += [h]
    return path

def verify_tree_path(d, h_U, path):
    h_d = hash_data_point(d)
    psi = hash_hex(path[0]+h_d)
    for node in path[1:]:
        psi = hash_hex(psi+node)
    return psi == h_U

def main(trials_dir, trial_name, no_samples_D, no_samples_U_prev, no_samples_U_plus, proof_config):

    print("[+] Arguments")
    print(f"    - {'no_samples_D':<20}: {no_samples_D}")
    print(f"    - {'no_samples_U_prev':<20}: {no_samples_U_prev}")
    print(f"    - {'no_samples_U_plus':<20}: {no_samples_U_plus}")
    print(f"    - {'trials_dir':<20}: {trials_dir}")
    print(f"    - {'trial_name':<20}: {trial_name}")
    print(f"    - {'trial_dir':<20}: {trials_dir/trial_name}")
    print(f'    - {"proof_config":<25}')
    for name, value in proof_config.items():
        print(f'      - {name:<25}: {value}')

    set_seeds(2022)
    working_dir = setup_working_dir(trials_dir, trial_name)

    dataset = Dataset.make_classification(no_samples=no_samples_D+no_samples_U_prev+no_samples_U_plus, no_features=1)
    dataset = dataset.shift(proof_config['precision'])

    X = [ [ twos_complement(x_i) for x_i in x ] for x in dataset.X ]
    Y = [ twos_complement(y_i) for y_i in dataset.Y ]

    U_prev = [ [ Y[idx] ] + X[idx] for idx in range(no_samples_D, no_samples_D+no_samples_U_prev) ]
    U_plus = [ [ Y[idx] ] + X[idx] for idx in range(no_samples_D+no_samples_U_prev, no_samples_D+no_samples_U_prev+no_samples_U_plus) ]

    stats = {}
    U = U_prev + U_plus
    H_U = [ hash_data_point(leaf) for leaf in tqdm(U, ncols=80) ]
    h_U = hash_unlearnt(H_U)
    d = U[-1]

    # prove unlearn
    tic = time.time()
    path = compute_tree_path(d, H_U)
    running_time = time.time() - tic
    print(f"[+] Compute tree path")
    print(f"    {format_running_time(running_time)}")
    stats['compute_tree_path'] = running_time        
    
    # verifiy unlearn
    tic = time.time()
    verified = verify_tree_path(d, h_U, path)
    running_time = time.time() - tic
    print(f"[+] Verify tree path")
    print(f"    {format_running_time(running_time)}")
    print(f"    {verified}")
    stats['verify_tree_path'] = running_time        
    
    working_dir.joinpath('stats.json').write_text(json.dumps(stats, indent=4))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--trials_dir', type=Path, default=Path.home().joinpath('verifiable-unlearning/evaluation/trials'))
    parser.add_argument('--trial_name', type=str, default=f'unsorted/{time.strftime("%Y-%m-%d")}_{pydng.generate_name()}', help='')

    parser.add_argument('--no_samples_D', type=int, default=1)
    parser.add_argument('--no_samples_U_prev', type=int, default=1)
    parser.add_argument('--no_samples_U_plus', type=int, default=1)

    proof_config_parser = parser.add_argument_group('proof_config')
    proof_config_parser.add_argument('--precision', type=int, default=1e5)

    args = parser.parse_args()
    main(**parse_and_group_arguments(parser))

