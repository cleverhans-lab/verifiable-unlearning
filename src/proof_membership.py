import sys
from pathlib import Path
sys.path.append(Path.home().joinpath('verifiable-unlearning/pycrypto').as_posix())  
import argparse
import binascii
import json
import struct
import time

import numpy as np
import pydng
from tqdm import tqdm
from zokrates_pycrypto.gadgets.pedersenHasher import PedersenHasher

from classifier.linear_regression import LinearRegression
from dataset import Dataset
from snarks import *
from utils import parse_and_group_arguments, set_seeds, setup_working_dir

def twos_complement(x):
    return int(binascii.hexlify(struct.pack(">q", int(x))), 16)

def hash_hex(preimage):
    # preimage = bytes.fromhex(x)
    hasher = PedersenHasher(b"test")
    digest = hasher.hash_bytes(preimage)
    x, y = int(digest.x), int(digest.y)
    digest_hex = int.to_bytes(y | ((x & 1) << 255), 32, "big")
    return digest_hex

def hash_int(x):
    return hash_hex(bytes.fromhex(("0"*128 + hex(x)[2:])[-128:]))

def hash_data_point(x):
    h = hash_int(x[0])
    for x_i in x[1:]:
        lhs = h
        # rhs = hash_int(x_i)
        rhs = bytes.fromhex(("0"*64 + hex(x_i)[2:])[-64:])
        h = hash_hex(lhs+rhs)
    return h

def format_running_time(running_time):
    return f"{running_time // 3600:.0f}h {(running_time % 3600) // 60:.0f}m {(running_time % 60):.0f}s"

######

def hash_data(H_D):
    no_hashs = 0
    tree = []
    node_idx = 0
    for leaf_idx in range(len(H_D)):
        tree += [H_D[leaf_idx]]
        node_idx += 1
    previous_level = list(range(len(H_D)))
    while len(previous_level) != 1:
        current_level = []
        for idx in range(0, len(previous_level)-1, 2):
            tree += [hash_hex(tree[previous_level[idx]]+tree[previous_level[idx+1]])]
            no_hashs += 1
            current_level += [node_idx]
            node_idx += 1
        if len(previous_level) % 2 == 1:
            current_level.append(previous_level[-1])
        previous_level = current_level
    # print('no hashs', no_hashs)
    return tree[-1]

def hash_unlearnt(H_U):
    tree = [hash_int(0)]
    no_hashs = 0
    for leaf_idx in range(len(H_U)):
        tree += [ H_U[leaf_idx] ]
        tree += [ hash_hex(tree[-2]+tree[-1]) ]
    return tree[-1]

def verify_dataset_update(H_U, H_U_prev, H_D):
    verified = True
    # if not set(H_U_prev).issubset(set(H_U)):
    #     verified = False
    if len(set.intersection(set(H_U), set(H_D))) > 0:
        verified = False
    return verified

def prove_update(state_server, D_plus, U_plus):
    D, H_D, H_U, = state_server
    # update dataset
    D = D.update(D_plus, U_plus)
    # hash updates
    H_D_plus = [ hash_data_point(d) for d in D_plus ]
    H_U_plus = [ hash_data_point(d) for d in U_plus ]
    # update hashed dataset
    H_D = [ h_d for h_d in H_D + H_D_plus 
                 if h_d not in H_U_plus    ] 
    H_U = H_U[:] + H_U_plus
    # retrain model
    m = train_model(D)
    # commitment
    h_m = hash_model(m)
    h_D = hash_data(H_D)
    h_U = hash_unlearnt(H_U)
    com = (h_m, h_D, h_U)
    # compute proof
    # 
    p = None
    up = H_D_plus, H_U_plus
    state_server = (D, H_D, H_U)
    return (state_server, com, m, p, up)

def verify_update(state_auditor, com, m, p, up):
    verified = True
    # parse params
    H_D_prime, H_U_prime = state_auditor
    H_D_plus, H_U_plus = up
    # update hashed dataset
    H_D_prime = [ h_d for h_d in H_D_prime + H_D_plus 
                 if h_d not in H_U_plus    ]
    H_U_prime_prev = H_U_prime[:]
    H_U_prime = H_U_prime[:] + H_U_plus
    # commitment
    h_m_prime = hash_model(m)
    h_D_prime = hash_data(H_D)
    h_U_prime = hash_unlearnt(H_U_prime)
    if com[0] != h_m_prime or com[1] != h_D_prime or com[2] != h_U_prime:
        verified = False
    # check model
    pass
    # dataset
    if not verify_dataset_update(H_U_prime, H_U_prime_prev, H_D_prime):
        verified = False

    state_auditor = (H_D_prime, H_U_prime)
    return state_auditor, verified

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

def prove_unlearn(state_server, d):
    _, _, H_U, = state_server
    return compute_tree_path(d, H_U)

def verify_unlearn(d, com, path):
    _, _, h_U = com
    return verify_tree_path(d, h_U, path)

# def vrfy_dataset(h_D, H_D_minus, H_D, H_D_all, H_D_minus_prev):

#     verify_root = (h_D != accumulate_data(H_D))
#     verify_intersection = len(set.intersection(set(H_D), set(H_D_minus))) == 0
#     verify_union = set.union(set(H_D), set(H_D_minus)) == set(H_D_all)
#     verify_subset = set(H_D_minus_prev).is


#     if h_D != accumulate_data(H_D):
#         return False

#     if set.intersection(set(H_D), set(H_D_minus))

#     return psi == H_D_minus


def train_model(dataset):
        
    train_config = {
        'epochs' : 1,
        'batch_size' : 1,
        'lr' : 0.01,
        'precision' : 1e5,
    }

    model = LinearRegression()

    weights_init = (np.random.rand(model.no_weights(dataset, bias=False))-0.5).tolist()
    # weights_init_str = model.format_weights_init(train_config, dataset, weights_init)
    # print(weights_init_str)

    model.train(train_config, dataset, weights_init)
    return model

def hash_model(model):
    weights = [ twos_complement(w_i) for w_i in model.weights ]
    h = hash_int(weights[0])
    for w_i in weights[1:]:
        lhs = h
        rhs = bytes.fromhex(("0"*64 + hex(w_i)[2:])[-64:])
        h = hash_hex(lhs+rhs)
    return h
    

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

    D =      [ [ Y[idx] ] + X[idx] for idx in range(0, no_samples_D) ]
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

