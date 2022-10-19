
import sys
from pathlib import Path
sys.path.append(Path.home().joinpath('verifiable-unlearning/pycrypto').as_posix())  
import argparse
import binascii
import json
import math
import sys
import time
from pathlib import Path

import pydng
from jinja2 import Template

from dataset import Dataset
from snarks import *
from utils import parse_and_group_arguments, set_seeds, setup_working_dir


def build_merkle_tree(leafs, var_names=('tree_D', 'H_D', 'h_D'), append_only=False):
    
    if not append_only:
        H_leafs_zip = [ (hash_input(leaf), leaf) for leaf in leafs ]
        H_leafs_zip = sorted(H_leafs_zip, key=lambda d: int(binascii.hexlify(d[0]), 16))
        H_leafs, leafs = zip(*H_leafs_zip)
        print()
        tree = []
        node_idx = 0
        for leaf_idx in range(len(leafs)):
            tree += [ H_leafs[leaf_idx] ]
            node_idx += 1

        previous_level = list(range(len(leafs)))
        while len(previous_level) != 1:
            current_level = []
            for idx in range(0, len(previous_level)-1, 2):
                tree += [hash_hex(tree[previous_level[idx]]+tree[previous_level[idx+1]])]
                current_level += [node_idx]
                node_idx += 1
            if len(previous_level) % 2 == 1:
                current_level.append(previous_level[-1])
            previous_level = current_level

        no_samples = len(leafs)

        verification_proof_src = ""
        verification_proof_src += f'u32[{no_samples*2-1}][8] mut tree = [[0; 8]; {no_samples*2-1}];\n\n'

        node_idx = 0
        for leaf_idx in range(no_samples):
            verification_proof_src += f'tree[{node_idx}] = H_D[{leaf_idx}];\n'
            node_idx += 1
        verification_proof_src += "\n"

        previous_level = list(range(no_samples))
        while len(previous_level) != 1:
            current_level = []
            for idx in range(0, len(previous_level)-1, 2):
                verification_proof_src += f'tree[{node_idx}] = hash_digest(tree[{previous_level[idx]}], tree[{previous_level[idx+1]}]);\n'
                current_level += [node_idx]
                node_idx += 1
            if len(previous_level) % 2 == 1:
                current_level.append(previous_level[-1])
            verification_proof_src += "\n"
            previous_level = current_level
        verification_proof_src += f'assert(is_equal(tree[{node_idx-1}], h_D));'
    
    else:
        H_leafs = []
        tree = [hash_int(0)]
        for leaf_idx in range(len(leafs)):
            h = hash_input(leafs[leaf_idx])
            H_leafs += [ h ]
            tree += [ h ]
            tree += [ hash_hex(tree[-2]+tree[-1]) ]

        verification_proof_src = ""
        verification_proof_src += f'u32[{len(tree)}][8] mut tree = [[0; 8]; {len(tree)}];\n\n'
        verification_proof_src += f'tree[0] = hash_int(0);\n'
        node_idx = 1
        for leaf_idx in range(len(leafs)):
            verification_proof_src += f'tree[{node_idx}] = H_D[{leaf_idx}];\n'
            node_idx += 1
            verification_proof_src += f'tree[{node_idx}] = hash_digest(tree[{node_idx-2}], tree[{node_idx-1}]);\n'
            node_idx += 1
        verification_proof_src += f'assert(is_equal(tree[{node_idx-1}], h_D));'

    verification_proof_src = verification_proof_src.replace('tree', var_names[0])
    verification_proof_src = verification_proof_src.replace('H_D', var_names[1])
    verification_proof_src = verification_proof_src.replace('h_D', var_names[2])
    verification_proof_src = verification_proof_src.replace('\n\n', '\n')

    return H_leafs, tree[-1], verification_proof_src

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


    H_D, h_D, h_D_circuit = build_merkle_tree(D, ('tree_D', 'H_D', 'h_D'), append_only=False)
    H_U_prev, h_U_prev, h_U_prev_circuit = build_merkle_tree(U_prev, ('tree_U_prev', 'H_U_prev', 'h_U_prev'), append_only=True)
    _, h_U, _ = build_merkle_tree(U_prev+U_plus, append_only=True)
    H_U_plus = [ hash_input(d) for d in U_plus ]

    template = Template(proof_config['proof_template'].read_text())
    proof_zk = template.render(max_depth_D=math.ceil(math.log2(len(D))),
                               no_samples_D=len(D),
                               no_samples_U_prev=len(U_prev),
                               no_samples_D_plus=len(U_plus),
                               h_D_circuit=h_D_circuit,
                               h_U_prev_circuit=h_U_prev_circuit)

    witness_args = " ".join([
        # D
        " ".join([str(int(c, 16)) for c in to_u32(h_D)]),
        " ".join([" ".join([str(int(c, 16)) for c in to_u32(h)]) for h in H_D]),
        # U
        " ".join([str(int(c, 16)) for c in to_u32(h_U_prev)]),
        " ".join([" ".join([str(int(c, 16)) for c in to_u32(h)]) for h in H_U_prev]),
        # U plus
        " ".join([str(int(c, 16)) for c in to_u32(h_U)]),
        " ".join([" ".join([str(int(c, 16)) for c in to_u32(h)]) for h in H_U_plus]),
    ])
    
    zokrates = ZoKrates(debug=proof_config["debug"], backend=proof_config["backend"])
    stats = {}

    print(f'\n[+] Setup')
    tic = time.time()
    no_constraints, verification_key, proving_key, circuit = zokrates.setup(working_dir.joinpath('setup'), proof_zk)
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

    parser.add_argument('--no_samples_D', type=int, default=1)
    parser.add_argument('--no_samples_U_prev', type=int, default=1)
    parser.add_argument('--no_samples_U_plus', type=int, default=1)

    proof_config_parser = parser.add_argument_group('proof_config')
    proof_config_parser.add_argument('--proof_template', type=Path, default=Path.home().joinpath("verifiable-unlearning/templates/dataset.zk.template"))
    proof_config_parser.add_argument('--precision', type=int, default=1e5)
    proof_config_parser.add_argument('--backend', type=str, default='ark')
    proof_config_parser.add_argument('--debug', action="store_true")

    args = parser.parse_args()
    main(**parse_and_group_arguments(parser))
