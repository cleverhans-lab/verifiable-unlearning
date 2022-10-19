import sys
from pathlib import Path
sys.path.append(Path.home().joinpath('verifiable-unlearning/pycrypto').as_posix())  
import binascii
import re
import struct
import sys
from pathlib import Path
from subprocess import run

from jinja2 import Template
from zokrates_pycrypto.gadgets.pedersenHasher import PedersenHasher


def twos_complement(x):
    return int(binascii.hexlify(struct.pack(">q", int(x))), 16)

def hash_hex(preimage):
    hasher = PedersenHasher(b"test")
    digest = hasher.hash_bytes(preimage)
    x, y = int(digest.x), int(digest.y)
    digest_hex = int.to_bytes(y | ((x & 1) << 255), 32, "big")
    return digest_hex

def hash_int(x):
    return hash_hex(bytes.fromhex(("0"*128 + hex(x)[2:])[-128:]))

def hash_input(x):
    h = hash_int(x[0])
    for x_i in x[1:]:
        lhs = h
        rhs = bytes.fromhex(("0"*64 + hex(x_i)[2:])[-64:])
        h = hash_hex(lhs+rhs)
    return h

def to_u32(x, to_str=False, to_z=False):
    x = binascii.hexlify(x)
    chunks = [] 
    for idx in range(0, 64, 8):
        chunks += [ x[idx:idx+8] ]
    if to_str:
        return "[" + ", ".join([c.decode() for c in chunks]) + "]"
    if to_z:
        return "[" + ", ".join([str(int(c, 16)) for c in chunks]) + "]"
    return chunks

def format_running_time(running_time):
    return f"{running_time // 3600:.0f}h {(running_time % 3600) // 60:.0f}m {(running_time % 60):.0f}s"

def build_merkle_tree(leafs, append_only=False):

    if not append_only:
        tree = []
        node_idx = 0
        for leaf_idx in range(len(leafs)):
            tree += [hash_input(leafs[leaf_idx])]
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
    
    else:
        tree = [hash_int(0)]
        for leaf_idx in range(len(leafs)):
            tree += [ hash_int(int(leafs[leaf_idx])) ]
            tree += [ hash_hex(tree[-2]+tree[-1]) ]

    return tree[-1]

def create_circuit(no_samples, no_features, no_weights, weights_init_str, proof_config, train_config):

    if proof_config['skip_verification']:
        verification_proof_src = None
    else:
        verification_proof_src = ""
        verification_proof_src += f'u32[{no_samples*2-1}][8] mut tree = [[0; 8]; {no_samples*2-1}];\n\n'

        node_idx = 0
        for leaf_idx in range(no_samples):
            verification_proof_src += f'tree[{node_idx}] = hash_input(X[{leaf_idx}], Y[{leaf_idx}]);\n'
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

        verification_proof_src += f'assert(is_equal(tree[{node_idx-1}],verification_accumulator));'

    if train_config['neural_network_2'] is True:
        no_neurons = 2
    elif train_config['neural_network_4'] is True:
        no_neurons = 4
    else:
        no_neurons = 0

    template = Template(proof_config['proof_template'].read_text())
    proof_zk = template.render(no_samples=no_samples, 
                                precision=f'{train_config["precision"]:.0f}',
                                epochs=f'{train_config["epochs"]:.0f}', 
                                no_features=f'{no_features:.0f}', 
                                no_weights=f'{no_weights:.0f}', 
                                lr=f'{train_config["precision"]*train_config["lr"]:.0f}', 
                                tree_generation=verification_proof_src, 
                                skip_verification=proof_config['skip_verification'], 
                                skip_regression=proof_config['skip_regression'],
                                linear_regression=train_config['linear_regression'],
                                logistic_regression=train_config['logistic_regression'],
                                neural_network=no_neurons>0,
                                no_neurons=no_neurons,
                                W0=f'{int(0.5*train_config["precision"]):.0f}',
                                W1=f'{int(0.1501*train_config["precision"]):.0f}',
                                W3=f'{int(0.0016*train_config["precision"]):.0f}',
                                weights_init_str=weights_init_str)

    return proof_zk

class ZoKrates:

    def __init__(self, debug=False, backend="ark"):
        self.debug = debug
        self.backend = backend

    def run(self, cmd, cwd):
        p = run(cmd, cwd=cwd, shell=True, capture_output=True)
        if p.returncode != 0:
            print(p.stdout.decode())
            raise RuntimeError()
        if self.debug:
            print(p.stdout.decode())
        return p.stdout.decode()

    def setup(self, working_dir, proof_src):
        working_dir.mkdir(parents=True)
        working_dir.joinpath('proof.zk').write_text(proof_src)
        stdout = self.run(f'zokrates compile {"--debug" if self.debug else ""} -i proof.zk', working_dir)
        no_constraints = int(re.findall(r'Number of constraints: (\d+)', stdout).pop())
        self.run(f'zokrates setup --backend {self.backend}', working_dir)
        return (no_constraints,
                working_dir.joinpath('verification.key'),
                working_dir.joinpath('proving.key'),
                working_dir.joinpath('out'))

    def create_proof(self, working_dir, circuit, witness_args, proving_key):
        working_dir.mkdir(parents=True)
        self.run(f'zokrates compute-witness --input {circuit} -a ' + witness_args, working_dir)
        self.run(f'zokrates generate-proof --proving-key-path {proving_key} --input {circuit} --backend {self.backend}', working_dir)
        return working_dir.joinpath('proof.json')

    def verify_proof(self, working_dir, verification_key, proof):
        working_dir.mkdir(parents=True)
        stdout = self.run(f'zokrates verify --proof-path {proof} --verification-key-path {verification_key} --backend {self.backend}', working_dir)
        if "PASSED" not in stdout:
            print(stdout)
