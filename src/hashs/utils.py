import binascii
import struct

from .poseidon import poseidon_hash_2


def compute_tree_path(d, H_U):
    h_d = hash_input(d)
    idx = H_U.index(h_d)
   
    psi = poseidon_hash_2(0, 0)
    for h in H_U:
        psi = poseidon_hash_2(psi, h)

    path = [psi]
    for h in H_U[idx+1:]:
        path += [h]
    return path

def verify_tree_path(d, h_U, path):
    h_d = hash_input(d)
    psi =  poseidon_hash_2(path[0], h_d)
    for node in path[1:]:
        psi = poseidon_hash_2(psi, node)
    return psi == h_U

def hash_dataset(data):
    H = []
    h = poseidon_hash_2(0, 0)
    for d in data:
        H += [ hash_input(d) ]
        h = poseidon_hash_2(h, hash_input(d))
    return H, h

# def hash_deltas(deltas):
#     H = []
#     h = poseidon_hash_2(0, 0)
#     for delta in deltas:
#         H += [ hash_input(delta) ]
#         h = poseidon_hash_2(h, hash_input(delta))
#     return H, h

# def hash_delta(delta):
#     h = poseidon_hash_2(twos_complement(delta[0]), twos_complement(delta[1]))
#     for delta_i in delta[2:]:
#         h = poseidon_hash_2(h, twos_complement(delta_i))
#     return h

# def hash_deltas(deltas):
#     H = []
#     h = poseidon_hash_2(0, 0)
#     for d in deltas:
#         H += [ d ]
#         h = poseidon_hash_2(h, hash_input(d))
#     return H, h

def hash_list(inputs, pad=False):
    assert len(inputs) > 1, inputs
    h = poseidon_hash_2(twos_complement(inputs[0]), twos_complement(inputs[1]))
    for input in inputs[2:]:
        h = poseidon_hash_2(h, twos_complement(input))
    return h

def twos_complement(x):
    return int(binascii.hexlify(struct.pack(">q", int(x))), 16)

def hash_input(x):
    h = poseidon_hash_2(twos_complement(x[0]), twos_complement(x[1]))
    for x_i in x[2:]:
        h = poseidon_hash_2(h, twos_complement(x_i))
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

