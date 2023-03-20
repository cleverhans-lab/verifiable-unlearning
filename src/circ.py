
from subprocess import run
import shutil
import re
import binascii
import struct
import json
from pathlib import Path

def to_hex_str(x, no_bytes):
    return binascii.hexlify(bytes.fromhex(("0"*no_bytes+ hex(x)[2:])[-no_bytes:])).decode()

def twos_complement(x):
    return int(binascii.hexlify(struct.pack(">q", int(x))), 16)

class CirC:

    def __init__(self, circ_path, debug=False):
        self.debug = debug
        self.circ_path = circ_path

    def run(self, cmd, cwd):
        if self.debug:
            print(f'[+] {cmd}')
            import os
            os.environ['RUST_BACKTRACE'] = "1"
        p = run(cmd, cwd=cwd, shell=True, capture_output=True)
        if p.returncode != 0:
            print(p.stdout.decode())
            print(p.stderr.decode())
            raise RuntimeError()
        if self.debug:
            print(p.stdout.decode())
            print(p.stderr.decode())
        return p.stdout.decode()

    def spartan_nizk(self, params, working_dir):
        working_dir.mkdir(exist_ok=True, parents=True)
        self._make_params(params, working_dir)
        cmd = f'{self.circ_path.joinpath("target/release/examples/unlearning")} circuit.zok nizk --pin circuit.pin --vin circuit.vin'
        stdout = self.run(cmd, working_dir)
        working_dir.joinpath('circ.log.txt').write_text(stdout)

    def spartan_snark(self, params, working_dir):
        working_dir.mkdir(exist_ok=True, parents=True)
        self._make_params(params, working_dir)
        cmd = f'{self.circ_path.joinpath("target/release/examples/unlearning")} circuit.zok snark --pin circuit.pin --vin circuit.vin'
        stdout = self.run(cmd, working_dir)
        working_dir.joinpath('circ.log.txt').write_text(stdout)
        # results = {
        #     'setup' : int(re.findall(r'\[\+\] Setup total (\d+) ms', stdout).pop()),
        #     'prove' : int(re.findall(r'\[\+\] Prove\n    took (\d+) ms', stdout).pop()),
        #     'verify' : int(re.findall(r'\[\+\] Verify\n    took (\d+) ms', stdout).pop()),
        #     'r1cs' : int(re.findall(r'- final R1CS size: (\d+)', stdout).pop())
        # }
        # working_dir.joinpath('circ.json').write_text(json.dumps(results, indent=4))

    # def zk(self, working_dir, params):
    #     working_dir.mkdir(exist_ok=True, parents=True)
    #     self._make_params(params, working_dir)
    #     cmd = f'{self.circ_path.joinpath("target/release/examples/zk")} --pin circuit.pin --vin circuit.vin --action spartan'
    #     self.run(cmd, working_dir)

    # def circ(self, working_dir):
    #     working_dir.mkdir(exist_ok=True, parents=True)
    #     cmd = f'{self.circ_path.joinpath("target/release/examples/circ")} circuit.zok  r1cs --action spartansetup'
    #     self.run(cmd, working_dir)

    # def zk(self, working_dir, params):
    #     working_dir.mkdir(exist_ok=True, parents=True)
    #     self._make_params(params, working_dir)
    #     cmd = f'{self.circ_path.joinpath("target/release/examples/zk")} --pin circuit.pin --vin circuit.vin --action spartan'
    #     self.run(cmd, working_dir)

    def _make_params(self, params, working_dir):
        proving_params = [ self._make_param(param) for param in params ]
        working_dir.joinpath('circuit.pin').write_text(
            "(set_default_modulus 7237005577332262213973186563042994240857116359379907606001950938285454250989\n" \
            "(let (\n" \
            + "".join(proving_params) + \
            ") true ;ignored\n" \
            ")\n" \
            ")"
        )
        verification_params = [ self._make_param(param) for param in params if param[0] == "public" ]
        working_dir.joinpath('circuit.vin').write_text(
            "(set_default_modulus 7237005577332262213973186563042994240857116359379907606001950938285454250989\n" \
            "(let (\n" \
            + "".join(verification_params) + \
            "\t(return #f1)\n" \
            ") true ;ignored\n" \
            ")\n" \
            ")"
        )

    def _make_literal(self, p, p_type):
        assert int(p) == p
        p = int(p)
        if p_type == 'field':
            return f'#f{p}'
        if p_type == 'u64':
            return f'#x{to_hex_str(twos_complement(p), 16)}'
        if p_type == 'u32':
            return f'#x{to_hex_str(twos_complement(p), 8)}'
        raise ValueError(p_type, p)

    def _make_param(self, param):

        _, p_name, p_type, p_data = param

        # PATTERN: type
        pattern = re.compile(r'^(field|u32|u64)$')
        if pattern.match(p_type):
            # print(p_name, p_type)
            return f'\t({p_name} {self._make_literal(p_data, p_type)})' + '\n'

        # PATTERN: type[no_samples]
        pattern = re.compile(r'^(field|u32|u64)\[(\d+)\]$')
        if pattern.match(p_type):
            p_type_literal, no_samples = pattern.findall(p_type).pop()
            if len(p_data) < int(no_samples):    
                pad_entries = [0] * (int(no_samples) - len(p_data))
                p_data += pad_entries
            return "\n".join([ f'\t({p_name}.{idx} {self._make_literal(elem, p_type_literal)})' for idx, elem in enumerate(p_data) ]) + "\n"

        # PATTERN: type[no_samples][no_features]
        pattern = re.compile(r'^(field|u32|u64)\[(\d+)\]\[(\d+)\]$')
        if pattern.match(p_type):
            p_type_literal, no_samples, no_features = pattern.findall(p_type).pop()
            if len(p_data) < int(no_samples):    
                pad_entries = [ [0 for _ in range(int(no_features))] for _ in range(int(no_samples) - len(p_data)) ]
                p_data += pad_entries
            return "\n".join([ f'\t({p_name}.{elem_idx}.{elem_i_idx} {self._make_literal(elem_i, p_type_literal)})' for elem_idx, elem in enumerate(p_data) for elem_i_idx, elem_i in enumerate(elem) ]) + "\n"

        raise ValueError(f"Unknown {p_name} {p_type}")