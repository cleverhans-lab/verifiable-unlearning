from pathlib import Path
import json
from collections import defaultdict
import pandas as pd
import re

TRIALS_DIR = Path("/root/verifiable-unlearning/evaluation/trials/")

def format_running_time(running_time):
    running_time = running_time / 1000
    return f"{running_time // 3600:2.0f}h {(running_time % 3600) // 60:2.0f}m {(running_time % 60):2.0f}s"


for trials_dir in TRIALS_DIR.glob('*'):

    if trials_dir.name == 'unsorted':
        continue

    print(f'\n{trials_dir.name.upper()}')
    results = defaultdict(list)
    for trial in trials_dir.rglob('circ.log.txt'):
        log = trial.read_text()        
        results['name']  += [ trial.parent.name ]

        results['setup'] += [ format_running_time(int(re.findall(r'\[\+\] Generate public parameters\n    took (\d+) ms', log).pop())) ]
        results['prove'] += [ format_running_time(int(re.findall(r'\[\+\] Prove\n    took (\d+) ms', log).pop())) ]
        results['verify'] += [ format_running_time(int(re.findall(r'\[\+\] Verify\n    took (\d+) ms', log).pop())) ]

        results['compile'] += [ format_running_time(int(re.findall(r'\[\+\] Compile circuit\n    took (\d+) ms', log).pop())) ]
        results['optimize'] += [ format_running_time(int(re.findall(r'\[\+\] Optimize circuit\n    took (\d+) ms', log).pop())) ]
        results['count'] += [ format_running_time(int(re.findall(r'\[\+\] Get instance size\n    took (\d+) ms', log).pop())) ]
        results['compile_total'] += [ format_running_time(  int(re.findall(r'\[\+\] Compile circuit\n    took (\d+) ms', log).pop()) 
                                                         + int(re.findall(r'\[\+\] Optimize circuit\n    took (\d+) ms', log).pop())
                                                         + int(re.findall(r'\[\+\] Get instance size\n    took (\d+) ms', log).pop()))  ]

        results['r1cs'] += [ int(re.findall(r'- final R1CS size: (\d+)', log).pop()) ]
        results['r1cs_pre-opt'] += [ int(re.findall(r'- pre-opt R1CS size: (\d+)', log).pop()) ]
        results['r1cs_opt-factor'] += [ f"{results['r1cs'][-1]/results['r1cs_pre-opt'][-1]:.2f}" ]

        try:
            results['proof_size'] += [ int(re.findall(r'proof size (\d+)', log).pop()) ]
        except:
            results['proof_size'] += [ None ]

        results['num_cons'] += [ int(re.findall(r'num_cons (\d+)', log).pop()) ]
        results['num_vars'] += [ int(re.findall(r'num_vars (\d+)', log).pop()) ]
        results['num_inputs'] += [ int(re.findall(r'num_inputs (\d+)', log).pop()) ]
        results['num_non_zero_entries'] += [ int(re.findall(r'num_non_zero_entries (\d+)', log).pop()) ]

    print(pd.DataFrame(results))
