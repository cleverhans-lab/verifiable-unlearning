from pathlib import Path
import json
from collections import defaultdict
import pandas as pd
import re

def format_running_time(running_time):
    running_time = running_time / 1000
    return f"{(running_time) // 60:2.0f}m {(running_time % 60):2.0f}s"

trials_dir = Path("/root/verifiable-unlearning/evaluation/trials/models")

print(f'\n{trials_dir.name.upper()}')
results = defaultdict(list)
for trial in trials_dir.rglob('circ.log.txt'):
    log = trial.read_text()        
    results['name']  += [ trial.parent.name ]

    results['setup'] += [ (int(re.findall(r'\[\+\] Generate public parameters\n    took (\d+) ms', log).pop())) ]
    results['prove'] += [ (int(re.findall(r'\[\+\] Prove\n    took (\d+) ms', log).pop())) ]
    results['verify'] += [ (int(re.findall(r'\[\+\] Verify\n    took (\d+) ms', log).pop())) ]

    results['compile'] += [ (int(re.findall(r'\[\+\] Compile circuit\n    took (\d+) ms', log).pop())) ]
    results['optimize'] += [ (int(re.findall(r'\[\+\] Optimize circuit\n    took (\d+) ms', log).pop())) ]
    results['count'] += [ (int(re.findall(r'\[\+\] Get instance size\n    took (\d+) ms', log).pop())) ]
    results['compile_total'] += [ (  int(re.findall(r'\[\+\] Compile circuit\n    took (\d+) ms', log).pop()) 
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

def format_entry(field_name, field, format):
    if not format:
        return field
    if field_name in ['r1cs', 'num_non_zero_entries']:
        return f'{int(field):,}'
    else:
        return format_running_time(field)

rows = [
    ['Linear Regression', 'linear_regression', '\\\\'],
    ['\\rule{0pt}{2ex}%<--- do not remove','\\\\'],
    ['Logistic Regression', 'logistic_regression', '\\\\'],
    ['\\rule{0pt}{2ex}%<--- do not remove','\\\\'],
    ['Neural Network ($N=2$)', 'neural_network_2', '\\\\'],
    ['\\rule{0pt}{2ex}%<--- do not remove','\\\\'],
    ['Neural Network ($N=4$)', 'neural_network_4', '\\\\']
]

def get_field(results, classifier, field_name, format=True):
    idx = results["name"].index(classifier)
    return format_entry(field_name, results[field_name][idx], format)

for row in rows:
    if len(row) == 2:
        print(f'{row[0]} {row[1]}')
        continue
    label, classifier, line_end = row
    print(f"{label} && {get_field(results, classifier, 'r1cs')} && {get_field(results, classifier, 'prove')} && {get_field(results, classifier, 'verify')} {line_end}")
