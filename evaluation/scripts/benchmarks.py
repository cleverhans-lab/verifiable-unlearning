
from pathlib import Path
import json

TRIALS_DIR = Path("/root/verifiable-unlearning/evaluation/trials")

def format_running_time(running_time):
    return f"{running_time // 3600:2.0f}h {(running_time % 3600) // 60:2.0f}m {(running_time % 60):2.0f}s"

print("\nDataset".upper())
dataset_trials = sorted(TRIALS_DIR.joinpath('benchmarks', 'dataset').rglob("stats.json"))
for trial in dataset_trials:
    samples = trial.parent.name
    stats = json.loads(trial.read_text())
    print(f"[+] {samples}")
    for k, v in stats.items():
        if k == 'no_constraints':
            print(f"    {k:<20}: {v}")
        else:
            print(f"    {k:<20}: {format_running_time(v)} ({v:.2f} {v/int(samples):.2f} ) ")

print("\nModel".upper())
model_trials = sorted(TRIALS_DIR.joinpath('benchmarks', 'model').rglob("stats.json"))
for trial in model_trials:
    samples = trial.parent.name
    stats = json.loads(trial.read_text())
    print(f"[+] {samples}")
    for k, v in stats.items():
        if k == 'no_constraints':
            print(f"    {k:<20}: {v}")
        else:
            print(f"    {k:<20}: {format_running_time(v)} ({v:.2f} {v/int(samples):.2f} ) ")

print("\nMembership".upper())
membership_trials = sorted(TRIALS_DIR.joinpath('benchmarks', 'membership').rglob("stats.json"))
for trial in membership_trials:
    samples = trial.parent.name
    stats = json.loads(trial.read_text())
    print(f"[+] {samples}")
    for k, v in stats.items():
        print(k)
        if k == 'no_constraints':
            print(f"    {k:<20}: {v}")
        else:
            print(f"    {k:<20}: {format_running_time(v)} ({v:.2f} {v/int(samples):.2f} ) ")
