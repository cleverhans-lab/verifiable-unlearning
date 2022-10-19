from pathlib import Path
import json

trial_dirs = Path.home().joinpath('verifiable-unlearning/evaluation/trials/models')

def format_running_time(running_time):
    return f"{running_time // 3600:2.0f}h {(running_time % 3600) // 60:2.0f}m {(running_time % 60):2.0f}s"

print(f'{"":>20}  Setup           |  Prove                         | Verify')
print(f'{"":>20}  Time       Gate |  Time        Gate  Constraints | Time')

for classifier in ["linear_regression", "logistic_regression", "neural_network_2", "neural_network_4"]:
    trial_dir = trial_dirs.joinpath(classifier)
    stats = json.loads(trial_dir.joinpath("stats.json").read_text())
    setup_gate = stats["setup"] / stats["no_constraints"] * 10**6
    prove_gate = stats["prove"] / stats["no_constraints"] * 10**6
    print(f'{classifier:>20} {format_running_time(stats["setup"])} {setup_gate:4.0f} | {format_running_time(stats["prove"])} {prove_gate:4.0f}    {stats["no_constraints"]:>8}   | {stats["verify"]:.4f}s')
