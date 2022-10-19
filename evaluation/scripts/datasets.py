import json
from collections import defaultdict
from itertools import product
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt


#
# CONFIG
#

TRIALS_DIR = Path.home().joinpath('verifiable-unlearning/evalauation/trials/classification')
PLOTS_DIR = Path.home().joinpath('verifiable-unlearning/evlautation/plots')
PLOTS_DIR.mkdir(exist_ok=True, parents=True)

# setup matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')

#
# PARSE TRIALS
#
dataset_names = [
    "analcatdata_creditscore",
    "postoperative_patient_data",
    "analcatdata_cyyoung9302",
    "corral",
]

classifier_names =  [
    'linear_regression', 
    'logistic_regression',
    'neural_network_2',
    'neural_network_4'
]

running_times = defaultdict(lambda: defaultdict(dict))
classifier_accs = []
for dataset_name, classifier_name in product(dataset_names, classifier_names):
    trial_dir = TRIALS_DIR / dataset_name / classifier_name
    try:
        print(f'[+] {dataset_name} {classifier_name}')
        stats = json.loads(trial_dir.joinpath("stats.json").read_text())
        try:
            classifier_accs += [ json.loads(trial_dir.joinpath('model.json').read_text())['acc'] ]
        except:
            print(f"    no model accuracy available")
        running_times[dataset_name][classifier_name]['setup'] = stats['setup']
        running_times[dataset_name][classifier_name]['proof'] = stats['prove']

        print(f'    Setup: {stats["setup"] // 3600:.0f}h {(stats["setup"] % 3600) // 60:.0f}m {(stats["setup"] % 60):>2.0f}s')
        print(f'    Proof: {stats["prove"] // 3600:.0f}h {(stats["prove"] % 3600) // 60:.0f}m {(stats["prove"] % 60):>2.0f}s')
    except:
        pass
print(f"[+] Model accuracies: {min(classifier_accs):.2f}-{max(classifier_accs):.2f}")


#
# PLOT
#

colors = { 
    'green' : '#798376',
    'blue' : '#41678B',
    'orange' : '#CB9471',
    'red' : '#B65555',
    'mint' : '#6AA56E',
    'grey' : '#616161'
}

fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(4.5, 5.5), gridspec_kw={'height_ratios': [2, 1]})

# setup axis
axs[0].grid(alpha=.7)
axs[0].spines['right'].set_visible(False)
axs[0].spines['top'].set_visible(False)
axs[0].set_ylabel("$\Pi.\mathsf{Setup}$", fontsize="large")
axs[0].set_axisbelow(True)
axs[0].legend(labels=['Linear Regression', 'Logistic Regression', 'Neural Network ($N=2$)', 'Neural Network ($N=4$)'], 
              loc="upper right", ncol=2, fontsize="small")

# Y ticks
step = 1800
max_running_time = int(3600 * 3.5)
ticks_setup = [y for y in range(0, max_running_time+step, step)], 
labels_setup =[f'{y // 3600:.0f}h {(y % 3600) // 60:>2}m' for y in range(0, max_running_time+step, step)]
axs[0].set_yticks(ticks_setup, labels=labels_setup)

# setup axis
axs[1].grid(alpha=.7)
axs[1].spines['right'].set_visible(False)
axs[1].spines['top'].set_visible(False)
axs[1].set_xlabel(f"Datasets", fontsize="large")
axs[1].set_ylabel("$\Pi.\mathsf{Prove}$", fontsize="large")
axs[1].set_axisbelow(True)

# Y ticks
max_running_time_proof = 4800
step_proof = 1800
ticks_proof = [y for y in range(0, int(max_running_time_proof)+step_proof, step_proof)]
labels_proof = [f'{y // 3600:.0f}h {(y % 3600) // 60:>2}m' 
                 for y in range(0, int(max_running_time_proof)+step_proof, step_proof)]
axs[1].set_yticks(ticks_proof, labels=labels_proof)

# X ticks
X = [
   "creditscore",
   "patient",
   "cyyoung",
   "corral",
]
axs[1].set_xticks([idx for idx in range(len(X))], labels=[f'{x}' for x in X])

#
# BARS
#
offsets = {
    0: -0.3,
    1: -0.1,
    2: 0.1,
    3: +0.3
}

colors_map = {
    'linear_regression' : 'green', 
    'logistic_regression' : 'orange',
    'neural_network_2' : 'grey',
    'neural_network_4' : 'blue'
}

for idx, classifier in enumerate(classifier_names):
    X_setup = []
    X_proof = []
    for dataset_name in dataset_names:
        try:
            setup = running_times[dataset_name][classifier]['setup']
            proof = running_times[dataset_name][classifier]['proof']
        except:
            setup = 0
            proof = 0
        X_setup += [setup]
        X_proof += [proof]
    ticks = [ x+offsets[idx] for x in range(len(dataset_names)) ]
    axs[0].bar(ticks, X_setup, 0.2, color=colors[colors_map[classifier]]+'E8')
    axs[1].bar(ticks, X_proof, 0.2, color=colors[colors_map[classifier]]+'E8')

# save plot
fig.tight_layout()
plt.savefig(PLOTS_DIR.joinpath(f'datasets.pdf'), dpi=400)
plt.close()
