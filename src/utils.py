import logging
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import random
import numpy as np

def parse_and_group_arguments(parser):
    # parse and group arguments
    # -> args w/o group are added directly to result dict
    # -> args w/  group are first merged together
    args = parser.parse_args()
    args_dict = {}
    for group in parser._action_groups:  # dirty, but argparse does not support this natively
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        if group.title in ["positional arguments", "optional arguments"]:
            # args w/o group 
            args_dict.update(group_dict)
        else:
            # args w/ group
            args_dict[group.title] = group_dict
    if 'help' in args_dict:
        del args_dict['help']
    else:
        assert 'options' in args_dict
        args_dict.update(args_dict['options'])
        del args_dict['options']
        del args_dict['help']
    return args_dict

def setup_working_dir(trials_dir, trial_name):
    working_dir = trials_dir.joinpath(trial_name)
    if working_dir.is_dir():
        print(f'[!] Working dir already exist: {working_dir}')
        if input("    Enter yes to overwrite: ") == 'yes':
            print(f'    -> removed dir')
            shutil.rmtree(working_dir)
    working_dir.mkdir(parents=True)
    return working_dir

def setup_file_logger(working_dir):
    working_dir.mkdir(exist_ok=True, parents=True)
    log_file: Path = working_dir / f'log.txt'
    logger: logging.Logger = logging.getLogger(working_dir.name)
    file_handler = logging.FileHandler(log_file.as_posix())
    file_handler.setFormatter(None)
    file_handler.setLevel(logging.DEBUG)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    print("\033c") # clear terminal
    logger.info(f"\n[+] Working dir @ {working_dir}")
    return logger

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
