import argparse
import sys
import numpy as np
import copy
import random
import torch
import time
import json
import portalocker
import os

from collections import namedtuple

from config import Task, BENCHMARK


def para_set(args):
    required = ['task']
    # only required for bench tasks:
    bench_required = ['dataset']
    # only required for user-specific tasks:
    user_required = ['utility_function', 'full_check', 'player_num']
    paras = {'log_file': 'std',
             'utility_record_file': '',
             'TC': False,
             'TC_threshold': 0.01,
             'SV_cache_size': 5,
             'algo': 'MC',
             'sampling': 'random',
             'convergence_threshold': 0.1,
             'parallel_threads_num': 1,
             'GA': False,
             'TSS': False,
             'manual_seed': 42,
             'privacy': None,
             'privacy_level': 0.5,
             'utility_function': None
             }
    for required_key in required:
        if required_key not in args:
            print(f'Missing required argument: {required_key}')
            return -1
    if args.get('task') in BENCHMARK:
        for bench_key in bench_required:
            if bench_key not in args:
                print(
                    f'Missing required argument for benchmark task: {bench_key}')
                return -1
    else:
        for user_key in user_required:
            if user_key not in args:
                print(
                    f'Missing required argument for user-specific task: {user_key}')
                return -1

    args.update({key: value for key, value in paras.items() if key not in args})
    return 0


def open_log_file(log_file):
    if log_file != 'std':
        try:
            file = open(log_file, 'w')
            sys.stdout = file
        except Exception as e:
            print(f"Open log file error:\n{e}")
            return -1
    return 0


def sv_calc(**kwargs):
    if para_set(kwargs) == -1:
        exit(-1)
    print(f'Experiment arguments:\n{kwargs}')
    ARGS = namedtuple('ARGS', kwargs.keys())
    kwargs = ARGS(**kwargs)

    if open_log_file(kwargs.log_file) == -1:
        exit(-1)

    task = Task(args=kwargs)
    if not task.init_flag:
        exit(-1)

    task.run()


if __name__ == '__main__':
    sv_calc(task='DV',
            dataset='iris')
