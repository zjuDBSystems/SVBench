import threading
import sys
import importlib
import os
import numpy as np
import portalocker
import time

from Tasks import data_valuation, federated_learning, result_interpretation
from calculator import Shapley
from sampler import Sampler
from output import Output


OUT_PRINT_FLUSH_INTERVAL = 5
UTILITY_RECORD_FILEWRITE_INTERVAL = 10
BENCHMARK_ALGO = {
    'MC': ('permutation'),
    'MLE': ('coalition'),
    'GT': ('coalition'),
}
BENCHMARK = {
    'DV': {
        'iris': (120),
        'wine': (142),
    },
    'FL': {
        'cifar': (10),
        'mnist': (10),
    },
    'RI': {
        'iris': (4),
        'wine': (13),
    },
    'DSV': {
        'iris': (12),
        'wine': (14),
    }
}


class Task():
    def __init__(self, args):
        self.args = args

        self.init_flag = True

        self.task = args.task
        self.utility_function = self.utility_computation_func_load(
            args.get('utility_function'))
        if self.utility_function is None:
            self.init_flag = False
            return

        self.utility_record_file = args.utility_record_file
        self.utility_records = self.read_history_utility_record()
        if self.utility_records is None:
            self.init_flag = False
            return
        self.utility_record_write_lock = threading.Lock()
        self.utility_record_filewrite_lock = threading.Lock()
        self.utility_record_write_flag = False
        self.dirty_utility_record_num = 0

        if args.task in BENCHMARK:
            self.player_num = BENCHMARK[args.task][args.dataset][0]
            full_check_type = BENCHMARK_ALGO[args.algo][0]
        else:
            self.player_num = args.player_num
            full_check_type = args.full_check

        self.GA = args.GA
        self.TSS = args.TSS
        self.parallel_threads_num = 1 if self.GA else args.parallel_threads_num

        self.shapley = Shapley(
            player_num=self.player_num,
            utility_function=self.utility_computation_call,
            cache_size=args.SV_cache_size,
            argorithm=args.argo,
            truncation=args.TC,
            truncation_threshold=args.TC_threshold,
            privacy_protection_measure=args.privacy,
            privacy_protection_level=args.privacy_level,
            parallel_threads_num=self.parallel_threads_num,
            sampler=Sampler(
                sampling_strategy=args.sampling,
                algo=args.algo, player_num=self.player_num),
            output=Output(
                convergence_threshold=args.convergence_threshold,
                cache_size=args.cache_size,
                player_num=self.player_num,
                full_check_type=full_check_type))

        self.task_terminated = False
        self.flush_event = threading.Event()

    def utility_computation_func_load(self, utility_function_api):
        try:
            if self.task == 'DV':
                DV = data_valuation.DV(self.args)
                return DV.utility_computation
            # elif self.task == 'FL':
            #     return federated_learning.utility_computation
            # elif self.task == 'RI':
            #     return result_interpretation.utility_computation
            else:
                return utility_function_api
        except Exception as e:
            print(
                f"Get utility function {utility_function_api} error:\n{e}")
            return None

    def utility_computation_call(self, player_list):
        utility_record_idx = str(
            sorted(player_list) if not self.GA else player_list)
        if utility_record_idx in self.utility_records:
            return self.utility_records[utility_record_idx][0], -1

        start_time = time.time()
        try:
            if self.task in BENCHMARK:
                utility = self.utility_function(
                    player_list, GA=self.GA, TSS=self.TSS)
            else:
                utility = self.utility_function(player_list)
            time_cost = time.time() - start_time
            if self.write_utility_record(utility_record_idx, utility, time_cost) == -1:
                exit(-1)
        except Exception as e:
            print(
                f"Call utility computation error:\n{e}")
            exit(-1)
        return utility, time_cost

    def printFlush(self):
        while not self.task_terminated:
            if self.flush_event.wait(OUT_PRINT_FLUSH_INTERVAL):
                sys.stdout.flush()

    def run_print_flush_thread(self):
        thread = threading.Thread(target=self.printFlush)
        thread.daemon = True
        thread.start()

    def pre_exp_statistic(self):
        utility_computation_timecost = dict()
        for player_idx in range(self.player_num):
            if self.task == 'DV':
                utility, time_cost = self.utility_computation_call(
                    range(player_idx))
                print(
                    f'Computing utility with {player_idx + 1} players tasks {time_cost} timeCost {utility} utility...')
                utility_computation_timecost[player_idx] = time_cost
        print('Average time cost for computing utility: ',
              np.mean(list(utility_computation_timecost.values())))

    def read_history_utility_record(self):
        self.utility_record_file = f'./Tasks/utility_records/{self.task}{"_GA" if self.GA else ""}.log' \
            if self.utility_record_file == '' else self.utility_record_file
        if os.path.exists(self.utility_record_file):
            with open(self.utility_record_file, 'r', encoding='utf-8') as file:
                try:
                    portalocker.lock(file, portalocker.LOCK_SH)
                    utility_records = eval(file.read().strip())
                except Exception as e:
                    print(
                        f"Read utility record file {self.utility_record_file} error:\n{e}")
                    utility_records = None
                finally:
                    portalocker.unlock(file)
                    return utility_records
        else:
            return {str([]): (0, 0)}

    def write_utility_record(self, utility_record_idx, utility, time_cost):
        if utility_record_idx in self.utility_records:
            return 0

        with self.utility_record_write_lock:
            self.utility_records[utility_record_idx] = (utility, time_cost)
            self.dirty_utility_record_num += 1

        ret = 0
        if self.dirty_utility_record_num > UTILITY_RECORD_FILEWRITE_INTERVAL    \
                and self.utility_record_filewrite_lock.acquire(blocking=False):
            log_file_exist = os.path.exists(self.utility_record_file)
            try:
                with portalocker.Lock(self.utility_record_file, mode="a+", timeout=0) as file:
                    with self.utility_record_write_lock:
                        if log_file_exist:
                            file.seek(0)
                            self.utility_records.update(
                                eval(file.read().strip()))
                        self.dirty_utility_record_num = 0
                    file.seek(0)
                    file.write(str(self.utility_records))
            except Exception as e:
                print(
                    f"Write utility record file {self.utility_record_file} error:\n{e}")
                ret = -1
            finally:
                self.utility_record_filewrite_lock.release()

        return ret

    def run(self):
        self.run_print_flush_thread()

        self.pre_exp_statistic()

        self.shapley.SV_calculate()
