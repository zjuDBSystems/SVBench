import argparse
import numpy as np
import sys
from svbench import sv_calc
from Tasks import federated_learning, result_interpretation


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--log_file', type=str, default='std',
                        help="path of log file")
    parser.add_argument('--num_parallel_threads', type=int, default=1,
                        help="number of parallelThreads")
    parser.add_argument('--manual_seed', type=int, default=42,
                        help="random seed")

    # dataset parameters
    parser.add_argument('--dataset', type=str, default="mnist",
                        help="{MNIST, Iris}")

    # task parameters
    parser.add_argument('--task', type=str, default="DV",
                        help="{DV, FL, FA}")

    # SV parameters
    parser.add_argument('--conv_check_num', type=int, default=5,
                        help="SV cache_size")
    parser.add_argument('--base_algo', type=str, default="MC",
                        help="{MC, RE, MLE, GT, CP}")
    parser.add_argument('--convergence_threshold', type=float, default=0.01,
                        help="approximation convergence_threshold")
    parser.add_argument('--checker_mode', type=str, default='SV_var',
                        help="checker_mode: SV_var or comp_count")
    
    parser.add_argument('--sampling_strategy', type=str, default="random",
                        help="{random, antithetic, stratified}")
    parser.add_argument('--optimization_strategy', type=str, default='',
                        help="")
    parser.add_argument('--TC_threshold', type=float, default=0.01,
                        help="truncation threshold")

    # SV's privacy protection parameters
    parser.add_argument('--privacy_protection_measure', type=str, default=None,
                        help="{None, DP, QT, DR}")
    parser.add_argument('--privacy_protection_level', type=float, default=0.0,
                        help="privacy_protection_level")
    
    # attack parameters
    parser.add_argument('--attack', type=str, default=None, 
                        help="{MIA,FIA}")
    parser.add_argument('--maxIter_in_MIA', type=int, default=32)
    parser.add_argument('--num_querySample_in_MIA', type=int, default=8)
    parser.add_argument('--num_samples_each_class', type=int, default=3)
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = args_parser()
    if args.log_file!='std':
        file = open(args.log_file, 'w')
        sys.stdout = file
        
    if args.task == 'FL' and 'GA' in args.optimization_strategy:
        FL = federated_learning.FL(
            dataset=args.dataset,
            manual_seed=args.manual_seed,
            GA='GA' in args.optimization_strategy,
            TSS='TSS' in args.optimization_strategy)
        sys.stdout.flush()
        
        round_SV = dict()
        round_SV_var = dict()
        for ridx in range(FL.max_round):
            FL.ridx = ridx
            SV, SV_var = sv_calc(
                task = f'FL_{args.dataset}_R{ridx}',
                dataset = args.dataset,
                player_num = len(FL.players),
                utility_function = FL.utility_computation,
                base_algo = args.base_algo,
                conv_check_num = args.conv_check_num,
                convergence_threshold = args.convergence_threshold,
                checker_mode = args.checker_mode,
                sampling_strategy = args.sampling_strategy,
                optimization_strategy = args.optimization_strategy,
                TC_threshold = args.TC_threshold,
                privacy_protection_measure = args.privacy_protection_measure,
                privacy_protection_level = args.privacy_protection_level,
                log_file = args.log_file,
                num_parallel_threads = args.num_parallel_threads,
                manual_seed = args.manual_seed
            )
            round_SV[ridx] = SV
            round_SV_var[ridx] = SV_var
            print(args.task, 'SV results:', SV)
            sys.stdout.flush()
            
        print('average final results:',
              np.mean(np.array([[value for value in sv.values()]
                                for sv in round_SV.values()]), 0))
    elif args.task == 'RI':
        RI = result_interpretation.RI(
            dataset=args.dataset,
            manual_seed=args.manual_seed)
        sys.stdout.flush()
        
        RI.testSampleFeatureSV = dict()
        RI.testSampleFeatureSV_var = dict()
        # compute SV for only selected test samples for saving time cost
        for test_idx in RI.selected_test_samples:
            RI.Tst.idxs = RI.complete_Tst_idx[test_idx:test_idx+1]
            SV, SV_var = sv_calc(
                task = f'RI_{args.dataset}_Idx{test_idx}',
                dataset = args.dataset,
                player_num = len(RI.players),
                utility_function = RI.utility_computation,
                base_algo = args.base_algo,
                conv_check_num = args.conv_check_num,
                convergence_threshold = args.convergence_threshold,
                checker_mode = args.checker_mode,
                sampling_strategy = args.sampling_strategy,
                optimization_strategy = args.optimization_strategy,
                TC_threshold = args.TC_threshold,
                privacy_protection_measure = args.privacy_protection_measure,
                privacy_protection_level = args.privacy_protection_level,
                log_file = args.log_file,
                num_parallel_threads = args.num_parallel_threads,
                manual_seed = args.manual_seed
            )

            RI.testSampleFeatureSV[test_idx] = SV
            RI.testSampleFeatureSV_var[test_idx] = dict([
                (fidx, np.var(SV_var[fidx]))
                for fidx in SV_var.keys()])
            if args.dataset != 'adult':
                print(f'\n {test_idx} test sample data: ', RI.Tst.dataset[test_idx], 
                      f'\n {test_idx} test sample label: ', RI.Tst.labels[test_idx])
            print('SV of test sample %s/%s: ' % (test_idx, len(RI.selected_test_samples)),
                  RI.testSampleFeatureSV[test_idx], '\n')
            sys.stdout.flush()
            
        RI.Tst.idx = RI.complete_Tst_idx
        print('average final results:',
              np.mean(np.array([[value for value in sv.values()]
                                for sv in RI.testSampleFeatureSV.values()]), 0))
    else:
        sv_calc(task = args.task,
                dataset = args.dataset,
                base_algo = args.base_algo,
                conv_check_num = args.conv_check_num,
                convergence_threshold = args.convergence_threshold,
                checker_mode = args.checker_mode,
                sampling_strategy = args.sampling_strategy,
                optimization_strategy = args.optimization_strategy,
                TC_threshold = args.TC_threshold,
                privacy_protection_measure = args.privacy_protection_measure,
                privacy_protection_level = args.privacy_protection_level,
                log_file = args.log_file,
                num_parallel_threads=args.num_parallel_threads,
                manual_seed = args.manual_seed)
        
