#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import argparse
import numpy as np


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--log_file', type=str, default='',
                        help="path of log file")
    parser.add_argument('--num_parallelThreads', type=int, default=1,
                        help="number of parallelThreads")
    parser.add_argument('--manual_seed', type=int, default=42,
                        help="random seed")
    parser.add_argument('--gpu', type=int, default=-1,
                        help="GPU ID, -1 for CPU")
    parser.add_argument('--cuda', type=str, default=None, help="")

    # dataset parameters
    parser.add_argument('--dataset', type=str, default="mnist",
                        help="{MNIST, Iris}")
    parser.add_argument('--num_classes', type=int, default=10,
                        help="number of classes")
    parser.add_argument('--num_feature', type=int, default=4,
                        help="number of features")
    parser.add_argument('--num_trainDatasets', type=int, default=1,
                        help="range(1,10)")
    parser.add_argument('--data_allocation', type=int, default=0,
                        help="{0,1,2,3,4,5}")
    parser.add_argument('--data_size_group', type=int, default=1,
                        help="data_size_group")
    parser.add_argument('--group_size', type=str, default='10',
                        help='group_size')
    parser.add_argument('--data_size_mean', type=float, default=100.0,
                        help="data_size_mean")
    parser.add_argument('--multiplier', type=str, default='1',
                        help='multiplier for data_size_mean of each group')

    # task parameters
    parser.add_argument('--task', type=str, default="DV",
                        help="{DV, FL, FA}")
    parser.add_argument('--test_bs', type=int, default=128,
                        help="FL task  parameter")

    # DV task parameter
    parser.add_argument('--tuple_to_set', type=int, default=0,
                        help="DV task parameter")

    # DV & FA task parameters
    parser.add_argument('--ep', type=int, default=50,
                        help="FL task  parameter")
    parser.add_argument('--bs', type=int, default=64,
                        help="FL task  parameter")
    # FL task parameters
    parser.add_argument('--maxRound', type=int, default=10,
                        help="FL task  parameter")
    parser.add_argument('--local_ep', type=int, default=3,
                        help="FL task  parameter")
    parser.add_argument('--local_bs', type=int, default=64,
                        help="FL task  parameter")
    parser.add_argument('--num_clients', type=int, default=10,
                        help="FL task  parameter")

    # model parameters
    parser.add_argument('--model_name', type=str, default='KNN',
                        help="{KNN, CNN, Tree, Linear}")
    parser.add_argument('--test_metric', type=str, default="tst_accuracy",
                        help="{tst_accuracy, tst_F1, tst_loss, prediction}")
    # KNN model parameters
    parser.add_argument('--n_neighbors', type=int, default=5,
                        help="KNN model parameter")
    # Tree model parameters
    parser.add_argument('--tree_maxDepth', type=int, default=5,
                        help="KNN model parameter")
    # CNN model parameters
    parser.add_argument('--num_channels', type=int, default=1,
                        help="CNN model parameter")
    parser.add_argument('--lr', type=float, default=0.1,
                        help="CNN model parameter")
    parser.add_argument('--momentum', type=int, default=0,
                        help="CNN model parameter")
    parser.add_argument('--decay_rate', type=int, default=1,
                        help="CNN model parameter")
    parser.add_argument('--weight_decay', type=int, default=0,
                        help="CNN model parameter")
    parser.add_argument('--max_norm', type=int, default=5,
                        help="CNN model parameter")

    # SV parameters
    parser.add_argument('--base_compFunc', type=str, default="classical",
                        help="{classical, RE, MLE, GT, CP}")
    parser.add_argument('--convergence_threshold', type=float, default=0.5,
                        help="approximation convergence_threshold")
    parser.add_argument('--scannedIter_maxNum', type=float, default=np.inf,
                        help="approximation convergence_threshold")
    parser.add_argument('--MLE_maxInterval', type=int, default=10000,
                        help="maximum number of tests on different discretization in MLE")
    parser.add_argument('--GT_epsilon', type=float, default=0.00001,
                        help="epsilon used in GT")
    parser.add_argument('--num_measurement', type=int, default=10,
                        help="num_measurement used in CP")
    parser.add_argument('--CP_epsilon', type=float, default=0.00001,
                        help="epsilon used in CP")

    parser.add_argument('--sampling_strategy', type=str, default="random",
                        help="{random, antithetic, stratified}")
    parser.add_argument('--truncation', type=bool, default=False,
                        help="{False, True}")
    parser.add_argument('--truncationThreshold', type=float, default=0.01,
                        help="truncation threshold")
    parser.add_argument('--gradient_approximation', type=bool, default=False,
                        help="{False, True}")
    parser.add_argument('--testSampleSkip', type=bool, default=False,
                        help="{False, True}")

    # SV's privacy protection parameters
    parser.add_argument('--privacy_protection_measure', type=str, default=None,
                        help="{None, DP, QT, DR}")
    parser.add_argument('--privacy_protection_level', type=float, default=0.0,
                        help="privacy_protection_level")

    # attack parameters
    parser.add_argument('--attack_type', type=str, default=None,
                        help="{MIA,FIA}")
    parser.add_argument('--maxIter_in_MIA', type=int, default=32)

    args = parser.parse_args()
    return args
