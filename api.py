from SV import Shapley
import numpy as np


# default arguments
args = {
    # SV parameters
    'num_parallelThreads': 1,
    'base_compFunc': 'MC',
    'convergence_threshold': 0.5,
    'scannedIter_maxNum': np.inf,
    'MLE_maxInterval': 10000,
    'GT_epsilon': 0.00001,
    'num_measurement': 10,
    'CP_epsilon': 0.00001,
    'sampling_strategy': 'random',
    'truncation': False,
    'truncationThreshold': 0.01,
    'gradient_approximation': False,
    'testSampleSkip': False,

    # SV's privacy protection parameters
    'privacy_protection_measure': None,
    'privacy_protection_level': 0.0,
}


def SV_calc(player_num, taskUtilityFunc, **kwargs):
    for key, value in kwargs.items():
        setattr(args, key, value)

    shap = Shapley(player_num, taskUtilityFunc, args)
    return shap.CalSV()
