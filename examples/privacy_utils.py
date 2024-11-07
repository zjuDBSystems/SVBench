import numpy as np


def quantization(SVs, level):
    # level=0~1
    level = int(len(SVs)*level)
    sorted_SV = sorted(SVs.values())
    interval = int(np.ceil(len(SVs)/level))
    quantization_map = dict()
    for level_index in range(level):
        index_lower = level_index*interval
        index_higher = min(len(SVs), (level_index+1)*interval)
        for sv_idx in range(index_lower, index_higher):
            quantization_map[
                sorted_SV[sv_idx]
            ] = sorted_SV[int((index_lower + index_higher)/2)]
        if index_higher >= len(SVs):
            break
    SVs = dict([(k, quantization_map[v]) for (k, v) in SVs.items()])
    return SVs


def differential_privacy(SVs, level):
    '''
    it introduces Gaussian noise aligned with the function’s global sensitivity, 
    which is the maximum output change when a single data point is added/removed 
    from any given dataset
    Guassian std: σ = Δ_{max}*np.sqrt(1.25/δ)/ε, 
    where δ and ε are parameter of DP

    Note: It is hard to bound the global sensitivity for KNN-SV! 
    Even if we are able to bound the global sensitivity, 
    the bound will highly likely be large compared with the magnitude of KNN-SV.
    '''
    '''
    In DP-KNN, the variance of normal distribution is 
    1/(K(K+1)) and (1/(K(K+1)))^2 for cases without and with subsampling,
    where K is the number of neighbors in KNN model and
    K=5 is set for all of that paper's experiments 
    '''
    noise = np.random.normal(0, level, len(SVs))
    SVs = dict([(k, v + noise[idx])
                for idx, (k, v) in enumerate(SVs.items())])
    return SVs


def dimension_reduction(SVs, level, reference_var=dict()):
    # original paper: exposing the top k important players
    # based on their Shapley variance instead of absolute value intensity
    # here: for simplicity, we expose based on absolute value intensity
    if len(reference_var) == len(SVs):
        exposed_idx = dict(
            sorted(reference_var.items(),
                   key=lambda item: item[1])[-int(len(reference_var)*level):]
        ).keys()

    else:
        exposed_idx = dict(
            sorted(SVs.items(),
                   key=lambda item: item[1])[-int(len(SVs)*level):]
        ).keys()
    SVs = dict([(key, (SVs[key] if key in exposed_idx else 0))
                for key in SVs.keys()])
    return SVs


def privacy_protect(privacy_protection_measure, privacy_protection_level,
                    SV, SV_var=dict()):
    if privacy_protection_measure == 'DP':
        SV = differential_privacy(SV, privacy_protection_level)
    elif privacy_protection_measure == 'QT':
        SV = quantization(SV, privacy_protection_level)
    elif privacy_protection_measure == 'DR':
        for player_id in SV_var.keys():
            SV_var[player_id] = np.var(SV_var[player_id])

        SV = dimension_reduction(SV, privacy_protection_level, SV_var)
    else:
        pass
    return SV
