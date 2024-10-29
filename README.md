# SV4DA

SV4DA is a free, powerful library providing a series of accurate or approximate calculation methods of Shapley Value(SV) for data analysis(DA). It provides high-level APIs in Python to calculate Shapley Value for different task targets.

A cooperative game is composed of a player set and a utility function that defines the utility of each coalition (i.e., a subset of the player set). Shapley Value (SV) is a solution concept in cooperative game theory, designed for fairly allocate the total utility generated by the collective efforts of all players within a game. SV has already been widely applied in various DA tasks modeled as cooperative games.

Modeling DA tasks as cooperative games is the fundamental step for applying SV to those tasks. The key is to define the player and the utility according to the underlying purpose of applying SV. When users try to use SV to deal with DA tasks, they only need to define the player and provide corresponding utility computation methods and related parameters of the task to complete the call to SV4DA and obtain the results of SV calculation.

[![LICENSE](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://github.com/apecloud/foxlake/blob/main/LICENSE) [![GitHub CodeQL Advanced](https://github.com/DDDDDstar/SV4DA/actions/workflows/codeql.yml/badge.svg)](https://github.com/DDDDDstar/SV4DA/actions/workflows/codeql.yml) [![Contributors](https://img.shields.io/github/contributors/DDDDDstar/SV4DA?color=3ba272)](https://github.com/DDDDDstar/SV4DA/graphs/contributors) [![Language](https://img.shields.io/badge/Language-Python-blue.svg)](https://www.python.org/)

## Get Started

To use SV4DA, you need the following steps:

1. Download all the codes to your project directory;
2. Add `from api import Sv_calc` statement in the Python file where you need to call the SV4DA computing function;
3. Call the `Sv_calc` function with the relevant parameters of the task, and the results of SV calculation will be returned.

More detailed descriptions of the `Sv_calc` function including what relevant parameters and the format of the return results will be expanded in subsequent paragraphs.

## Parameters

### Necessary Parameters

For the `SV_valc` function call, there are three necessary parameters: `player_num`, `taskUtilityFunc` and `method`.

|    Parameter    |         Options or Scope          | Introduction                                                 | Default |
| :-------------: | :-------------------------------: | ------------------------------------------------------------ | :-----: |
|   player_num    |                 -                 | The number of player.                                        |    -    |
| taskUtilityFunc |                 -                 | The function interface used to calculate the utility of each coalition. A list-type parameter should be accepted, which represents the list of player numbers of the target coalition to be calculated its utility. |    -    |
|     method      | `exact` `MC` `RE` `MLE` `GT` `CP` | The method used to calculate SV. Six methods of exact calculation(`exact`), Monte Carlo random sampling(`MC`), regression-based SV formulation(`RE`), multilinear-extension-based SV formulation(`MLE`), group-testing-based SV formulation(`GT`) and compressive-permutation-based SV formulation(`CP`) are provided. |  `MC`   |

### Optional Parameters

| Parameter |          Options or Scope          | Introduction                                                 | Default | Applicable Methods |
| :------: | :------: | -------- | :------: | :------: |
|     sampling_strategy      | `random` `antithetic` `stratified` | Three sampling strategies of random sampling, antithetic sampling and stratified sampling are provided to reduce the approximate error. | `random` | `MC` `RE` `MLE` `GT` `CP` |
| truncation | `True` `False` | Whether to truncate the unnecessary calculations of some marginal contributions in runtime of approximating SV. | `False` | `MC` `RE` `MLE` `CP` |
| privacy_protection_measure | `DP` `QT` `DR` | The measure to protect privacy. Three methods of differential privacy(`DP`), quantization(`QT`) and dimension reduction(`DR`) are provided. | `None` | `exact` `MC` `RE` `MLE` `GT` `CP` |
| privacy_protection_level | 0 ~ 1 | The intensity level of providing privacy protection measures, 1 for the highest intensity protection, 0 for non-protection. | 0.0 | `exact` `MC` `RE` `MLE` `GT` `CP` |
| num_parallelThreads | - | The number of threads used for parallel computing. | 1 | `MC` `MLE` `CP` `RE` |
| scannedIter_maxNum | - | Maximum number of samplings in MC method. | inf | `MC` |
| MLE_maxInterval | - | The maximum interval that can be reached in MLE to limit the running time of the algorithm. | 10000 | `MLE` |
| GT_epsilon | - | The epsilon used in GT method. | 0.00001 | `GT` |
| CP_epsilon | - | The epsilon used in CP method. | 0.00001 | `CP` |
| num_measurement | - | The number of measurement. | 10 | `CP` |

## Usage Examples

Specific usage examples of SV4DA can be found in [README.md](./examples/README.md).