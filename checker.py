# -*- coding: utf-8 -*-
# For paper: A Comprehensive Study of Shapley Value in Data Analytics
import numpy as np

class Checker():
    def __init__(self, player_num, cache_size, threshold):
        self.player_num = player_num
        self.cache_size = cache_size
        self.threshold = threshold

    def convergence_check(self, SVs_var, utility_comp_times, iter_count, checker_mode):
        # SV_var_mode
        if len(SVs_var) <= 0:
            return False
        if len(SVs_var[0]) < self.cache_size:
            return False

        count = 0
        sum_ = 0
        for (player_id, _) in SVs_var.items():
            latest_SV = SVs_var[player_id][-1]
            for ridx in range(2, self.cache_size+1):
                count += 1
                sum_ += np.abs((latest_SV-SVs_var[player_id][-ridx]) /
                               (latest_SV if latest_SV != 0 else 10**(-12)))

        convergence_diff = sum_/(count if count > 0 else 10**(-12))
        print("Current average convergence_diff (count %s): " % count,
              convergence_diff)
        if checker_mode == 'comp_count':
            num_players = len(SVs_var)
            if utility_comp_times >= 2**num_players and \
                iter_count >= 2**num_players * num_players:
                print(f"[{checker_mode}] Convergence checking passed.")
                return True
        else:
            if convergence_diff <= self.threshold:
                print(f"[{checker_mode}] Convergence checking passed.")
                return True
        return False
