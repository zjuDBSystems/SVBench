import copy
import math
import time
import numpy as np

from privacy_utils import privacy_protect


class Output():
    def __init__(self,
                 convergence_threshold,
                 cache_size,
                 player_num,
                 privacy_protection_measure,
                 privacy_protection_level,
                 full_check_type):
        self.threshold = convergence_threshold
        self.cache_size = cache_size
        self.SV_cache = []
        self.player_num = player_num
        self.privacy_protection_measure = privacy_protection_measure
        self.privacy_protection_level = privacy_protection_level
        self.full_check_type = full_check_type

    def cache_SV(self, SVs):
        self.SV_cache.append(copy.deepcopy(SVs))
        return len(self.SV_cache) < self.cache_size

    def convergence_check(self,
                          calculated_num,
                          resultant_SVs,
                          SVs_var,
                          start_time,
                          utility_comp_num,
                          avg_time_cost):
        if calculated_num == 0:
            return False
        
        print("Average time cost of a single time of utility computation: ", avg_time_cost)
        
        if self.cache_SV(resultant_SVs):
            return False
        

        self.SVs = resultant_SVs
        self.SVs_var = SVs_var
        self.start_time = start_time
        self.utility_comp_num = utility_comp_num
        self.avg_time_cost = avg_time_cost


        if self.full_check_type == 'coalition'  \
                and calculated_num >= 2**(self.player_num):
            print("Full sampling for all coalitions!")
            return True
        elif self.full_check_type == 'permutation'  \
                and calculated_num >= math.factorial(self.player_num):
            print("Full sampling for all permutations!")
            return True

        count = 0
        sum_ = 0
        for SVs in self.SV_cache[-self.cache_size:-1]:
            for (player_id, SV) in SVs.items():
                count += 1
                latest_SV = self.SV_cache[-1][player_id]
                sum_ += np.abs((latest_SV-SV) /
                               (latest_SV if latest_SV != 0 else 10**(-12)))
        convergence_diff = sum_/(count if count > 0 else 10**(-12))
        print("Current average convergence_diff (count %s): " % count,
              convergence_diff)
        if convergence_diff <= self.threshold:
            print("Convergence checking passed.")
            self.privacy_protect()
            return True
        return False

    def privacy_protect(self):
        # add privacy protection only when specified
        self.SVs = privacy_protect(self.privacy_protection_measure,
                                   self.privacy_protection_level,
                                   self.SVs, self.SVs_var)
        self.result_output()

    def result_output(self):
        print(f"Final Resultant SVs: {self.SVs}")
        print(f"Final Resultant SV_var: {self.SVs_var}")
        print(f"Total runtime: {time.time()-self.start_time}")
        print(f"Total times of utility computation: {self.utility_comp_num}")
        print(
            f"Average time cost of utility computation: {self.avg_time_cost}")
        return  \
            self.SVs, self.SVs_var, self.utility_comp_num, self.avg_time_cost
