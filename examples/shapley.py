# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 15:36:16 2024

@author: admin
"""
import numpy as np
import random, math, time
from Privacy_utils import privacy_protect
#from arguments import args_parser
from scipy.optimize import minimize#, linprog
#from sympy import Symbol, solve, Eq
import datetime,threading, queue, pulp, copy
import warnings
warnings.filterwarnings('ignore')
class Shapley():
    def __init__(self, players, taskUtilityFunc, args, targetedplayer_id=None):
        self.args = args
        
        self.task = args.task
        self.players = players
        self.targetedplayer_id = (targetedplayer_id \
                                   if targetedplayer_id!=None \
                                   else range(len(players)))
        self.utilityComputation = taskUtilityFunc
            
        # utility setting
        #self.utility_func = args.utility_func
        
        # SV settings
        self.SV = dict([(player_id, 0.0) \
                        for player_id in range(len(self.players))])
        self.SV_var = dict([(player_id, []) \
                        for player_id in range(len(self.players))])
        self.SV_cache = []
        self.cache_size = 5
        # SV computation method's components
        self.method = args.method
        self.sampling_strategy = args.sampling_strategy
        self.truncationFlag = args.truncation
        self.gradient_approximation = args.gradient_approximation
        self.testSampleSkip = args.testSampleSkip
        self.privacy_protection_measure = args.privacy_protection_measure
        self.privacy_protection_level = args.privacy_protection_level
        
        # utility information
        # self.utility_records = dict()
        self.taskTotalUtility = 0
        self.emptySet_utility = 0
        
        # runtime records
        self.startTime = 0
        self.num_utility_comp = 0
        self.timeCost_per_utility_comp = []
        
    def clearLogFile(self):
        lines = [line for line in open(self.args.log_file, 'r').readlines()\
                 if 'Done' not in line and len(line.replace(' ','').strip())>0] 
        with open(self.args.log_file, 'w') as log_file:
            log_file.writelines(lines)
        log_file.close()
    
    def truncation(self, truncation, bef_addition):
        if truncation == False:
            return False
        truncation_flag = False
        #print ('check whether to truncate...')
        if np.abs((self.taskTotalUtility - bef_addition)/\
                  (self.taskTotalUtility+10**(-15))) < self.args.truncationThreshold:
            truncation_flag = True
            #print("[Truncation!]", 
            #      self.taskTotalUtility, bef_addition, self.args.truncationThreshold)
        return truncation_flag
     
    def generateRandomPermutation(self, ori_permutation, exclude_list=set()):
        permutation = ori_permutation
        while ",".join(map(str,permutation)) in exclude_list:
            random.shuffle(permutation)
        return permutation
    
    def generateRandomSubset(self, N, q_k, exclude_list=set()):
        k = np.random.choice(range(1,N), p=q_k, size=1)
        selected_players=np.random.choice(
            range(N), int(k), replace=False)
        while ",".join(map(str,sorted(selected_players))) in exclude_list:
            k = np.random.choice(range(0,N+1), size=1) #p=q_k,
            selected_players=np.random.choice(
                range(N), int(k), replace=False)
        return selected_players
    
    def sampling (self, sampling_strategy, iter_time, 
                  current_permutation, scanned_permutations):
        if sampling_strategy == 'antithetic':
            if iter_time%2 == 1:
                permutation = self.generateRandomPermutation(
                    current_permutation, scanned_permutations)
            else:
                # antithetic sampling (also called paired sampling)
                permutation = list(reversed(current_permutation))
        elif sampling_strategy == 'stratified':
            if iter_time%len(current_permutation) == 1:
                permutation = self.generateRandomPermutation(
                    current_permutation, scanned_permutations)
            else:
                # stratified sampling 
                permutation = current_permutation[-1:] + current_permutation[:-1]
        else:
            permutation = self.generateRandomPermutation(
                current_permutation, scanned_permutations)
        return permutation
            
    def convergenceCheck(self, sampling_strategy,
                         scanned_coalition_or_permutation):
        if len(self.SV_cache)< self.cache_size:
            return False
        
        if sampling_strategy == 'stratified' and \
        len(scanned_coalition_or_permutation)%len(self.players) !=0:
            # ensure for each player that the number of permutation 
            # samples in each stratum is the same
            return False
        
        if sampling_strategy == 'antithetic' and \
        len(scanned_coalition_or_permutation)%2 !=0:
            # ensure the correctness for paired sampling
            return False
        
        count = 0
        sum_ = 0
        for SVs in self.SV_cache[:-1]:
            for (player_id, SV) in SVs.items():
                count += 1
                sum_ += np.abs(
                    (self.SV_cache[-1][player_id]-SV)/\
                    (self.SV_cache[-1][player_id] + \
                     (10**(-12) if self.SV_cache[-1][player_id]==0\
                      else 0))
                        )
        if count == 0:
            count += 10**(-12)
        print("Current average convergence_diff (count %s): "%count, 
              sum_/count)
        if sum_/count <= self.args.convergence_threshold: 
            return True
        else:
            return False
        
    def PlayerIteration(self, order, player_id, permutation, iter_time, 
                        truncation, gradient_approximation, testSampleSkip, 
                        convergence_diff, diff_mode='relative'):
        startTime= datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        #lock.acquire()
        subset = permutation[:order]
        # utility before adding the targeted player
        bef_addition, timeCost = self.utilityComputation(
            subset, gradient_approximation, testSampleSkip)
        if timeCost>0:
            self.num_utility_comp += 1
            self.timeCost_per_utility_comp.append(timeCost)
        
        if self.truncation(truncation, bef_addition):
            aft_addition = bef_addition
        else:
            # utility after adding the targeted player
            aft_addition, timeCost = self.utilityComputation(
                list(subset)+[player_id], 
                gradient_approximation, testSampleSkip)
            if timeCost>0:
                self.num_utility_comp += 1
                self.timeCost_per_utility_comp.append(timeCost)
        # update SV
        old_SV  = self.SV[player_id] 
        
        # start updating
        self.SV[player_id] = (iter_time-1)/iter_time * old_SV + \
                              1/iter_time * (aft_addition - bef_addition)  
        self.SV_var[player_id].append(self.SV[player_id])
        
        # compute difference
        diff = np.abs((self.SV[player_id] - old_SV)/\
                      (self.SV[player_id] + \
                       (10**(-12) if self.SV[player_id]==0 else 0)))
        print(('[%s -- %s] Player %s at position %s/%s: utility_bef: %s, '+\
              'utility_aft: %s, SV_bef: %s, SV_aft: %s, diff: %s.')%(
                  startTime, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                  player_id, order, len(permutation), 
                  bef_addition, aft_addition, 
                  old_SV, self.SV[player_id], diff))   
        #lock.release()
        if diff_mode == 'utility_diff':
            convergence_diff[player_id]=aft_addition-bef_addition
        else:    
            convergence_diff[player_id]=diff
    
    def MC(self, sampling_strategy = 'random', 
           truncation=False, gradient_approximation=False, 
           testSampleSkip = False):
    
        # Monte Carlo sampling
        self.SV = dict([(player_id, 0.0) \
                        for player_id in range(len(self.players))])
        convergence = False
        iter_time = 0
        permutation = list(range(len(self.players)))
        scanned_permutations = set()
        scanned_coalitions = set()
        while not convergence:
            iter_time += 1
            permutation = self.sampling(sampling_strategy, iter_time, 
                                        permutation, scanned_permutations)
            scanned_permutations.add(",".join(map(str,permutation)))
            
            print('\n Monte Carlo iteration %s: '%iter_time, permutation)
            # speed up by multiple threads
            convergence_diff = dict()
            threads = []
            for odd_even in ([0,1] if self.args.num_parallelThreads>1 \
                             else [0]):
                for order, player_id in enumerate(permutation):
                    if order % 2 != odd_even and \
                    self.args.num_parallelThreads>1:
                        continue
                    if player_id not in self.targetedplayer_id:
                        # use self.args.convergence_threshold (instead of 0) for avoiding 
                        # too early convergence of computing the targeted player's SV
                        convergence_diff[player_id] = self.args.convergence_threshold
                        continue
                    # parallelThreads may lead to the utility of 
                    # the same player coalition to be computed for two times
                    # since the players in the utility_aft computation with
                    # permutation[:order]+[player_id] are the same as those 
                    # in the utility_bef computation with permutation[:order+1]
                    thread = threading.Thread(
                        target=self.PlayerIteration,
                        args=(order, player_id, permutation, 
                              iter_time, truncation, 
                              gradient_approximation, testSampleSkip,
                              convergence_diff))
                    scanned_coalitions.add(
                        ",".join(map(str,sorted(permutation[:order+1]))))
                    thread.daemon = True
                    thread.start()
                    threads.append(thread)
                    if self.args.num_parallelThreads <=1 or\
                    (len(threads)%self.args.num_parallelThreads==0):
                        for t in threads:
                            t.join()
                        threads = []
                        print('Done %s/%s...'%(order,len(permutation)))
                for t in threads:
                    t.join()
                threads = []
                
            while len(convergence_diff)!=len(self.players):
                time.sleep(3)
             
            # store current SV
            self.SV_cache.append(copy.deepcopy(self.SV))
            if len(self.SV_cache)>self.cache_size:
                self.SV_cache = self.SV_cache[-self.cache_size:]
             
            # print current progress
            print('Monte Carlo iteration %s done '%iter_time)      
            print("Current SV: ", self.SV)
            print("Current runtime: ", time.time()-self.startTime)
            print("Current times of utility computation: ", len(scanned_coalitions)) #, self.num_utility_comp)
            print("Current average time cost of a single time of utility computation: ", 
                  (np.average(self.timeCost_per_utility_comp) \
                   if len(self.timeCost_per_utility_comp)>0 else 0))
            
            
            # convergence check
            if len(scanned_permutations) >= min(self.args.scannedIter_maxNum,
                                                math.factorial(len(self.players))):
                convergence = True
            else:
                convergence = self.convergenceCheck(sampling_strategy,
                                                    scanned_permutations)
            
    def MLE_parallelableThread(self, player_id, subset, bef_addition,
                               truncation=False, gradient_approximation=False, 
                               testSampleSkip = False, results=None):
        if player_id in subset:
            results.put((player_id, 0, 0)) # needed according to the original paper
            return
        if self.truncation(truncation, bef_addition):
            timeCost = 0
            aft_addition = bef_addition
        else:
            # utility after adding the targeted player
            aft_addition, timeCost = self.utilityComputation(
                list(subset)+[player_id], 
                gradient_approximation, testSampleSkip)
        results.put((player_id, aft_addition-bef_addition, timeCost))
        
    
    def MLE(self, sampling_strategy = 'random', 
           truncation=False, gradient_approximation=False, 
           testSampleSkip = False):
        
        # multilinear extension
        # refer to paper: A Multilinear Sampling Algorithm to Estimate Shapley Values
        self.SV = dict([(player_id, 0.0) \
                        for player_id in range(len(self.players))])
            
        #convergence_diff_records = []
        convergence = False
        MLE_interval = 0
        scanned_coalitions = set()
        # scanned_probabilities = set()
        e = np.zeros(len(self.players))
        num_comp = np.zeros(len(self.players))
        M = 2 
        while not convergence:
            MLE_interval += int(len(self.players)/M)
            #if self.sampling_strategy != 'antithetic':
            if self.sampling_strategy == 'antithetic':
                num_iter = int(MLE_interval/2) + 1
                M *= 2
            else:
                num_iter = MLE_interval + 1
                
            print('Multilinear extension iteration (with MLE_interval_%s) start!'%MLE_interval) 
            results = queue.Queue()
            I_mq = None 
            for iter_ in range(num_iter):
                if len(scanned_coalitions) >= 2**len(self.players):
                    break
                for m in range(M):
                    if len(scanned_coalitions) >= 2**len(self.players):
                        break
                    # generate Bernoulli random numbers independently
                    if self.sampling_strategy == 'antithetic' and m%2==1:
                        I_mq = 1-I_mq
                    else:
                        q = iter_/MLE_interval
                        I_mq = np.random.binomial(1, q, 
                                                  size=(len(self.players)))
                        # check whether to be computed in the previous iterations
                        while ",".join(map(str,list(I_mq))) in scanned_coalitions:
                            q = np.random.rand()
                            #while q in scanned_probabilities:
                            #    q = np.random.rand()
                            I_mq = np.random.binomial(1, q, 
                                                      size=(len(self.players)))
                        #scanned_probabilities.add(q)    
                           
                    scanned_coalitions.add(
                        ",".join(map(str,list(I_mq))))
                    
                    # utility before adding the targeted player
                    subset = [player_id \
                              for player_id in range(len(self.players)) \
                                  if I_mq[player_id]==1]
                    bef_addition, timeCost = self.utilityComputation(
                        subset, gradient_approximation, testSampleSkip)
                    results.put((-1, -1, timeCost))
                    
                    # speed up by multiple threads
                    threads = [] 
                    for player_id in range(len(self.players)):
                        # compute under the other q values
                        thread = threading.Thread(
                            target=self.MLE_parallelableThread,
                            args=(player_id, subset, bef_addition,
                                  truncation, gradient_approximation, 
                                  testSampleSkip, results))
                        thread.daemon = True
                        thread.start()
                        threads.append(thread)
                        if self.args.num_parallelThreads <=1 or\
                        (len(threads)%self.args.num_parallelThreads==0):
                            for t in threads:
                                t.join()
                            threads = []
                            print('Done %s/%s  (with MLE_interval_%s iter %s) ...'%(
                                  player_id+1, len(self.players), 
                                  MLE_interval, iter_+1))
                    for t in threads:
                        t.join()
            
            
            while not results.empty():
                (player_id, delta_utility, timeCost) = results.get()
                if player_id!=-1:
                    e[player_id] += delta_utility
                    num_comp[player_id]+=1
                if timeCost>0:
                    self.num_utility_comp +=1
                    self.timeCost_per_utility_comp.append(timeCost) 
            
            # update SV
            self.SV = dict([(player_id, e[player_id] / num_comp[player_id]) \
                            for player_id in range(len(self.players))])
            for player_id in range(len(self.players)): 
                self.SV_var[player_id].append(self.SV[player_id])
            self.SV_cache.append(copy.deepcopy(self.SV))
            if len(self.SV_cache)>self.cache_size:
                self.SV_cache = self.SV_cache[-self.cache_size:]
                
            # print current progress
            self.clearLogFile()
            print('Multilinear extension iteration'+\
                  '(with MLE_interval_%s) done'%MLE_interval)
            print("Current SV: ", self.SV)
            print("Current runtime: ", time.time()-self.startTime)
            print("Current times of utility computation: ", len(scanned_coalitions)) #self.num_utility_comp)
            print("Current average time cost of a single time of utility computation: ", 
                  (np.average(self.timeCost_per_utility_comp) \
                   if len(self.timeCost_per_utility_comp)>0 else 0))
            
            # convergence check
            #if MLE_interval >= self.args.MLE_maxInterval:
            if len(scanned_coalitions) >= 2**len(self.players):#math.factorial(len(self.players)):
                convergence = True
            else:
                convergence = self.convergenceCheck(sampling_strategy,
                                                    scanned_coalitions)
                    
    def GT(self, sampling_strategy = 'random', 
           truncation=False, gradient_approximation=False, 
           testSampleSkip = False):
        
        self.SV = dict([(player_id, 0.0) \
                        for player_id in range(len(self.players))])
        # group testing
        N = len(self.players)
        Z = 2* sum([1/k for k in range(1,N)])
        q_k = [1/Z*(1/k+1/(N-k)) for k in range(1,N)]
        
        convergence = False
        iter_time = 0
        scanned_coalitions = set()
        utilities = []
        #convergence_diff_records = list()
        while not convergence:
            # sampling coalitions
            selected_coalitions = []
            for _ in range(N):
                iter_time += 1
                if sampling_strategy == 'antithetic':
                    # antithetic sampling (also called paired sampling)
                    if iter_time%2 == 1:
                        selected_players = self.generateRandomSubset(
                            N, q_k, scanned_coalitions)
                    else:
                        selected_players = [player_id \
                                            for player_id in range(N) \
                                            if player_id not in selected_players]
                
                elif sampling_strategy == 'stratified':
                    pass
                    # stratified sampling
                    #k = q_k[iter_time%len(self.players)-1]
                    #selected_players = self.generateRandomSubset(
                    #    N, k, scanned_coalitions)
                    
                else:
                    selected_players = self.generateRandomSubset(
                        N, q_k, scanned_coalitions)
                selected_coalitions.append(selected_players)
                scanned_coalitions.add(
                    ",".join(map(str,sorted(selected_players))))
                if len(scanned_coalitions) >= 2**len(self.players):
                    break
                
            # compute utility (speed up by multi-thread)
            results = queue.Queue()
            threads = []
            for order, selected_players in enumerate(selected_coalitions):
                                    
                thread = threading.Thread(
                    target=self.RE_parallelableThread,
                    args=(order, selected_players, truncation, 
                          gradient_approximation, testSampleSkip,
                          results))
                thread.daemon = True
                thread.start()
                threads.append(thread)
                if self.args.num_parallelThreads <=1 or\
                (len(threads)%self.args.num_parallelThreads==0):
                    for t in threads:
                        t.join()
                    threads = []
                    print('Done %s/%s...'%(
                           order+1, len(selected_coalitions)))
            while results._qsize()!=len(selected_coalitions):
                time.sleep(3)
            
            while not results.empty():
                order, (value, timeCost) = results.get()
                utilities.append(
                    ([int(player_id in selected_coalitions[order]) \
                      for player_id in range(N)], 
                     value)
                    )
                if timeCost>0:
                    self.num_utility_comp += 1
                    self.timeCost_per_utility_comp.append(timeCost)
                            
            print('\n Group testing iteration %s with (k=%s): '%(
                iter_time, 
                [len(selected_players) \
                 for selected_players in selected_coalitions])
                    )
            delta_utility = np.zeros((N,N))
            for i in range(N):
                for j in range(i+1,N):
                    delta_utility[i,j]=Z/iter_time *sum(
                        [utility*(beta[i]-beta[j]) \
                         for (beta, utility) in utilities])
                        
                    delta_utility[j,i] = - delta_utility[i,j]
            
            # find SV by solving the feasibility problem
            MyProbLP = pulp.LpProblem("LPProbDemo1", sense=pulp.LpMaximize)
            sv = [pulp.LpVariable('%s'%player_id, cat='Continuous') \
                  for player_id in range(N)]
            MyProbLP += sum(sv)
            for i in range(N):
                for j in range(i+1,N):
                    MyProbLP += (sv[i]-sv[j]-delta_utility[i,j] <= self.args.GT_epsilon/2/np.sqrt(N))
                    MyProbLP += (sv[i]-sv[j]-delta_utility[i,j] >= -self.args.GT_epsilon/2/np.sqrt(N))
                    
            
            print("feasible problem solving ...")
            MyProbLP += (sum(sv) >= self.taskTotalUtility)
            MyProbLP += (sum(sv) <= self.taskTotalUtility)
            MyProbLP.solve()
            #status： “Not Solved”, “Infeasible”, “Unbounded”, “Undefined” or “Optimal”
            print("Status:", pulp.LpStatus[MyProbLP.status]) 
            result = dict()
            for v in MyProbLP.variables():
                #print(v.name, "=", v.varValue)  
                result[int(v.name)] = v.varValue
            print('One solution for reference:', v.name, "=", v.varValue)  
            print("F(x) = ", pulp.value(MyProbLP.objective), self.taskTotalUtility)  #输出最优解的目标函数值    
            
            # update SV
            self.SV = dict([(player_id, result[player_id]) \
                            for player_id in range(len(self.players))])
            for player_id in range(len(self.players)): 
                self.SV_var[player_id].append(self.SV[player_id])
            self.SV_cache.append(copy.deepcopy(self.SV))
            if len(self.SV_cache)>self.cache_size:
                self.SV_cache = self.SV_cache[-self.cache_size:]
                
            # print current progress
            print('Group testing iteration %s done (len utilities: %s)!'%(
                iter_time, len(utilities)))
            print("Current SV: ", self.SV)
            print("Current runtime: ", time.time()-self.startTime)
            print("Current times of utility computation: ", len(scanned_coalitions)) #, self.num_utility_comp)
            print("Current average time cost of a single time of utility computation: ", 
                  (np.average(self.timeCost_per_utility_comp) \
                   if len(self.timeCost_per_utility_comp)>0 else 0))
            
            # convergence check
            if len(scanned_coalitions) >= 2**len(self.players):#math.factorial(len(self.players)):
                convergence = True
            else:
                convergence = self.convergenceCheck(sampling_strategy,
                                                    scanned_coalitions)
            
    def CP(self, sampling_strategy = 'random', 
           truncation=False, gradient_approximation=False, 
           testSampleSkip = False):
        
        self.SV = dict([(player_id, 0.0) \
                        for player_id in range(len(self.players))])
        # compressive permutation sampling
        # sample a Bernoulli matirc A
        N = len(self.players)
        self.args.num_measurement = int(N/2)
        A = np.random.binomial(1, 0.5, 
                               size=(self.args.num_measurement, N))
        A = 1/np.sqrt(self.args.num_measurement)*(2*A -1)
        y = dict([(m,[]) for m in range(self.args.num_measurement)])
        
        convergence = False
        iter_time = 0
        permutation = list(range(len(self.players)))
        scanned_permutations = set()
        scanned_coalitions = set()
        while not convergence:
            iter_time += 1
            permutation = self.sampling(sampling_strategy, iter_time, 
                                        permutation, scanned_permutations)
            scanned_permutations.add(",".join(map(str,permutation)))
            
            print('\n Compressive permutation sampling iteration %s: '%iter_time, 
                  permutation)
            # speed up by multiple threads
            phi_t = dict()
            threads = []
            for odd_even in ([0,1] if self.args.num_parallelThreads>1 \
                             else [0]):
                for order, player_id in enumerate(permutation):
                    if order % 2 != odd_even and \
                    self.args.num_parallelThreads>1:
                        continue
                    thread = threading.Thread(
                        target=self.PlayerIteration,
                        args=(order, player_id, permutation, 
                              iter_time, truncation, 
                              gradient_approximation, testSampleSkip,
                              phi_t, 'utility_diff'))
                    scanned_coalitions.add(
                        ",".join(map(str,sorted(permutation[:order+1]))))
                    thread.daemon = True
                    thread.start()
                    threads.append(thread)
                    
                    if self.args.num_parallelThreads <=1 or\
                    (len(threads)%self.args.num_parallelThreads==0):
                        for t in threads:
                            t.join()
                        threads = []
                        print('Done %s/%s...'%(order+1,len(permutation)))
                for t in threads:
                    t.join()
                threads = []
                
            while len(phi_t)!=len(self.players):
                time.sleep(3)
            
            for m in range(self.args.num_measurement):
                y[m].append(sum([
                    A[m, player_id]*phi \
                    for (player_id, phi) in phi_t.items()]))
            
                
            y_mean=np.zeros(self.args.num_measurement)
            for m in range(len(y)):
                y_mean[m] = 1/iter_time * sum(y[m][:iter_time]) #sum(y[m,:iter_time])
            sv_mean = self.taskTotalUtility/len(self.players)
            fun = lambda sv_variance : np.linalg.norm(sv_variance, ord=1) 
             
            cons = (
                {'type': 'ineq', 'fun': 
                 lambda sv_variance: 
                     self.args.CP_epsilon-np.linalg.norm(
                         A.dot(sv_mean+sv_variance)-y_mean, ord=2)}, # self.args.CP_epsilon-x[0]>0 # inequality means that it is to be non-negative
                    )
            res = minimize(fun, np.zeros(N), 
                           method='SLSQP', constraints=cons)
            sv_variance = res.x
                
            # update SV
            self.SV = dict([(player_id, sv_mean+sv_variance[player_id]) \
                            for player_id in range(len(self.players))])
            for player_id in range(len(self.players)): 
                self.SV_var[player_id].append(self.SV[player_id])
            self.SV_cache.append(copy.deepcopy(self.SV))
            if len(self.SV_cache)>self.cache_size:
                self.SV_cache = self.SV_cache[-self.cache_size:]
                
            # print current progress
            print('CPS iteration %s done (len y: %s)!'%(iter_time, len(y[0])))
            print("Current SV: ", self.SV)
            print("Current runtime: ", time.time()-self.startTime)
            print("Current times of utility computation: ", len(scanned_coalitions)) #, self.num_utility_comp)
            print("Current average time cost of a single time of utility computation: ", 
                  (np.average(self.timeCost_per_utility_comp) \
                   if len(self.timeCost_per_utility_comp)>0 else 0))
            
            # convergence check
            if len(scanned_permutations) >= math.factorial(len(self.players)):
                convergence = True
            else:
                convergence = self.convergenceCheck(sampling_strategy,
                                                    scanned_permutations)
        
    def RE_parallelableThread(self, order, selected_players, 
                              truncation=False, gradient_approximation=False, 
                              testSampleSkip = False, results=None):
        results.put((order,
            self.utilityComputation(selected_players, 
                                    gradient_approximation, testSampleSkip)))
        
    
    def RE(self, sampling_strategy = 'random', 
           truncation=False, gradient_approximation=False, 
           testSampleSkip = False):
        # regression-based
        self.SV = dict([(player_id, 0.0) \
                        for player_id in range(len(self.players))])
        
        d = len(self.players)
        A=np.zeros((d,d))
        for i in range(d):
            for j in range(d):
                if i==j:
                    A[i,j] = sum([1/d/(d-k) for k in range(1,d)]) / \
                             sum([1/k/(d-k) for k in range(1,d)])
                    
                else:
                    A[i,j] = 1/d/(d-1)*\
                        sum([(k-1)/(d-k) for k in range(2,d)]) / \
                        sum([1/k/(d-k) for k in range(1,d)])
        z = np.array([0 for _ in range(d)]).reshape(1,-1)
        utilities = {0: self.emptySet_utility}
        permutation = list(range(len(self.players)))
        scanned_permutations = set()
        scanned_coalitions = set()
        convergence = False
        iter_time=0
        while not convergence:
            iter_time += 1
            print('\n Regression iteration %s start! '%iter_time)      
            
            permutation = self.sampling(sampling_strategy, iter_time, 
                                        permutation, scanned_permutations)
            scanned_permutations.add(",".join(map(str,permutation)))
            # speed up by multiple threads
            z_i = []
            results = queue.Queue()
            threads = []
            for order, _ in enumerate(permutation):
                if ",".join(map(str,sorted(permutation[:order+1]))) in scanned_coalitions:
                    continue
                
                z_i.append([int(player_id in permutation[:order+1]) \
                            for player_id in range(d)])
                    
                thread = threading.Thread(
                    target=self.RE_parallelableThread,
                    args=(len(z_i)-1, permutation[:order+1], truncation, 
                          gradient_approximation, testSampleSkip,
                          results))
                scanned_coalitions.add(
                    ",".join(map(str,sorted(permutation[:order+1]))))
                thread.daemon = True
                thread.start()
                threads.append(thread)
                if self.args.num_parallelThreads <=1 or\
                (len(threads)%self.args.num_parallelThreads==0):
                    for t in threads:
                        t.join()
                    threads = []
                    print('Done %s/%s...'%(order+1, len(permutation)))
            while results._qsize()!=len(z_i):
                time.sleep(3)
            
            while not results.empty():
                order, (value, timeCost) = results.get()
                utilities[len(z)+order] = value
                if timeCost>0:
                    self.num_utility_comp += 1
                    self.timeCost_per_utility_comp.append(timeCost)
                

            # regression computation
            z = (np.concatenate((z, np.array(z_i))) if len(z_i)>0 else z)
            b = np.zeros((d,1))
            E_Z = 0.5*np.ones((d,1))
            for (sample_id, z_i) in enumerate(z):
                #A += 1/len(z) * z_i.reshape((-1,1)).dot(z.reshape(1,-1))
                #b += 1/len(z) * (utilities[sample_id]-zero_utility)*z_i.reshape((-1,1))
                b += (z_i.reshape(-1,1) * utilities[sample_id] - \
                      E_Z * self.emptySet_utility) / len(z)
            inv_A=np.linalg.inv(A)
            ones = np.ones((d,1))
            beta = np.linalg.inv(A).dot(
                b-ones*(
                    (ones.T.dot(inv_A).dot(b)-self.taskTotalUtility+self.emptySet_utility)/\
                    ones.T.dot(inv_A).dot(ones)
                    )).reshape(-1)
                
            # update SV 
            self.SV = dict([(player_id, beta[player_id]) \
                            for player_id in range(len(self.players))])
            for player_id in range(len(self.players)): 
                self.SV_var[player_id].append(self.SV[player_id])
            self.SV_cache.append(copy.deepcopy(self.SV))
            if len(self.SV_cache)>self.cache_size:
                self.SV_cache = self.SV_cache[-self.cache_size:]
                
            # print current progress
            print('Regression iteration %s done (len z:%s)'%(
                iter_time, len(z)))        
            print("Current SV: ", self.SV)
            print("Current runtime: ", time.time()-self.startTime)
            print("Current times of utility computation: ", len(scanned_coalitions)) #, self.num_utility_comp)
            print("Current average time cost of a single time of utility computation: ", 
                  (np.average(self.timeCost_per_utility_comp) \
                   if len(self.timeCost_per_utility_comp)>0 else 0))
            
            # convergence check
            if len(scanned_permutations) >= math.factorial(len(self.players)):
                convergence = True
            else:
                convergence = self.convergenceCheck(sampling_strategy,
                                                    scanned_permutations)
    
            
    def computeAllSubsetUtility(self):
        N = len(self.players)
        exclude_list = []
        for k in range(2, N+1):
            num_itered_subsets = len(exclude_list)
            finish_thresholdNumber = math.factorial(N) // (math.factorial(k) * math.factorial(N - k))
            print('Utility computation for subsets with %s players...'%k)
            while True:
                selected_players=np.random.choice(
                    range(N), int(k), replace=False)
                while ",".join(map(str,sorted(selected_players))) in exclude_list:
                    selected_players=np.random.choice(
                        range(N), int(k), replace=False)
                exclude_list.append(",".join(map(str,sorted(selected_players))))
        
                self.utilityComputation(selected_players, 
                                        self.gradient_approximation, 
                                        self.testSampleSkip)
                print('Progress %s/%s with players %s...'%(
                    len(exclude_list)-num_itered_subsets, 
                    finish_thresholdNumber, exclude_list[-1]))
                if len(exclude_list) - num_itered_subsets >= finish_thresholdNumber:
                    break
                
    def problemScale_statistics(self):
        print('【Problem Scale of SV Exact Computation】')
        print('Total number of players: ', len(self.players))        
        print('(coalition sampling) Total number of utility computations:',
              '%e'%(2*len(self.players) * 2**(len(self.players)-1)) )
        print('(permutation sampling) Total number of utility computations:',
              '%e'%(2*len(self.players) * math.factorial(len(self.players))) )
        self.taskTotalUtility, _ = self.utilityComputation(
            range(len(self.players)), 
            self.gradient_approximation, self.testSampleSkip)
        print('The task\'s total utility: ', self.taskTotalUtility)
        
    def CalSV(self):
        # print problem scale and the task's overall utility
        self.problemScale_statistics()
        
        # reset rumtime records
        self.num_utility_comp = 0
        self.timeCost_per_utility_comp = []
        
        if self.method in ['EXACT', 'MC']:
            base_compFunc = self.MC
            if self.method == 'EXACT':
                self.args.convergence_threshold=0
        elif self.method == 'RE':
            base_compFunc = self.RE
        elif self.method == 'MLE':
            base_compFunc = self.MLE
        elif self.method == 'GT':
            base_compFunc = self.GT
        elif self.method == 'CP':
            base_compFunc = self.CP
        else:
            print("Unknown computation method!!")
            
            
        if self.method == 'RE':
            self.emptySet_utility, _ = self.utilityComputation(
                [], self.gradient_approximation, self.testSampleSkip)
            print('The task\'s emptySet utility: ', self.emptySet_utility)
            
        self.startTime = time.time()
        base_compFunc(sampling_strategy = self.sampling_strategy, 
                      truncation=self.truncationFlag, 
                      gradient_approximation=self.gradient_approximation, 
                      testSampleSkip = self.testSampleSkip)
        # add privacy protection only when specified
        self.SV = privacy_protect(self.privacy_protection_measure, 
                                  self.privacy_protection_level,
                                  self.SV, self.SV_var)
        print("Final Resultant SVs: ", self.SV) 
        print("Final Resultant SV_var: ", self.SV_var) 
        print("Total runtime: ", time.time()-self.startTime)
        print("Total times of utility computation: ", self.num_utility_comp)
        print("Average time cost of a single time of utility computation: ", 
              (np.average(self.timeCost_per_utility_comp) \
               if len(self.timeCost_per_utility_comp)>0 else 0))
        
