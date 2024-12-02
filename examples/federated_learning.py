# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 16:23:58 2024

@author: admin
"""

from shapley import Shapley
from arguments import args_parser
import numpy as np
import torch, copy, time, sys, threading, json, portalocker, os
torch.backends.cudnn.enabled = False
from models.Nets import  CNN, CNNCifar
from ML_utils import DNNTrain, DNNTest, find_free_gpu
#from torch.utils.data import DataLoader


class Task():
    def __init__(self, args):
        self.taskTerminated = False
        self.args = args
        free_gpu = find_free_gpu() #int(all_gpus[0])
        self.device = torch.device(
            'cuda:{}'.format(free_gpu) \
                if torch.cuda.is_available() else 'cpu'
                )
            
        # FL task settings
        self.task = args.task
        self.model = None
        self.model_name = args.model_name
        self.Tst = torch.load(
            'data/%s%s/test.pt'%(args.dataset, args.data_allocation), 
            #map_location=self.device
            )
        self.stored_gradients=dict()
        
        # player setting
        self.players = [
            torch.load('data/%s%s/train%s.pt'%(
                args.dataset, args.data_allocation, no), 
                #map_location=self.device
                ) \
            for no in range(self.args.num_clients)]
            
        model_paths = ['models/FL_%s_R%sC%s.pt'%(self.args.dataset,
                                                 ridx, trainer_idx) \
                       for ridx in range(self.args.maxRound) \
                           for trainer_idx in range(len(self.players))]
        if False not in [os.path.exists(model_path)\
                         for model_path in model_paths] and\
        self.args.modelRetrain == False:
            pass # all needed models have been trained 
        
        else:
            # model initialize and training 
            # (can be expedited by the above ML speedup functions)
            # FedAvg
            global_model = self.modelInitiation()
            loss_func = torch.nn.CrossEntropyLoss()
            for ridx in range(self.args.maxRound):
                localUpdates = dict()
                p_k = dict()
                start_time = time.time()
                for player_idx in range(len(self.players)):
                    # local model training
                    pstar_time = time.time()
                    localUpdates[player_idx] =  DNNTrain(
                        global_model, self.players[player_idx], 
                        self.args.local_ep, self.args.local_bs,
                        self.args.lr*(self.args.decay_rate**ridx), loss_func,
                        momentum=self.args.momentum, 
                        weight_decay=self.args.weight_decay, 
                        max_norm=self.args.max_norm, 
                        #device=self.device
                        ).state_dict()
                    p_k [player_idx] = len(self.players[player_idx])
                    torch.save(localUpdates[player_idx],
                               'models/FL_%s_R%sC%s.pt'%(self.args.dataset, 
                                                         ridx, player_idx)
                               )
                    torch.cuda.empty_cache()
                    print('Round %s player %s time cost:'%(ridx, player_idx),
                          time.time()-pstar_time)
                    sys.stdout.flush()
                # aggregation
                AggResults = self.WeightedAvg(localUpdates, p_k)  
                global_model.load_state_dict(AggResults)
                print('Round %s time cost: ', time.time()-start_time)
                
        self.player_data_size = [len(self.players[no])\
                                 for no in range(self.args.num_clients)]
        self.players = [
            dict([
                (ridx, 
                 torch.load('models/FL_%s_R%sC%s.pt'%(self.args.dataset, 
                                                      ridx, no), 
                            map_location=self.device)) \
                    for ridx in range(self.args.maxRound)]) \
                for no in range(self.args.num_clients)]
            
        # utility setting
        self.writingUtility = False
        self.utility_file_path = 'logs/UtilityRecord_%s_%s_%s_GA-%s%s.json'%(
                self.args.task, self.args.dataset, 
                self.args.manual_seed, self.args.gradient_approximation,
                (self.args.testSampleSkip if self.args.testSampleSkip else ""))
        self.utility_records = self.readUtilityRecords()
        if self.args.gradient_approximation:
            self.record_interval = 2*len(self.players)
        else:
            self.record_interval = 2 
        self.record_count = len(self.utility_records)//self.record_interval
        self.skippableTestSample = dict([(player_id, set()) \
                        for player_id in range(len(self.players))])
        print('len(utility): ', len(self.utility_records),
              self.utility_file_path)
        
    
    def readUtilityRecords(self, emptySet_utility = None):
        if type(emptySet_utility)==type(None):
            emptySet_utility =  {str([]):(0,0)}
            
        if not os.path.exists(self.utility_file_path):
            return emptySet_utility
        
        if not os.path.exists(self.utility_file_path):
            os.mknod(self.utility_file_path)
        with open(self.utility_file_path, 'r',
                  encoding='utf-8') as file:
            portalocker.lock(file, 
                             portalocker.LOCK_SH)#lock
            utilities = eval(file.read().strip())
            portalocker.unlock(file)
            
        return utilities
    
    def writeUtilityRecord(self):
        #print(self.utility_file_path,len(self.utility_records),self.record_count)
        if not self.writingUtility and\
        len(self.utility_records)-1 > (self.record_count+1)*self.record_interval:
            
            self.writingUtility = True
            create = False
            if not os.path.exists(self.utility_file_path):
                os.mknod(self.utility_file_path)
                create = True
            with portalocker.Lock(self.utility_file_path,
                                  mode="r+", 
                                  timeout=300)  as file:
                # copy is necessary under multi-thread environment
                ur = self.utility_records.copy() 
                if not create:
                    ur.update(eval(file.read().strip()))
                file.seek(0)
                file.write(str(ur))
                file.seek(0)
                self.utility_records = eval(file.read().strip())
                
            
            #print('\n', len(self.utility_records))
            self.record_count = len(self.utility_records)//self.record_interval
            
            self.writingUtility = False
            
            
    def modelInitiation(self):
        global_model = None
        if self.args.dataset == 'cifar':
            global_model = CNNCifar(args=self.args)
        elif self.args.dataset == 'mnist': 
            global_model = CNN(args=self.args)
        else:
            print('Model for given dataset has not been designed!')
            exit()
        return global_model
    
    def utilityComputation (self, player_idxs, 
                            gradient_approximation=False,  
                            testSampleSkip = False):
        startTime = time.time()
        utility_record_idx = str(sorted(player_idxs))
        if utility_record_idx in self.utility_records:
        #and not gradient_approximation:
            #print('Read FL compute utility with players:', utility_record_idx)
            return self.utility_records[utility_record_idx][0], 0#time.time()-startTime
        #print('FL compute utility with players:', utility_record_idx,
        #      'utility_records len:', len(self.utility_records))
        
        # model initialize and training 
        # (maybe expedited by some ML speedup functions)
        if gradient_approximation:
            ridx = max(self.stored_gradients.keys())
            if len(player_idxs) <1:
                AggResults = self.stored_gradients[ridx][2]
            else:
                localUpdates = dict()
                p_k = dict()
                for player_idx in player_idxs:
                    localUpdates[player_idx] = self.stored_gradients[ridx][0][player_idx]
                    p_k[player_idx] = self.stored_gradients[ridx][1][player_idx]
                    
                # aggregation
                AggResults = self.WeightedAvg(localUpdates, p_k)  
                
            global_model = self.modelInitiation()
            global_model.load_state_dict(AggResults)
            if testSampleSkip and len(player_idxs)>0:
                skippableTestSampleIdxs = set(range(len(self.Tst)))
                for player_idx in player_idxs:
                    skippableTestSampleIdxs = skippableTestSampleIdxs & self.skippableTestSample[player_idx]
                complete_idx = set(range(len(self.Tst)))
                testData = copy.deepcopy(self.Tst)
                testData.idxs = list(complete_idx - skippableTestSampleIdxs)
            else:
                testData = self.Tst
            utility = DNNTest(global_model, testData,
                              test_bs= self.args.test_bs,
                              metric = self.args.test_metric)
            if testSampleSkip and len(player_idxs)>0:
                del testData
            del global_model
            torch.cuda.empty_cache()
            endTime = time.time()
            self.utility_records[utility_record_idx] = (utility, endTime-startTime)
            self.writeUtilityRecord()
            return self.utility_records[utility_record_idx]
        '''
        if len(player_idxs) <= 0:
            global_model = self.modelInitiation()
            utility = DNNTest(global_model, self.Tst,
                              test_bs= self.args.test_bs,
                              metric = self.args.test_metric)
            del global_model
            torch.cuda.empty_cache()
            endTime = time.time()
            self.utility_records[utility_record_idx] = (utility, endTime-startTime)
            return self.utility_records[utility_record_idx]
        '''
        
        # model initialize and training 
        # (can be expedited by the above ML speedup functions)
        # FedAvg
        global_model = self.modelInitiation()
        #loss_func = torch.nn.CrossEntropyLoss()
        for ridx in range(self.args.maxRound):
            localUpdates = dict()
            p_k = dict()
            for player_idx in player_idxs:
                # local model training
                #ldr_train = DataLoader(self.players[player_idx], 
                #                       batch_size=self.args.local_bs, 
                #                       shuffle=True)
                #local_model = copy.deepcopy(global_model).to(self.device)
                localUpdates[player_idx] =  self.players[player_idx][ridx]
                p_k [player_idx] = self.player_data_size[player_idx]
                
            # aggregation
            AggResults = self.WeightedAvg(localUpdates, p_k)  
            global_model.load_state_dict(AggResults)
            
        if testSampleSkip and len(player_idxs)>0:
            skippableTestSampleIdxs = set(range(len(self.Tst)))
            for player_idx in player_idxs:
                skippableTestSampleIdxs = skippableTestSampleIdxs & self.skippableTestSample[player_idx]
            complete_idx = set(range(len(self.Tst)))
            testData = copy.deepcopy(self.Tst)
            testData.idxs = list(complete_idx - skippableTestSampleIdxs)
        else:
            testData = self.Tst
        utility = DNNTest(global_model, testData,
                          test_bs= self.args.test_bs,
                          metric = self.args.test_metric)
        if testSampleSkip and len(player_idxs)>0:
            del testData
        del global_model, localUpdates
        torch.cuda.empty_cache()
        endTime = time.time()
        self.utility_records[utility_record_idx] = (utility, endTime-startTime)
        self.writeUtilityRecord()
        
        return self.utility_records[utility_record_idx]
        
    
    def WeightedAvg(self, w_locals, p_k):
        parameter_keys = list(w_locals.values())[0].keys()
        idx_keys = list(w_locals.keys())
        net_glob = dict([(k,None) for k in parameter_keys])
        
        sum_pk = sum([p_k[idx] for idx in idx_keys])
        
        for k in parameter_keys:
            #flag = False
            for idx in idx_keys:
                if type(net_glob[k]) == type(None):
                    net_glob[k] = p_k[idx]/sum_pk * w_locals[idx][k].clone()
                    #net_glob[k] = copy.deepcopy(p_k[idx]/sum_pk * w_locals[idx][k])
                    #flag = True
                else:
                    net_glob[k] += p_k[idx]/sum_pk * w_locals[idx][k].to(net_glob[k].device).clone()
                    
        return net_glob
    
    def printFlush(self):
        while not self.taskTerminated:
            sys.stdout.flush()
            time.sleep(5)
            
    def run(self):
        thread = threading.Thread(target=task.printFlush)
        thread.daemon = True
        thread.start()
        
        if not self.args.gradient_approximation:
            # self.preExp_statistic()
            # reinitialize!!!
            self.utility_records = self.readUtilityRecords()
            SVtask = Shapley(players = self.players, 
                             taskUtilityFunc=self.utilityComputation, 
                             args = self.args)
            SVtask.CalSV()
        else:
            # model initialize and training 
            global_model = self.modelInitiation()
            round_SV = dict()
            #loss_func = torch.nn.CrossEntropyLoss()
            dict_utilityComputationTimeCost = dict()
            for ridx in range(self.args.maxRound):
                localUpdates = dict()
                p_k = dict()
                for player_idx in range(len(self.players)):
                    #ldr_train = DataLoader(self.players[player_idx], 
                    #                       batch_size=self.args.local_bs, 
                    #                       shuffle=True)
                    #local_model = copy.deepcopy(global_model).to(self.device)
                    local_model = self.players[player_idx][ridx]
                    localUpdates[player_idx] = local_model
                    p_k [player_idx] = self.player_data_size[player_idx]
                    torch.cuda.empty_cache()
                    print('Round %s player %s done!'%(ridx, player_idx))
                    
                self.stored_gradients[ridx] = (localUpdates, p_k,
                                               global_model.state_dict())
                if self.args.testSampleSkip:
                    self.skippableTestSample = dict([(player_id, set()) \
                                    for player_id in range(len(self.players))])
                
                    for player_idx, local_model in localUpdates.items():
                        tmp_model = self.modelInitiation()
                        tmp_model.load_state_dict(local_model)
                        DNNTest(tmp_model, self.Tst,
                                test_bs= self.args.test_bs,
                                metric = self.args.test_metric,
                                #device = self.device,
                                recordSkippableSample=(self.skippableTestSample,
                                                       player_idx))
                
                # compute SV of current round
                if self.args.truncation:
                    emptySet_utility = DNNTest(
                        global_model, self.Tst,
                        test_bs= self.args.test_bs,
                        metric = self.args.test_metric)
                    # aggregation
                    AggResults = self.WeightedAvg(localUpdates, p_k)  
                    global_model.load_state_dict(AggResults)
                    current_taskTotalUtility = DNNTest(
                        global_model, self.Tst,
                        test_bs= self.args.test_bs,
                        metric = self.args.test_metric)
                    
                    if np.abs((current_taskTotalUtility - emptySet_utility)/\
                              (current_taskTotalUtility+10**(-15))) < self.args.truncationThreshold:
                        print('Truncate the entire round!!!')
                        round_SV[ridx] = dict([(player_id, 0.0) \
                                               for player_id in range(len(self.players))])
                        continue
                    
                dict_utilityComputationTimeCost[ridx] = self.preExp_statistic()
                # reinitialize!!!
                self.utility_file_path = self.utility_file_path.replace(
                                            '.json','-R%s.json'%ridx)
                self.utility_records = self.readUtilityRecords(
                                            {str([]):self.utility_records[str([])]}
                                            )
                self.record_count = len(self.utility_records)//self.record_interval
                print(self.utility_file_path,
                      'len(self.utility_records):', 
                      len(self.utility_records))
                
                SVtask = Shapley(players = self.players, 
                                 taskUtilityFunc=self.utilityComputation, 
                                 args = self.args)
                SVtask.CalSV()
                round_SV[ridx] = SVtask.SV
                
                # aggregation
                AggResults = self.WeightedAvg(localUpdates, p_k)  
                global_model.load_state_dict(AggResults)
                start_time = time.time()
                self.utility_records = {
                    str([]): (DNNTest(global_model, self.Tst,
                                      test_bs = self.args.test_bs,
                                      metric = self.args.test_metric),
                              time.time()-start_time)
                    }
                self.utility_file_path = self.utility_file_path.replace(
                                            '-R%s'%ridx,'')
                
            print('Average time cost for computing utility (averged by all samples): ',
                  sum([np.sum(list(utilityComputationTimeCost.values()))\
                   for utilityComputationTimeCost in dict_utilityComputationTimeCost.values()]) /\
                  sum([len(utilityComputationTimeCost.values())\
                   for utilityComputationTimeCost in dict_utilityComputationTimeCost.values()])   )
            # compute the overall SV of each player
            MultiRoundSV=dict()
            for rSV in round_SV.values():
                for player_id in rSV.keys():
                    if player_id not in MultiRoundSV.keys():
                        MultiRoundSV[player_id] = rSV[player_id]
                    else:
                        MultiRoundSV[player_id] += rSV[player_id]
            print("\n Final Resultant SVs: ", MultiRoundSV)
        self.taskTerminated = True
        #thread.join()
        
    def preExp_statistic(self):
        # reinitialize!!!
        self.utility_records = {str([]):(0,0)}
        # pre-experiment statistics
        utilityComputationTimeCost=dict()
        for player_idx in range(len(task.players)):
            
            _, timeCost = task.utilityComputation(
                range(player_idx), 
                gradient_approximation=self.args.gradient_approximation,
                testSampleSkip=self.args.testSampleSkip)
            print('Computing utility for %s players tasks %s timeCost...'%(
                player_idx, timeCost))
            utilityComputationTimeCost[player_idx] = timeCost
        print('Average time cost for computing utility: ',
              np.mean(list(utilityComputationTimeCost.values())))
        return utilityComputationTimeCost
    
if __name__ == '__main__':    
    args = args_parser()
    if args.log_file!='':
        old_stdout = sys.stdout
        file = open(args.log_file, 'w')
        sys.stdout = file
        
    print('Experiment arguemtns: ', args)
    
    # formal experiments
    task = Task(args)
    task.run()
    # Task terminated!
    sys.stdout.flush()
    if args.log_file!='':
        sys.stdout = old_stdout 
        file.close()
    sys.exit()                  
