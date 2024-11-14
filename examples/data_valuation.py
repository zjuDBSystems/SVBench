# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 16:23:58 2024

@author: admin
"""

from shapley import Shapley
from arguments import args_parser
import torch, time 
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from models.Nets import RegressionModel, CNN, NN  # , CNNCifar
from ML_utils import DNNTrain, DNNTest, find_free_gpu
#from torch.utils.data import DataLoader
import copy, sys, threading, random

class Task():
    def __init__(self, args):
        self.taskTerminated = False
        self.args = args
        free_gpu = find_free_gpu() #int(all_gpus[0])
        self.device = torch.device(
            'cuda:{}'.format(free_gpu) \
                if torch.cuda.is_available() else 'cpu'
                )
            
        # DA task settings
        self.task = args.task
        self.model = None
        self.model_name = args.model_name
        
        
        # player setting
        self.trn_data = torch.load('data/%s%s/train0.pt'%(
            args.dataset, args.data_allocation), 
            #map_location=self.device
            )
        print('train size: ', len(self.trn_data))
        
        if self.args.tuple_to_set>0:
            all_dataIdx = list(range(len(self.trn_data)))
            random.shuffle(all_dataIdx)
            self.players = [all_dataIdx[start_idx: 
                                        start_idx+self.args.tuple_to_set]\
                            for start_idx in range(0, len(self.trn_data),
                                                   self.args.tuple_to_set)]
        else:
            self.players = self.trn_data 
            
        # utility setting
        self.utility_records = {str([]):(0,0)} 
        self.Tst = torch.load(
            'data/%s%s/test.pt'%(args.dataset, args.data_allocation), 
            #map_location=self.device
            )
        print('test size: ', len(self.Tst))
        print('test labels: ', self.Tst.labels)
        if self.model_name == 'KNN':
            self.X_test = []
            self.y_test = []
            for item in self.Tst:
                data, label = item[0], item[1]
                self.X_test.append(data.numpy().reshape(-1))
                self.y_test.append(label)
            self.X_test = np.array(self.X_test)
            self.y_test = np.array(self.y_test)
            
    def utilityComputation(self, player_idxs, gradient_approximation=False, 
                           testSampleSkip = False):
        if self.args.tuple_to_set>0:
            # invoked when the valuation target is the data set 
            all_data_tuple_idx = []
            for pidx in player_idxs:
                all_data_tuple_idx += self.players[pidx]
            player_idxs = all_data_tuple_idx
        
        startTime = time.time()
        utility_record_idx = str(sorted(player_idxs))
        
        if utility_record_idx in self.utility_records:
        #and not gradient_approximation:
            #print('Read DV compute utility with players:', utility_record_idx)
            return self.utility_records[utility_record_idx][0], 0#time.time()-startTime
        #print('DV compute utility with players:', utility_record_idx,
        #      'utility_records len:', len(self.utility_records))
        
        utility = 0.0
        if self.model_name == 'KNN':
            # model initialize and training 
            # (maybe expedited by some ML speedup functions)
            self.model = KNeighborsClassifier(
                n_neighbors= min(len(player_idxs), self.args.n_neighbors)
                )
            X_train = []
            y_train = []
            for idx in player_idxs:
                X_train.append(self.trn_data[idx][0].numpy().reshape(-1))
                y_train.append(self.trn_data[idx][1])
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            self.model = DNNTrain(self.model, (X_train, y_train))
            #self.model.fit(X_train, y_train)
            
            # model testing (maybe expedited by some ML speedup functions)
            utility = DNNTest(self.model, (self.X_test, self.y_test),
                              test_bs= len(self.y_test),
                              metric = self.args.test_metric)
        
        elif self.model_name in ['CNN', 'Linear', 'NN']:
            # model initialize and training 
            if not gradient_approximation or type(self.model) == type(None) or\
            len(player_idxs) <= 0:
                if gradient_approximation:
                    print ('model initialize...')
                if self.model_name == 'CNN':
                    self.model = CNN(args=self.args)
                elif self.model_name == 'Linear':
                    self.model = RegressionModel(args=self.args)
                elif self.model_name == 'NN':
                    self.model = NN(args=self.args)

            if len(player_idxs) <= 0:
                utility =  DNNTest(self.model, self.Tst,
                                   test_bs= self.args.test_bs,
                                   metric = self.args.test_metric)
                endTime = time.time()
                self.utility_records[utility_record_idx] = (utility, endTime-startTime)
                return utility, endTime-startTime
            
            loss_func = torch.nn.CrossEntropyLoss()
            if gradient_approximation:
                epoch = 1
                batch_size = 1
                player_idxs = player_idxs[-1:]
            else:
                epoch = self.args.ep
                batch_size = self.args.bs
            
            trn_data = copy.deepcopy(self.trn_data)
            trn_data.idxs = player_idxs # iterate samples in sub-coalition
            self.model = DNNTrain(self.model, trn_data, 
                                  epoch, batch_size,
                                  self.args.lr, loss_func)
            del trn_data
            torch.cuda.empty_cache()
                    
            # model testing (maybe expedited by some ML speedup functions)
            utility = DNNTest(self.model, self.Tst,
                              test_bs= self.args.test_bs,
                              metric = self.args.test_metric)
           
        else:
            # DA with other types of ML models are left for future experiments
            pass
        
        endTime = time.time()
        self.utility_records[utility_record_idx] = (utility, endTime-startTime)
        return utility, endTime-startTime
    
    def printFlush(self):
        while not self.taskTerminated:
            sys.stdout.flush()
            time.sleep(5)
            
    def run(self):
        thread = threading.Thread(target=task.printFlush)
        thread.daemon = True
        thread.start()

        if self.args.gradient_approximation:
            self.args.num_parallelThreads=1

        self.preExp_statistic()
        # reinitialize!!!
        self.utility_records = {str([]):(0,0)}
        SVtask = Shapley(players = self.players, 
                         taskUtilityFunc=self.utilityComputation, 
                         args = self.args)
        SVtask.CalSV()
        self.taskTerminated = True
        #thread.join()
        
    def preExp_statistic(self):
        # pre-experiment statistics
        self.utility_records = {str([]):(0,0)}
        utilityComputationTimeCost=dict()
        for player_idx in range(len(self.players)+1):
            
            _, timeCost = self.utilityComputation(
                range(player_idx), 
                gradient_approximation=self.args.gradient_approximation,
                testSampleSkip=self.args.testSampleSkip)
            print('Computing utility with %s players tasks %s timeCost...'%(
                player_idx, timeCost))
            utilityComputationTimeCost[player_idx] = timeCost
        print('Average time cost for computing utility: ',
              np.mean(list(utilityComputationTimeCost.values())))
          
        
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
