# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 16:23:58 2024

@author: admin
"""

from shapley import Shapley
from arguments import args_parser
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from models.Nets import  RegressionModel, CNN, NN#, CNNCifar
from ML_utils import DNNTrain, DNNTest, find_free_gpu
#from torch.utils.data import DataLoader
from functools import reduce
import sys, threading, queue, copy, torch, time, json, portalocker, os

class Task():
    def __init__(self, args):
        self.taskTerminated = False
        self.args = args
        free_gpu = find_free_gpu() #int(all_gpus[0])
        self.device = torch.device(
            'cuda:{}'.format(free_gpu) \
                if torch.cuda.is_available() else 'cpu'
                )
            
        # FA task settings
        self.task = args.task + \
                    ('-n' if args.dataNormalize else '') +\
                    ('-maxIter%s'%args.scannedIter_maxNum \
                     if args.scannedIter_maxNum!=np.inf else "")
        self.model = None
        self.model_name = args.model_name
        self.Trn = torch.load('data/%s%s/train0.pt'%(
            args.dataset, args.data_allocation), 
            #map_location=self.device
            )
        if args.dataNormalize:
            shape =  self.Trn.dataset.shape
            self.Trn.dataset = self.Trn.dataset.reshape((shape[0],-1))
            min_vals, _ = torch.min(self.Trn.dataset, dim=0, keepdim=True)
            max_vals, _ = torch.max(self.Trn.dataset, dim=0, keepdim=True)
            self.Trn.dataset = (self.Trn.dataset - min_vals) / (max_vals - min_vals)
            self.Trn.dataset = self.Trn.dataset.reshape(shape)
        
        #if self.model_name in ['KNN', 'Tree']:
        self.X_train = []
        self.y_train = []
        for item in self.Trn:
            data, label = item[0], item[1]
            if type(data)!=np.ndarray:
                data = data.numpy()
            self.X_train.append(data.reshape(-1))
            self.y_train.append(label)
        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)
        
        baseline_idxs = []
        for label in np.unique(self.y_train):
            class_idx = np.where(self.y_train==label)[0]
            class_data = self.X_train[class_idx]
            intra_class_dis = [np.mean(
                np.linalg.norm(
                    class_data - class_data[data_idx], ord=1, axis=1)
                ) for data_idx in range(len(class_data))]
            #print('label %s'%label, 
            #      np.array(intra_class_dis)[np.argsort(intra_class_dis)])
            baseline_idxs += class_idx[np.argsort(intra_class_dis)][:5].tolist()
            #baseline_idxs += np.random.choice(np.where(self.y_train==label)[0], 
            #                                  5, replace=False).tolist()
        print('selected baseline idx: ',baseline_idxs)
        self.initial_feature_baseline = copy.deepcopy(self.X_train)[baseline_idxs]
        self.feature_baseline = copy.deepcopy(self.X_train)[baseline_idxs]
        '''
        self.feature_baseline = copy.deepcopy(self.X_train)
        for feature_id in range(self.X_train.shape[-1]):
            val, counts = np.unique(self.X_train[:, feature_id],
                                    return_counts = True)
            sorted_val = sorted(val)
            diff = set([
                    round(sorted_val[val_id]-sorted_val[val_id-1],3) \
                    for val_id in range(1,len(sorted_val))])
            if len(diff)>1:#set(val) != set(range(len(val))):
                print('continuous feature%s:'%feature_id, sorted_val[:10])
                # use mean for replacement when continuous features are unknown
                self.feature_baseline[:, feature_id] = np.mean(self.X_train[:, feature_id])
            else:
                print('norminal feature%s:'%feature_id, sorted_val)
                self.feature_baseline[:, feature_id] = np.argmax(
                    np.bincount(self.X_train[:, feature_id].tolist()))
                
        self.feature_baseline = np.array(
            [eval(baseline) \
             for baseline in set([str(tmp) for tmp in self.feature_baseline.tolist()])
             ])  
        
        print('feature_baseline shape: ', self.feature_baseline.shape)
        '''
        self.Tst = torch.load(
            'data/%s%s/test.pt'%(args.dataset, args.data_allocation), 
            #map_location=self.device
            )
        if args.dataNormalize:
            shape =  self.Tst.dataset.shape
            self.Tst.dataset = self.Tst.dataset.reshape((shape[0],-1))
            min_vals, _ = torch.min(self.Tst.dataset, dim=0, keepdim=True)
            max_vals, _ = torch.max(self.Tst.dataset, dim=0, keepdim=True)
            self.Tst.dataset = (self.Tst.dataset - min_vals) / (max_vals - min_vals)
            self.Tst.dataset = self.Tst.dataset.reshape(shape)
            
        #if self.model_name in ['KNN', 'Tree']:
        self.X_test = []
        self.y_test = []
        for item in self.Tst:
            data, label = item[0], item[1]
            if type(data)!=np.ndarray:
                data = data.numpy()
            self.X_test.append(data.reshape(-1))
            self.y_test.append(label)
        self.X_test = np.array(self.X_test)
        self.complete_X_test = copy.deepcopy(self.X_test)
        self.y_test = np.array(self.y_test)
        self.complete_y_test = copy.deepcopy(self.y_test)
        
        # select test samples 
        self.num_samples_each_class = 1 # select only one sample in each class
        self.selected_test_samples = []
        if self.args.attack_type == None and\
        len(self.players)>5 and \
        False not in [len(
            np.where(self.complete_y_test==label)[0]
            ) > self.num_samples_each_class  \
                for label in np.unique(self.complete_y_test)]: 
            for label in np.unique(self.complete_y_test):
                self.selected_test_samples += np.random.choice(
                    np.where(self.complete_y_test==label)[0], 
                    self.num_samples_each_class, replace=False).tolist() 
        else:
            self.selected_test_samples = range(len(self.complete_X_test))
        
        # player setting
        self.players = [feature_idx \
                        for feature_idx in range(self.Tst[0][0].reshape(-1).shape[-1])]
        # utility setting
        self.writingUtility = False
        self.utility_file_path = 'logs/UtilityRecord_%s_%s_%s.json'%(
                self.args.task, self.args.dataset, 
                self.args.manual_seed)
        self.utility_records = self.readUtilityRecords()
        self.record_interval = 5*len(self.players)
        self.record_count = len(self.utility_records)//self.record_interval
        print('len(utility): ', len(self.utility_records),
              self.utility_file_path)
        
        #self.skippableTestSample = dict([(player_id, set()) \
        #                for player_id in range(len(self.players))])
        
    
    def readUtilityRecords(self, emptySet_utility = None):
        if type(emptySet_utility)==type(None):
            emptySet_utility =  {str([]):(0,0)}
            
        if not os.path.exists(self.utility_file_path):
            return emptySet_utility
        
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
            
            
    def sklearn_predict(self, replace_idxs, replace_val, results):
        X_test = copy.deepcopy(self.X_test)
        X_test[:, replace_idxs] = replace_val
        results.put(#self.model.predict(X_test)
            DNNTest(self.model, (X_test, self.y_test),
                    test_bs= len(X_test),
                    metric = self.args.test_metric)
            )
        
        
    def torch_predict(self, replace_idxs, replace_val, results):
        Tst = copy.deepcopy(self.Tst)                
        ori_shape = Tst.dataset.shape
        dataset = Tst.dataset.reshape((ori_shape[0],-1))
        dataset[:, replace_idxs]= torch.FloatTensor(replace_val)
        Tst.dataset = dataset.reshape(ori_shape)
        results.put(#self.model.predict(X_test)
            DNNTest(self.model, Tst,
                    test_bs= self.args.test_bs,
                    metric = self.args.test_metric)
            )
    
    def utilityComputation(self, player_idxs, gradient_approximation=False, 
                           testSampleSkip = False):
        
        startTime = time.time()
        utility_record_idx = str(sorted(player_idxs))
        if utility_record_idx in self.utility_records:
        #and not gradient_approximation:
            #print('Read FA compute utility with players:', utility_record_idx)
            return self.utility_records[utility_record_idx][0], 0#time.time()-startTime
        #print('FA compute utility with players:', utility_record_idx,
        #      'utility_records len:', len(self.utility_records))
        
        utility = 0.0
        '''
        # adjust baselines (hard to adjust due to features with continuous values)
        line_idxs = np.where(
            self.X_train[:, player_idxs] == self.X_test[0, player_idxs])[0].tolist()
        baseline_idxs = sorted(
            set([idx for idx in line_idxs \
                 if line_idxs.count(idx)==len(player_idxs)])
                )
        if len(baseline_idxs) > 0:
            self.feature_baseline = self.X_train[baseline_idxs]
        else:
            print('Baseline cannot be found under conditional distribution！','\n',
                  self.X_test[0, player_idxs],'\n',
                  self.X_train[:, player_idxs])
            self.feature_baseline = self.initial_feature_baseline
        '''
        replace_idxs = [feature_idx \
                        for feature_idx in range(self.X_train.shape[-1])\
                        if feature_idx not in player_idxs]
        baselines = [
            eval(baseline) \
            for baseline in set([
                    str(tmp) \
                    for tmp in self.feature_baseline[:, replace_idxs].tolist()])
             ]
        if self.model_name in ['KNN', 'Tree']:
            # model testing (maybe expedited by some ML speedup functions)
            results = queue.Queue()
            for order, replace_val in enumerate(baselines):
                thread = threading.Thread(
                    target=self.sklearn_predict,
                    args=(replace_idxs, replace_val, results))
                thread.daemon = True
                thread.start()
                if self.args.num_parallelThreads <=1 or\
                (order>0 and order%self.args.num_parallelThreads==0):
                    thread.join()
                    
            while not len(baselines) == results._qsize():
                time.sleep(3)
            
            predictions = np.sum(list(results.queue), 0)/len(baselines)
            utility = predictions.sum()/reduce((lambda x,y:x*y),predictions.shape)
            
        
        elif self.model_name in ['CNN', 'Linear', 'NN']:
            # model testing (maybe expedited by some ML speedup functions)
            results = queue.Queue()
            for order, replace_val in enumerate(baselines):
                thread = threading.Thread(
                    target=self.torch_predict,
                    args=(replace_idxs, replace_val, results))
                thread.daemon = True
                thread.start()
                if self.args.num_parallelThreads <=1 or\
                (order>0 and order%self.args.num_parallelThreads==0):
                    thread.join()
            while not len(baselines) == results._qsize():
                time.sleep(3)
            
            predictions = np.sum(list(results.queue), 0)/len(baselines)
            utility = predictions.sum()/reduce((lambda x,y:x*y),predictions.shape)
            
        else:
            # DA with other types of ML models are left for future experiments
            print('Note: DA with other types of ML models'+\
                  'are left for future experiments!')
        
        endTime = time.time()
        self.utility_records[utility_record_idx] = (utility, endTime-startTime)
        self.writeUtilityRecord()
        return utility, endTime-startTime
    
    def printFlush(self):
        while not self.taskTerminated:
            sys.stdout.flush()
            time.sleep(5)
            
    def trainModel(self):
        model_path = 'models/attribution_%s-%s.pt'%(self.task,self.args.dataset)
        if os.path.exists(model_path) and\
        self.args.modelRetrain == False:
            self.model = torch.load(model_path, map_location=self.device)
            if self.model_name in ['KNN', 'Tree']:
                testData =  (self.X_test, self.y_test)
            elif self.model_name in ['CNN', 'Linear', 'NN']:
                testData = self.Tst
            print('Given model accuracy: ',
                  DNNTest(self.model, testData,
                          test_bs= len(self.X_test),
                          metric = 'tst_accuracy', 
                          pred_print = True)
                  )
            return
        
        if self.model_name in ['KNN', 'Tree']:
            if self.model_name == 'KNN':
                self.model = KNeighborsClassifier(
                    n_neighbors= self.args.n_neighbors
                    )
            else:
                self.model = DecisionTreeClassifier(
                    criterion='entropy', 
                    random_state=self.args.manual_seed, 
                    max_depth= self.args.tree_maxDepth#self.X_train.shape(-1)-1
                    )
            self.model = DNNTrain(self.model, 
                                  (self.X_train, self.y_train))
            print('Given model accuracy: ',
                  DNNTest(self.model, (self.X_test, self.y_test),
                          test_bs= len(self.X_test),
                          metric = 'tst_accuracy', 
                          pred_print = True)
                  )
        elif self.model_name in ['CNN', 'Linear', 'NN']:
            if self.model_name == 'CNN':
                self.model = CNN(args=self.args)
            elif self.model_name == 'NN':
                self.model = NN(args=self.args)
            else:
                self.model = RegressionModel(args=self.args)
            loss_func = torch.nn.CrossEntropyLoss()
            self.model = DNNTrain(self.model, self.Trn, 
                                  self.args.ep, self.args.bs,
                                  self.args.lr, loss_func)
            print('Given model accuracy: ',
                  DNNTest(self.model, self.Tst,
                          test_bs= self.args.test_bs,
                          metric = 'tst_accuracy', 
                          pred_print = True) 
                  )
        torch.save(self.model, model_path)
        
    def run(self):
        thread = threading.Thread(target=self.printFlush)
        thread.daemon = True
        thread.start()
        
        self.trainModel()
        
        self.testSampleFeatureSV = dict()
        dict_utilityComputationTimeCost = dict()
        if self.model_name in ['KNN', 'Tree']:
            
            # start testing
            for test_idx in range(len(self.complete_X_test)):
                # compute SV for only selected test samples for saving time cost
                if test_idx not in self.selected_test_samples:
                    continue
                
                # reinitialize!!!
                self.X_test = self.complete_X_test[test_idx:test_idx+1]
                self.y_test = self.complete_y_test[test_idx:test_idx+1]
                   
                dict_utilityComputationTimeCost[test_idx] = self.preExp_statistic()
                # reinitialize!!!
                self.utility_file_path = 'logs/UtilityRecord_%s_%s_%s_Idx%s.json'%(
                        self.task, self.args.dataset, 
                        self.args.manual_seed, test_idx)
                self.utility_records = self.readUtilityRecords()
                self.record_count = len(self.utility_records)//self.record_interval
                # formal exp
                if len(self.players)<15:
                    print('\n test sample data: ', self.X_test,
                          '\n test sample label: ', self.y_test)
                    print(self.utility_file_path,
                          'len(self.utility_records):', 
                          len(self.utility_records))
                SVtask = Shapley(players = self.players, 
                                 taskUtilityFunc=self.utilityComputation, 
                                 args = self.args)
                SVtask.CalSV()
                self.testSampleFeatureSV[test_idx] = SVtask.SV
                print('SV of test sample %s/%s: '%(test_idx,len(self.complete_X_test)),
                      self.testSampleFeatureSV[test_idx])
                
            self.X_test = self.complete_X_test
            self.y_test = self.complete_y_test
        else:
            complete_Tst_idx = self.Tst.idxs
            
            # start testing
            for test_idx in range(len(complete_Tst_idx)):
                # compute SV for only selected test samples for saving time cost
                if test_idx not in self.selected_test_samples:
                    continue
                    
                self.Tst.idxs = complete_Tst_idx[test_idx:test_idx+1]
                
                dict_utilityComputationTimeCost[test_idx] = self.preExp_statistic()
                # reinitialize!!!
                self.utility_file_path = 'logs/UtilityRecord_%s_%s_%s_Idx%s.json'%(
                        self.task, self.args.dataset, 
                        self.args.manual_seed, test_idx)
                self.utility_records = self.readUtilityRecords()
                self.record_count = len(self.utility_records)//self.record_interval
                # formal exp
                if len(self.players)<15:
                    print('\n test sample data: ', self.Tst.dataset[test_idx],
                          '\n test sample label: ', self.Tst.labels[test_idx])
                    print(self.utility_file_path,
                          'len(self.utility_records):', 
                          len(self.utility_records))
                    
                SVtask = Shapley(players = self.players, 
                                 taskUtilityFunc=self.utilityComputation, 
                                 args = self.args)
                SVtask.CalSV()
                self.testSampleFeatureSV[test_idx] = SVtask.SV
                print('SV of test sample %s/%s: '%(test_idx,len(complete_Tst_idx)),
                      self.testSampleFeatureSV[test_idx], '\n')
                
            self.Tst.idx = complete_Tst_idx
        
        
        print('Average time cost for computing utility (averged by all samples): ',
              sum([np.sum(list(utilityComputationTimeCost.values()))\
               for utilityComputationTimeCost in dict_utilityComputationTimeCost.values()]) /\
              sum([len(utilityComputationTimeCost.values())\
               for utilityComputationTimeCost in dict_utilityComputationTimeCost.values()])   )
        
        SV_matrice=np.zeros((len(self.testSampleFeatureSV), len(SVtask.SV)))
        for matrice_idx, test_idx in enumerate(self.testSampleFeatureSV.keys()):
            for feature_idx in self.testSampleFeatureSV[test_idx].keys():
                SV_matrice[matrice_idx, feature_idx] = self.testSampleFeatureSV[test_idx][feature_idx]
        print('Final average SV: ', np.mean(SV_matrice,0))
        self.taskTerminated = True 
        #thread.join()
    
    def preExp_statistic(self):
        # reinitialize!!!
        self.utility_records = {str([]):(0,0)}
        utilityComputationTimeCost=dict()
        for player_idx in range(len(self.players)+1):
            utility, timeCost = self.utilityComputation(
                range(player_idx))
            print('Computing utility %s for %s players tasks %s timeCost...'%(
                utility, player_idx, timeCost))
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
    
    task = Task(args)
    task.run()
    # Task terminated!
    sys.stdout.flush()
    if args.log_file!='':
        sys.stdout = old_stdout 
        file.close()
    sys.exit() 
