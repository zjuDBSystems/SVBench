import torch
import copy
import os
import numpy as np

from .data_preparation import data_prepare
from .nets import RegressionModel, NN, NN_ttt
from .utils import DNNTrain, DNNTest


class DV():
    def __init__(self, dataset, manual_seed, GA):
        self.GA = GA
        self.dataset_info = {
            'iris': (3, 4, 100, 16, 0.05, 
                     {42:
                      [23, 71, 98, 55, 67, 8, 11, 82, 15, 12, 93, 99, 106, 101, 81, 21, 68, 77],
                      10: 
                      [33, 13, 104, 14, 27, 52, 110, 82, 47, 18, 95, 12, 109, 85, 24, 106, 42, 119],
                      100: 
                      [26, 4, 66, 52, 35, 78, 53, 39, 110, 82, 95, 115, 16, 46, 21, 63, 24, 106]
                      }
                     ),
            'wine': (3, 13, 100, 16, 0.001, 
                     {42: 
                      [23, 30, 47, 40, 111, 94, 131, 84, 109, 58, 41, 135, 70, 101, 78],
                      10: 
                      [61, 98, 118, 21, 92, 65, 140, 58, 100, 131, 108, 105, 35, 113, 17],
                      100: 
                      [118, 139, 49, 76, 55, 65, 140, 66, 92, 97, 141, 2, 106, 17, 95, 70, 72, 27]
                       }),
            'wind': (2, 14, 100, 16, 0.01, 
                     {42: 
                      [2598, 3025, 3549, 2232, 1820, 5045, 648, 4592, 3771, 2853, 1605, 4593, 4399, 1835, 2289, 4689, 1499, 1411],
                      10: 
                      [514, 966, 1906, 2825, 5231, 4892, 2604, 3021, 4344, 1091, 374, 4832, 3973, 1212, 1773, 3998, 3577, 1112, 2947, 1809, 2318, 1698, 4303],
                      100: 
                      [2715, 3636, 2327, 4557, 1810, 1041, 4051, 4811, 1263, 2605, 945, 4687, 4947, 614, 1216, 319, 4595, 4084, 901, 4199]
                      }),
                     'ttt': (2, 9, 100, 16, 0.005,
                    {42: [443, 569, 739, 437, 608, 590, 678, 586, 359, 201, 30, 756, 474, 51, 574, 467, 633, 671, 727, 269, 101, 14, 666, 60, 203, 18], 
                     10: [160, 708, 498, 53, 387, 628, 404, 501, 752, 737, 281, 235, 123, 109, 257, 210, 2, 726, 336, 370, 125], 
                     100: [238, 457, 29, 208, 240, 380, 712, 12, 500, 390, 226, 88, 14, 199, 369, 563, 182, 581, 362, 139, 269, 359]}),
            'bank': (2, 29, 100, 16, 0.001,
                    {42: [5801, 431, 5932, 3693, 4711, 5165, 3733, 958, 944, 1635, 4670, 5862, 3153, 1851, 3438, 5128, 4190, 5077], 
                     10: [3278, 2162, 888, 5023, 2094, 2532, 2302, 2698, 3630, 1626, 1753, 2654, 1699, 1392, 660, 2445, 2786, 2367, 4132, 1883, 4922, 6182], 
                     100:[4797, 4089, 886, 5437, 2889, 6204, 1999, 732, 5143, 4380, 449, 6025, 5855, 1144, 1454, 5568, 4839, 1466, 5833, 3275, 5933, 167, 5560, 3446, 1537, 1592]}),
            
            
        }
        self.num_classes, self.num_feature, \
        self.ep, self.bs, self.lr, select_idx = self.dataset_info[dataset]

        self.model = None
        self.dataset = dataset

        trn_path = 'data/%s0/train0.pt' % (dataset)
        if not os.path.exists(trn_path):
            data_prepare(manual_seed=manual_seed,
                         dataset=dataset,
                         num_classes=self.num_classes)
        
        
        # player setting
        self.Tst = torch.load('data/%s0/test.pt' % (dataset), weights_only=False)
        self.trn_data = torch.load(trn_path, weights_only=False)
        # adjust the scale of players in order to generate exact SV with limited computing budgets
        if select_idx[manual_seed]!=None:
            self.trn_data.idxs = select_idx[manual_seed]
        else:
            unique_labels = np.unique(self.trn_data.labels)
            label_count = dict()
            for l in unique_labels:
                label_count[l]=list(self.trn_data.labels).count(l)
            
            loss_func = torch.nn.CrossEntropyLoss()
            if self.dataset in ['wine', 'bank']:
                self.model = NN(num_feature=self.num_feature,
                                num_classes=self.num_classes)
            elif self.dataset in ['ttt']:
                self.model = NN_ttt(num_feature=self.num_feature,
                                num_classes=self.num_classes)
            else:
                self.model = RegressionModel(
                    num_feature=self.num_feature,
                    num_classes=self.num_classes)
            model_trained_on_complete_dataset = DNNTrain(
                self.model, self.trn_data, self.lr,
                self.ep, self.bs, loss_func)
            base_overall_utility = DNNTest(
                model_trained_on_complete_dataset, self.Tst,
                metric='tst_accuracy')
            num_players_utility = dict()
            for num_players in range(15, 30):
                select_idx = []
                for lidx, l in enumerate(unique_labels):
                    if lidx == len(unique_labels)-1:
                        select_idx += np.random.choice(
                            np.where(np.array(self.trn_data.labels)==l)[0],
                            num_players-len(select_idx), 
                            replace=False).tolist()
                    else:
                        select_idx += np.random.choice(
                            np.where(np.array(self.trn_data.labels)==l)[0],
                            int(num_players*label_count[l]/len(self.trn_data)), 
                            replace=False).tolist()
                tmp_trn_data = copy.deepcopy(self.trn_data)
                tmp_trn_data.idxs = select_idx
                tmp_model = DNNTrain(
                    self.model, tmp_trn_data, self.lr,
                    self.ep, self.bs, loss_func)
                tmp_utility = DNNTest(tmp_model, self.Tst,
                                      metric='tst_accuracy')
                num_players_utility[num_players] = (tmp_utility,select_idx)
                print('select_idx: ', select_idx, '\n',
                      'num_players: ', num_players,
                      'tmp_utility: ', tmp_utility, 
                      ' base_overall_utility:',base_overall_utility)
            
            # finding the minimal number of players 
            # whose collective utility can reach at least 
            # 80% of base_overall_utility
            qualified_items = []
            max_utility = 0
            for num_players, res in num_players_utility.items():
                if res[0]>max_utility:
                    qualified_items = [num_players]
                    max_utility = res[0]
                elif res[0] == max_utility:
                    qualified_items.append(num_players)
            self.trn_data.idxs = num_players_utility[min(qualified_items)][1]
            print('final select_idx: ', self.trn_data.idxs, '\n',
                  ' player num:', len(self.trn_data.idxs), '\n',
                  ' overall_utility:',num_players_utility[min(qualified_items)][0])
            if max_utility < base_overall_utility*0.75:
                print('Warning: overall utility is not qualified'+\
                      'after reducing the number of training data!')
            self.model = None
        self.players = self.trn_data
        # exit()
        
    def utility_computation(self, player_list):
        num_players = len(player_list)
        # server for MIA experiments (no use in the other cases)
        player_list = [self.trn_data.idxs[pidx]
                       for pidx in player_list]

        utility = 0.0
        # model initialize and training
        initial_flag=False
        if not self.GA or type(self.model) == type(None) or\
        num_players <= 0:
            initial_flag=True
            if self.GA:
                print('model initialize...')
                
            if self.dataset in ['wine', 'bank']:
                self.model = NN(num_feature=self.num_feature,
                                num_classes=self.num_classes)
            else:
                self.model = RegressionModel(
                    num_feature=self.num_feature,
                    num_classes=self.num_classes)
        if num_players <= 0:
            utility = DNNTest(self.model, self.Tst,
                              metric='tst_accuracy')
            self.model = None
            return utility

        loss_func = torch.nn.CrossEntropyLoss()
        if self.GA:
            epoch = 1
            batch_size = 1
            if not (initial_flag and num_players>1):
                player_list = player_list[-1:]
        else:
            epoch = self.ep
            batch_size = self.bs

        trn_data = copy.deepcopy(self.trn_data)
        trn_data.idxs = player_list  # iterate samples in sub-coalition
        self.model = DNNTrain(self.model, trn_data, self.lr,
                              epoch, batch_size, loss_func)
        del trn_data
        torch.cuda.empty_cache()

        # model testing (maybe expedited by some ML speedup functions)
        utility = DNNTest(self.model, self.Tst,
                          metric='tst_accuracy')
        
        if not self.GA:
            self.model = None # reset model
        else: 
            if num_players==len(self.players):
                self.model = None
        return utility
