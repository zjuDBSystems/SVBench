import torch
import copy
import os
import numpy as np

from .data_preparation import data_prepare
from .nets import RegressionModel, NN
from .utils import DNNTrain, DNNTest


class DV():
    def __init__(self, dataset, manual_seed, GA):
        self.GA = GA
        self.dataset_info = {
            'iris': (3, 4, 100, 16, 0.05),
            'wine': (3, 13, 100, 16, 0.001),
            'wind': (2, 14, 100, 16, 0.01)
        }
        self.num_classes, self.num_feature, self.ep, self.bs, self.lr  \
            = self.dataset_info[dataset]

        self.model = None
        self.dataset = dataset

        trn_path = 'data/%s0/train0.pt' % (dataset)
        if not os.path.exists(trn_path):
            data_prepare(manual_seed=manual_seed,
                         dataset=dataset,
                         num_classes=self.num_classes)
        
        # player setting
        self.Tst = torch.load('data/%s0/test.pt' % (dataset))
        self.trn_data = torch.load(trn_path)
        # adjust the scale of players in order to generate exact SV with limited computing budgets
        if len(self.trn_data)<15:
            pass
        else:
            unique_labels = np.unique(self.trn_data.labels)
            label_count = dict()
            for l in unique_labels:
                label_count[l]=list(self.trn_data.labels).count(l)
            
            if self.dataset == 'wine':
                self.model = NN(num_feature=self.num_feature,
                                num_classes=self.num_classes)
            else:
                self.model = RegressionModel(
                    num_feature=self.num_feature,
                    num_classes=self.num_classes)
            model_trained_on_complete_dataset = DNNTrain(
                self.model, self.trn_data, self.lr,
                self.ep, self.bs, torch.nn.CrossEntropyLoss())
            base_overall_utility = DNNTest(
                model_trained_on_complete_dataset, self.Tst,
                metric='tst_accuracy')

            for num_players in range(15, 25):
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
                    self.ep, self.bs, torch.nn.CrossEntropyLoss())
                tmp_utility = DNNTest(tmp_model, self.Tst,
                                      metric='tst_accuracy')
                print('num_players: ', num_players,
                      'tmp_utility: ', tmp_utility, 
                      ' base_overall_utility:',base_overall_utility)
                # finding the minimal number of players 
                # whose collective utility can reach at least 
                # 80% of base_overall_utility
                if tmp_utility >= base_overall_utility*0.8:
                    break
            self.trn_data = tmp_trn_data
        self.players = self.trn_data

    def utility_computation(self, player_list):
        # server for MIA (no use in the other cases)
        player_list = [self.trn_data.idxs[pidx]
                       for pidx in player_list]

        utility = 0.0
        # model initialize and training
        if not self.GA or type(self.model) == type(None) or\
                len(player_list) <= 0:
            if self.GA:
                print('model initialize...')
            if self.dataset == 'wine':
                self.model = NN(num_feature=self.num_feature,
                                num_classes=self.num_classes)
            else:
                self.model = RegressionModel(
                    num_feature=self.num_feature,
                    num_classes=self.num_classes)
        if len(player_list) <= 0:
            utility = DNNTest(self.model, self.Tst,
                              metric='tst_accuracy')
            return utility

        loss_func = torch.nn.CrossEntropyLoss()
        if self.GA:
            epoch = 1
            batch_size = 1
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

        return utility
