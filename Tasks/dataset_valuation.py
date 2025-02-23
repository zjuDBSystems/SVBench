import torch
import copy
import random
import os

from .data_preparation import data_prepare
from .nets import RegressionModel, NN, CNNCifar, CNN
from .utils import DNNTrain, DNNTest


class DSV():
    def __init__(self, dataset, manual_seed, GA):
        self.GA = GA
        self.dataset_info = {
            #'iris': (3, 4, 50, 16, 0.01, 12),
            #'wine': (3, 13, 100, 16, 0.001, 15),
            'mnist': (10, 1, 1, 128, 0.1, 6000),
            'cifar': (10, 3, 1, 128, 0.1, 5000),
            '2dplanes': (2, 10, 100, 64, 0.01, 3261),
            'bank': (2, 29, 100, 16, 0.001, 628),
            'dota': (2, 115, 10, 64, 0.005, 8236)
        }
        self.num_classes, self.num_feature, self.ep, \
            self.bs, self.lr, self.tuple_to_set  = self.dataset_info[dataset]

        self.model = None
        self.dataset = dataset

        trn_path = 'data/%s0/train0.pt' % (dataset)
        if not os.path.exists(trn_path):
            data_prepare(manual_seed=manual_seed,
                         dataset=dataset,
                         num_classes=self.num_classes)

        # player setting
        self.trn_data = torch.load(trn_path)

        all_dataIdx = list(range(len(self.trn_data)))
        random.shuffle(all_dataIdx)
        self.players = [all_dataIdx[
            start_idx: min(len(self.trn_data),
                           start_idx+self.tuple_to_set)]
                        for start_idx in range(0, len(self.trn_data),
                                           self.tuple_to_set)]
        '''
        if self.dataset == ('mnist' or 'cifar'):
            for player_idx, dataIdxs in enumerate(self.players):
                self.players[player_idx] = np.random.choice(dataIdxs, 512,
                                                            replace = False)
        '''        
        self.Tst = torch.load('data/%s0/test.pt' % (dataset))

    def utility_computation(self, player_list):
        all_data_tuple_idx = []
        for pidx in player_list:
            all_data_tuple_idx += self.players[pidx]
        player_list = all_data_tuple_idx

        utility = 0.0
        # model initialize and training
        if not self.GA or type(self.model) == type(None) or\
                len(player_list) <= 0:
            if self.GA:
                print('model initialize...')
            if self.dataset == 'cifar':
                self.model = CNNCifar(
                    num_channels=self.num_feature, num_classes=self.num_classes)
            elif self.dataset == 'mnist':
                self.model = CNN(num_channels=self.num_feature,
                                   num_classes=self.num_classes)
            elif self.dataset in ['wine', 'bank', 'dota']:
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
