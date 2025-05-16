# -*- coding: utf-8 -*-
import torch
import copy
import os
import time
import sys

from .utils import DNNTrain, DNNTest, find_free_device
from .nets import CNNCifar, CNN, RegressionModel, NN
from .data_preparation import data_prepare


class FL():
    def __init__(self, dataset, manual_seed, GA, TSS):
        torch.cuda.empty_cache()
        self.GA = GA
        self.TSS = TSS
        self.dataset = dataset
        self.manual_seed = manual_seed
        self.dataset_info = {
            'cifar': (10, 10, 10, 3, 3, 64, 0.1, 1, '5', 1000),
            'mnist': (10, 10, 10, 1, 1, 64, 0.1, 1, '6', 1000),
            'wind': (2, 10, 10, 14, 3, 64, 0.01, 1, '1', 657),
            'adult': (2, 10, 10, 14, 3, 64, 0.01, 1, '1', int(48842/10)),
            'dota': (2, 10, 10, 115, 10, 64, 0.005, 1, '1', int(102944/10)),
        }
        self.num_classes, self.num_clients, self.max_round,     \
            self.num_channels, self.local_ep, self.local_bs,    \
            self.lr, self.decay_rate, multiplier, data_size_mean = self.dataset_info[dataset]
        self.ridx = 0
        self.device = find_free_device()
        self.model = None

        tst_path = 'data/%s1/test.pt' % (dataset)
        if not os.path.exists(tst_path):
            data_prepare(manual_seed=manual_seed, dataset=dataset, num_classes=self.num_classes,
                         data_allocation=1, num_trainDatasets=10, group_size='10',
                         multiplier=multiplier, data_size_mean=data_size_mean)

        self.Tst = torch.load(tst_path, weights_only=False)
        self.stored_gradients = dict()
        self.skippable_test_sample = [
            dict([(no, set()) for no in range(self.num_clients)])
            for ridx in range(self.max_round)]

        # player setting
        self.player_datasets = [
            torch.load('data/%s1/train%s.pt' % (dataset, no), weights_only=False)
            for no in range(self.num_clients)]
        self.player_data_size = [len(self.player_datasets[no])
                                 for no in range(self.num_clients)]
        self.players = [
            dict([(ridx, None) for ridx in range(self.max_round)])
            for no in range(self.num_clients)]

        model_paths = ['models/FL_%s_R%sC%s.pt' % (
            (self.dataset if self.manual_seed == 42
             else (self.dataset+"-"+str(self.manual_seed))),
            ridx, trainer_idx)
            for ridx in range(self.max_round)
            for trainer_idx in range(len(self.player_datasets))]
        if False not in [os.path.exists(model_path)
                         for model_path in model_paths]:
            print("Models loading...")
            global_model = self.model_initiation()
            for ridx in range(self.max_round):
                localUpdates = dict()
                p_k = dict()
                for player_idx in range(len(self.player_datasets)):
                    localUpdates[player_idx] = torch.load('models/FL_%s_R%sC%s.pt' % (
                        (self.dataset if self.manual_seed == 42
                         else (self.dataset+"-"+str(self.manual_seed))),
                        ridx, player_idx),
                        map_location=self.device, weights_only=False)
                    self.players[player_idx][ridx] = localUpdates[player_idx]
                    p_k[player_idx] = len(self.player_datasets[player_idx])

                # prepare stored_gradients and skippable_test_sample
                self.stored_gradients[ridx] = (localUpdates, p_k,
                                               global_model.state_dict())
                if self.TSS:
                    for player_idx, local_model in localUpdates.items():
                        tmp_model = self.model_initiation()
                        tmp_model.load_state_dict(local_model)
                        DNNTest(tmp_model, self.Tst,
                                record_skippable_sample=(
                                    self.skippable_test_sample[ridx],
                                    player_idx))
                # aggregation
                agg_results = self.weighted_avg(localUpdates, p_k)
                global_model.load_state_dict(agg_results)
        else:
            # model initialize and training
            # FedAvg
            if not os.path.exists('models/'):
                os.mkdir('models/')
            print("Models generating...")
            global_model = self.model_initiation()
            loss_func = torch.nn.CrossEntropyLoss()
            for ridx in range(self.max_round):
                localUpdates = dict()
                p_k = dict()
                start_time = time.time()
                for player_idx in range(len(self.player_datasets)):
                    # local model training
                    pstar_time = time.time()
                    localUpdates[player_idx] = DNNTrain(
                        model=global_model, trn_data=self.player_datasets[player_idx],
                        lr=self.lr*(self.decay_rate**ridx), epoch=self.local_ep,
                        batch_size=self.local_bs, loss_func=loss_func).state_dict()
                    self.players[player_idx][ridx] = localUpdates[player_idx]

                    p_k[player_idx] = len(self.player_datasets[player_idx])
                    torch.save(localUpdates[player_idx],
                               'models/FL_%s_R%sC%s.pt' % (
                        (self.dataset if self.manual_seed == 42
                         else (self.dataset+"-"+str(self.manual_seed))),
                        ridx, player_idx)
                    )
                    torch.cuda.empty_cache()
                    print(
                        f'Round {ridx} player {player_idx} time cost: {time.time()-pstar_time}')
                    sys.stdout.flush()

                # prepare stored_gradients and skippable_test_sample
                self.stored_gradients[ridx] = (localUpdates, p_k,
                                               global_model.state_dict())
                if self.TSS:
                    for player_idx, local_model in localUpdates.items():
                        tmp_model = self.model_initiation()
                        tmp_model.load_state_dict(local_model)
                        DNNTest(tmp_model, self.Tst,
                                record_skippable_sample=(
                                    self.skippable_test_sample[ridx],
                                    player_idx))

                # aggregation
                agg_results = self.weighted_avg(localUpdates, p_k)
                global_model.load_state_dict(agg_results)
                print(f'Round {ridx} time cost: {time.time()-start_time}'+\
                      ' global model performance: ',DNNTest(global_model, self.Tst, pred_print=True))
        #print('skippable_test_sample: ', self.skippable_test_sample)
        
        
    def weighted_avg(self, w_locals, p_k):
        parameter_keys = list(w_locals.values())[0].keys()
        idx_keys = list(w_locals.keys())
        net_glob = dict([(k, None) for k in parameter_keys])
        sum_pk = sum([p_k[idx] for idx in idx_keys])
        for k in parameter_keys:
            for idx in idx_keys:
                if type(net_glob[k]) == type(None):
                    net_glob[k] = p_k[idx]/sum_pk * w_locals[idx][k].clone()
                else:
                    net_glob[k] += p_k[idx]/sum_pk * \
                        w_locals[idx][k].to(net_glob[k].device).clone()
        return net_glob

    def model_initiation(self):
        if self.dataset == 'cifar':
            global_model = CNNCifar(
                num_channels=self.num_channels, num_classes=self.num_classes)
        elif self.dataset == 'mnist':
            global_model = CNN(num_channels=self.num_channels,
                               num_classes=self.num_classes)
        elif self.dataset in ['wind']:
            global_model = RegressionModel(
                num_feature=self.num_channels,
                num_classes=self.num_classes)
        elif self.dataset in ['adult', 'dota']:
            global_model = NN(num_feature=self.num_channels,
                            num_classes=self.num_classes)
        return global_model

    def utility_computation(self, player_idxs):
        if self.GA:
            ridx = max(self.stored_gradients.keys())
            if len(player_idxs) < 1:
                agg_results = self.stored_gradients[ridx][2]
            else:
                localUpdates = dict()
                p_k = dict()
                for player_idx in player_idxs:
                    localUpdates[player_idx] = self.stored_gradients[ridx][0][player_idx]
                    p_k[player_idx] = self.stored_gradients[ridx][1][player_idx]
                # aggregation
                agg_results = self.weighted_avg(localUpdates, p_k)

            global_model = self.model_initiation()
            global_model.load_state_dict(agg_results)
            if self.TSS and len(player_idxs) > 0:
                skippable_test_sample_idxs = set(range(len(self.Tst)))
                for player_idx in player_idxs:
                    skippable_test_sample_idxs  \
                        = skippable_test_sample_idxs & self.skippable_test_sample[ridx][player_idx]
                complete_idx = set(range(len(self.Tst)))
                testData = copy.deepcopy(self.Tst)
                testData.idxs = list(complete_idx - skippable_test_sample_idxs)
            else:
                testData = self.Tst
            utility = DNNTest(global_model, testData)
            if self.TSS and len(player_idxs) > 0:
                del testData
            del global_model
            torch.cuda.empty_cache()
            return utility
        
        
        # model initialize and training
        # FedAvg
        global_model = self.model_initiation()
        if len(player_idxs) > 0:
            for ridx in range(self.max_round):
                localUpdates = dict()
                p_k = dict()
                for player_idx in player_idxs:
                    # local model training
                    localUpdates[player_idx] = self.players[player_idx][ridx]
                    p_k[player_idx] = self.player_data_size[player_idx]
                # aggregation
                agg_results = self.weighted_avg(localUpdates, p_k)
                global_model.load_state_dict(agg_results)
            del localUpdates
        testData = self.Tst
        utility = DNNTest(global_model, testData)
        del global_model, testData
        torch.cuda.empty_cache()
        return utility
