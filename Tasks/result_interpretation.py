import numpy as np
import queue
import os
import copy
import torch
import joblib
import pandas as pd

from functools import reduce
from sklearn.tree import DecisionTreeRegressor

from .data_preparation import data_prepare
from .nets import NN, RegressionModel
from .utils import DNNTrain, DNNTest, find_free_device


class RI():
    def __init__(self, dataset, manual_seed):
        self.dataset = dataset
        self.manual_seed = manual_seed
        self.dataset_info = {
            'iris': (3, 4, 0.01, 50, 16),
            'wine': (3, 13, 0.001, 100, 16),
            'adult': (2, 14, 0.01, 100, 64),
            '2dplanes': (2, 10, 0.01, 100, 64),
            'ttt': (2, 9, 0.001, 50, 16)
        }
        self.num_classes, self.num_feature, self.lr, self.ep, self.bs = self.dataset_info[
            dataset]

        self.device = find_free_device()
        self.model = None

        print("Data preprocessing...")
        # if dataset == 'adult':
        #     self.X_train, X_test, self.y_train, y_test = data_prepare(
        #         manual_seed=manual_seed, dataset=dataset,num_classes=self.num_classes)
        #     self.X_train = self.X_train.to_numpy()
        #     self.y_train = self.y_train.to_numpy()
        #     baseline_idxs = []
        #     for label in np.unique(self.y_train):
        #         class_idx = np.where(self.y_train == label)[0]
        #         class_data = self.X_train[class_idx]
        #         intra_class_dis = [np.mean(
        #             np.linalg.norm(
        #                 class_data - class_data[data_idx], ord=1, axis=1)
        #         ) for data_idx in range(len(class_data))]
        #         baseline_idxs += class_idx[np.argsort(intra_class_dis)
        #                                 ][:5].tolist()
        #     self.initial_feature_baseline = copy.deepcopy(self.X_train)[
        #         baseline_idxs]
        #     self.feature_baseline = copy.deepcopy(self.X_train)[baseline_idxs]
        # else:
        trn_path = 'data/%s0/train0.pt' % (dataset)
        if not os.path.exists(trn_path):
            data_prepare(manual_seed=manual_seed, dataset=dataset,
                        num_classes=self.num_classes)

        self.Trn = torch.load(trn_path)

        self.X_train = []
        self.y_train = []
        for item in self.Trn:
            data, label = item[0], item[1]
            if type(data) != np.ndarray:
                data = data.numpy()
            self.X_train.append(data.reshape(-1))
            self.y_train.append(label)
        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)

        baseline_idxs = []
        for label in np.unique(self.y_train):
            class_idx = np.where(self.y_train == label)[0]
            class_data = self.X_train[class_idx]
            intra_class_dis = [np.mean(
                np.linalg.norm(
                    class_data - class_data[data_idx], ord=1, axis=1)
            ) for data_idx in range(len(class_data))]
            baseline_idxs += class_idx[np.argsort(intra_class_dis)
                                    ][:5].tolist()
        self.initial_feature_baseline = copy.deepcopy(self.X_train)[
            baseline_idxs]
        self.feature_baseline = copy.deepcopy(self.X_train)[baseline_idxs]

        self.X_test = []
        self.y_test = []
        # if dataset == 'adult':
        #     # 遍历 X_test 和 y_test 并进行处理
        #     self.Tst = copy.deepcopy(pd.concat([X_test, y_test], axis=1))
        #     self.complete_Tst_idx = copy.deepcopy(self.Tst.index.to_numpy())
        #     self.complete_Tst = copy.deepcopy(self.Tst)
        #     for (_, row_X), (_, row_y) in zip(X_test.iterrows(), y_test.items()):
        #         data = row_X.to_numpy()  # 转换为 NumPy 数组
        #         label = np.array([row_y])  # 转换为 NumPy 数组（注意要用列表包裹，否则是一维的）
        #         self.X_test.append(data.reshape(-1))  # 展平数据
        #         self.y_test.append(label)

        #     # 转换为 NumPy 数组
        #     self.X_test = np.array(self.X_test)
        #     self.complete_X_test = copy.deepcopy(self.X_test)  # 备份完整测试数据
        #     self.y_test = np.array(self.y_test)
        #     self.complete_y_test = copy.deepcopy(self.y_test)
        #     self.players = [feature_idx 
        #                     for feature_idx in range(self.X_test[0].reshape(-1).shape[-1])]
        # else:
        self.Tst = torch.load('data/%s0/test.pt' % (dataset))
        self.complete_Tst_idx = copy.deepcopy(self.Tst.idxs)
        for item in self.Tst:
            data, label = item[0], item[1]
            if type(data) != np.ndarray:
                data = data.numpy()
            self.X_test.append(data.reshape(-1))
            self.y_test.append(label)
        self.X_test = np.array(self.X_test)
        self.complete_X_test = copy.deepcopy(self.X_test)
        self.y_test = np.array(self.y_test)
        self.complete_y_test = copy.deepcopy(self.y_test)

        # player setting
        self.players = [feature_idx
                        for feature_idx in range(self.Tst[0][0].reshape(-1).shape[-1])]

        # prepare trained model
        self.train_model()

        # used only by FIA
        self.randomSet = []
        self.testSampleFeatureSV = dict()
        self.testSampleFeatureSV_var = dict()

        # select test samples
        self.num_samples_each_class = 2
        self.selected_test_samples = []
        '''
        if len(self.players) > 5 and \
            False not in [len(
                np.where(self.complete_y_test == label)[0]
            ) > self.num_samples_each_class
                for label in np.unique(self.complete_y_test)]:
            for label in np.unique(self.complete_y_test):
                self.selected_test_samples += np.where(
                    self.complete_y_test == label)[0][:self.num_samples_each_class].tolist()
        else:
        '''
        if len(self.complete_y_test) > self.num_samples_each_class * \
                                       len(np.unique(self.complete_y_test)):
            for label in np.unique(self.complete_y_test):
                self.selected_test_samples += np.where(
                    self.complete_y_test == label)[0][:self.num_samples_each_class].tolist()
        else:
            self.selected_test_samples = range(len(self.complete_X_test))
        print('selected_test_samples: ', self.selected_test_samples)

    def train_model(self):
        if not os.path.exists('models/'):
            os.mkdir('models/')
            
        model_path = 'models/attribution_RI-%s.pt' % (
            (self.dataset if self.manual_seed == 42
             else (self.dataset+"-"+str(self.manual_seed)))
        )
        if os.path.exists(model_path):
            # if self.dataset == 'adult':
            #     self.model = joblib.load(model_path)
            #     testData = self.Tst
            #     print('Given model accuracy: ', AdultTest(self.model, testData))
            #     return
            # else:
            self.model = torch.load(model_path, map_location=self.device)
            testData = self.Tst
            print('Given model accuracy: ',
                DNNTest(model=self.model, test_data=testData,
                        pred_print=True)
                )
            return

        if self.dataset in ['wine', 'adult', 'ttt']:
            self.model = NN(num_feature=self.num_feature,
                            num_classes=self.num_classes)
        # elif self.dataset == 'adult':
        #     self.model = DecisionTreeRegressor(random_state=42, max_depth=5)  # 初始化一个决策树分类器模型。
        else:
            self.model = RegressionModel(
                num_feature=self.num_feature, num_classes=self.num_classes)
        
        # if self.dataset == 'adult':
        #     self.model = AdultTrain(self.model, self.X_train, self.y_train)
        #     print('Given model accuracy(r2_score): ',AdultTest(self.model, self.Tst))
        #     joblib.dump(self.model, model_path)
        # else:
        loss_func = torch.nn.CrossEntropyLoss()
        self.model = DNNTrain(model=self.model, trn_data=self.Trn,
                            lr=self.lr, epoch=self.ep, batch_size=self.bs,
                            loss_func=loss_func)
        print('Given model accuracy: ',
            DNNTest(model=self.model, test_data=self.Tst,
                    pred_print=True)
            )
        torch.save(self.model, model_path)

    def torch_predict(self, replace_idxs, replace_val, results):
        Tst = copy.deepcopy(self.Tst)
        ori_shape = Tst.dataset.shape
        dataset = Tst.dataset.reshape((ori_shape[0], -1))
        dataset[:, replace_idxs] = torch.FloatTensor(replace_val)
        Tst.dataset = dataset.reshape(ori_shape)
        results.put(DNNTest(model=self.model,
                    test_data=Tst, metric='prediction'))
    
    def adult_predict(self, replace_idxs, replace_val, results):
        Tst = copy.deepcopy(self.Tst)
        # print("ori_tst:", Tst)
        ori_shape = Tst.to_numpy().shape
        dataset = Tst.to_numpy().reshape((ori_shape[0], -1))
        # 替换指定特征索引的值
        dataset[:, replace_idxs] = np.array(replace_val, dtype=np.float32)
        # 重新转换为 DataFrame，确保数据格式不变
        Tst = pd.DataFrame(dataset, columns=Tst.columns)
        # print("Tst:", Tst.shape, Tst)
        res = AdultTest(self.model, Tst, metric='prediction')
        # print("AdultTest:", res)
        results.put(res)

    def utility_computation(self, player_idxs):
        # print("utility_computation: ", player_idxs)
        utility = 0.0
        replace_idxs = [feature_idx for feature_idx in range(self.X_train.shape[-1])
                        if feature_idx not in player_idxs]
        baselines = [eval(baseline)
                     for baseline in set([
                         str(tmp)
                         for tmp in self.feature_baseline[:, replace_idxs].tolist()])
                     ]
        # print("baselines:", baselines)
        # model testing (maybe expedited by some ML speedup functions)
        results = queue.Queue()
        for order, replace_val in enumerate(baselines):
            # if self.dataset == 'adult':
            #     self.adult_predict(replace_idxs, replace_val, results)
            # else:
            self.torch_predict(replace_idxs, replace_val, results)
        predictions = np.sum(list(results.queue), 0)/len(baselines)
        # print("predictions:", predictions)
        utility = predictions.sum()/reduce((lambda x, y: x*y), predictions.shape)

        return utility
