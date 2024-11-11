import numpy as np
from torchvision import datasets, transforms
import torch  # os,
# from .text_helper import Corpus, centralized
# from .image_helper import FEMNIST
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from .dataset_helper import ImageDataset

import pandas as pd
from sklearn.preprocessing import LabelEncoder, minmax_scale


def load_AdultData(file_path):
    column_names = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country",
        "income"
    ]
    data = pd.read_csv(file_path, names=column_names,
                       na_values=" ?", skipinitialspace=True)
    data.head()
    data = preprocess_AdultData(data)
    return data


def preprocess_AdultData(data):
    data = data.dropna()  # 删除缺失值
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])
    return data


def get_datasets(args):
    # load datasets
    if args.dataset == 'mnist':
        img_size = torch.Size([args.num_channels, 28, 28])

        trans_mnist = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST(
            root='./data/mnist/',
            train=True,
            download=True,
            transform=trans_mnist
        )

        dataset_test = datasets.MNIST(
            root='./data/mnist/',
            train=False,
            download=True,
            transform=trans_mnist
        )

    elif args.dataset == 'cifar':
        img_size = torch.Size([3, 32, 32])

        trans_cifar10_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
        trans_cifar10_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

        dataset_train = datasets.CIFAR10(
            root='./data/cifar',
            train=True,
            download=True,
            transform=trans_cifar10_train
        )

        dataset_test = datasets.CIFAR10(
            root='./data/cifar',
            train=False,
            download=True,
            transform=trans_cifar10_val
        )
    elif args.dataset == 'iris':
        iris = load_iris()
        x = iris.data
        y = iris.target
        # Split dataset
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2)

        x_train = torch.FloatTensor(x_train)
        x_test = torch.FloatTensor(x_test)
        y_train = torch.LongTensor(y_train)
        y_test = torch.LongTensor(y_test)

        dataset_train = ImageDataset(x_train, y_train,
                                     len(x_train), range(len(x_train)))
        dataset_test = ImageDataset(x_test, y_test,
                                    len(x_test), range(len(x_test)))
        # img_size = x_train[0].shape
        print('iris train data shape:', x_train.shape)
        print('iris test data shape:', x_test.shape)
        print('iris labels:', set(y_train.numpy().tolist()))
    elif args.dataset == 'wine':
        wine = load_wine()
        x = wine.data
        y = wine.target
        # Split dataset
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2)

        x_train = torch.FloatTensor(x_train)
        x_test = torch.FloatTensor(x_test)
        y_train = torch.LongTensor(y_train)
        y_test = torch.LongTensor(y_test)

        dataset_train = ImageDataset(x_train, y_train,
                                     len(x_train), range(len(x_train)))
        dataset_test = ImageDataset(x_test, y_test,
                                    len(x_test), range(len(x_test)))
        # img_size = x_train[0].shape
        print('wine train data shape:', x_train.shape)
        print('wine test data shape:', x_test.shape)
        print('wine labels:', set(y_train.numpy().tolist()))
    elif args.dataset == 'adult':
        # Dataset is acquired from the following link:
        # https://archive.ics.uci.edu/ml/datasets/adult
        train = load_AdultData('./data/adult/adult_data.csv')
        X_train = train.drop('income', axis=1).to_numpy()
        # normalize
        X_train = minmax_scale(X_train, feature_range=(0, 1))
        y_train = train['income'].to_numpy()
        dataset_train = ImageDataset(X_train, y_train,
                                     len(X_train), range(len(X_train)))

        test = load_AdultData('./data/adult/adult_test.csv')
        X_test = test.drop('income', axis=1).to_numpy()
        # normalize
        X_test = minmax_scale(X_test, feature_range=(0, 1))
        y_test = test['income'].to_numpy()
        test_idxs1 = np.random.choice(np.where(y_test == 0)[0],
                                      200, replace=False).tolist()
        test_idxs2 = np.random.choice(np.where(y_test == 1)[0],
                                      200, replace=False).tolist()
        test_idxs = test_idxs1+test_idxs2
        print('selected_idx:', test_idxs, X_test[test_idxs, :].shape)
        dataset_test = ImageDataset(X_test[test_idxs, :], y_test[test_idxs],
                                    len(test_idxs), range(len(test_idxs)))

        print('adult train data shape:', dataset_train.dataset.shape)
        print('adult train data sample:', X_train[0])
        print('adult test data shape:', dataset_test.dataset.shape)
        print('adult test data sample:', dataset_test.dataset[0])
        print('adult labels:', set(dataset_test.labels))
    else:
        exit('Error: unrecognized dataset')
    return dataset_train, dataset_test


def data_split(args):
    # load datasets
    dataset_train, dataset_test = get_datasets(args)
    validation_index = []  # np.random.choice(
    # len(dataset_test),int(len(dataset_test)*0.05), replace=False
    # )

    # sampling
    if args.data_allocation == 0 or args.num_trainDatasets == 1:
        data_size_group = 1
        data_size_means = [len(dataset_train)/args.num_trainDatasets]
        group_size = [args.num_trainDatasets]
    else:
        args.group_size = [int(tmp) for tmp in args.group_size.split(",")]
        args.multiplier = [int(tmp) for tmp in args.multiplier.split(",")]

        data_size_group = args.data_size_group
        group_size = args.group_size
        data_size_means = [args.data_size_mean*args.multiplier[gidx]
                           for gidx in range(data_size_group)]

    data_quantity = []
    for i in range(data_size_group):
        tmp = np.random.normal(data_size_means[i],
                               (0 if args.data_allocation == 0 or args.num_trainDatasets == 1
                                else data_size_means[i]/4),
                               group_size[i])
        tmp2 = []
        small_index = np.where(tmp <= data_size_means[i])[0]
        if len(small_index) >= group_size[i]/2:
            tmp2 += list(tmp[small_index][:int(group_size[i]/2)])
            tmp2 += list(2*data_size_means[i] -
                         tmp[small_index][:group_size[i]-int(group_size[i]/2)])
        else:
            large_index = np.where(tmp >= data_size_means[i])[0]
            tmp2 += list(tmp[large_index][:int(group_size[i]/2)])
            tmp2 += list(2*data_size_means[i] -
                         tmp[large_index][:group_size[i]-int(group_size[i]/2)])
        # tmp2 = tmp2[:group_size[i]]
        if len(tmp2) < group_size[i]:
            tmp2 += tmp2 + tmp2[int(group_size[i]/2):
                                int(group_size[i]/2)+(group_size[i]-len(tmp2))]
        data_quantity += tmp2
    data_quantity = np.array([(int(np.round(i)) if int(np.round(i)) >= 2 else 2)
                              for i in data_quantity])
    data_quantity = sorted(data_quantity)
    print(data_quantity)
    if len(group_size) <= 1:
        data_idx = list(range(sum(data_quantity)))
        # print(data_idx)
        np.random.shuffle(data_idx)
        workers_idxs = [[] for _ in range(args.num_trainDatasets)]
        for idx in range(args.num_trainDatasets):
            print('sampling worker %s...' % idx)
            workers_idxs[idx] = np.random.choice(data_idx,
                                                 data_quantity[idx], replace=False)
            data_idx = list(set(data_idx)-set(workers_idxs[idx]))
            np.random.shuffle(data_idx)
    else:
        try:
            idxs_labels = np.array(dataset_train.train_labels)
        except:
            try:
                idxs_labels = np.array(dataset_train.targets)
            except:
                idxs_labels = np.array(dataset_train.labels)

        class_num = dict([(c, 0) for c in range(args.num_classes)])
        worker_classes = dict()
        for idx in range(args.num_trainDatasets):
            worker_classes[idx] = range(args.num_classes)
            for tmp, c in enumerate(worker_classes[idx]):
                if tmp == len(worker_classes[idx])-1:
                    class_num[c] += data_quantity[idx] - \
                        int(data_quantity[idx]/len(worker_classes[idx])
                            )*(len(worker_classes[idx])-1)
                else:
                    class_num[c] += int(data_quantity[idx] /
                                        len(worker_classes[idx]))

        class_indexes = dict()
        for c, num in class_num.items():
            original_index = list(np.where(idxs_labels == c)[0])
            appended_index = []
            count = 0
            while len(appended_index) < num:
                appended_index += [tmp+count *
                                   len(idxs_labels) for tmp in original_index]
                count += 1
            np.random.shuffle(appended_index)
            class_indexes[c] = appended_index

        workers_idxs = [[] for _ in range(args.num_trainDatasets)]
        for idx in range(args.num_trainDatasets):
            print('sampling worker %s...' % idx)
            workers_idxs[idx] = []
            for tmp, c in enumerate(worker_classes[idx]):
                if tmp == len(worker_classes[idx])-1:
                    sampled_idx = list(np.random.choice(
                        class_indexes[c],
                        data_quantity[idx] -
                        int(data_quantity[idx]/len(worker_classes[idx])
                            )*(len(worker_classes[idx])-1),
                        replace=False))
                else:
                    sampled_idx = list(np.random.choice(
                        class_indexes[c],
                        int(data_quantity[idx]/len(worker_classes[idx])),
                        replace=False))
                workers_idxs[idx] += sampled_idx
                class_indexes[c] = list(set(class_indexes[c])-set(sampled_idx))
                np.random.shuffle(class_indexes[c])
            np.random.shuffle(workers_idxs[idx])
            print(data_quantity[idx], len(workers_idxs[idx]),
                  worker_classes[idx], set([idxs_labels[tmp % len(idxs_labels)] for tmp in workers_idxs[idx]]))

    dict_workers = {i: workers_idxs[i] for i in range(len(workers_idxs))}
    x = []
    combine = []
    for i in dict_workers.values():
        x.append(len(i))
        combine.append(len(i))
    print('train data partition')
    print('sum:', np.sum(np.array(x)))
    print('mean:', np.mean(np.array(x)))
    print('std:', np.std(np.array(x)))
    print('max:', max(np.array(x)))
    print('min:', min(np.array(x)))
    return dataset_train, dataset_test, validation_index, dict_workers
