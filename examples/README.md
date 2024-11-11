# Examples of Using SV4DA

The current folder `examples` contains sample codes and datasets related to using **SV4DA** for actual DA tasks. This README document will describe the specific usage examples provided by SV4DA and how users can run these examples themselves.

There are three specific tasks provided by **SV4DA** as examples. These tasks are as follows:

## Data Valuation(DV) Task

The DV task utilizes SV for data tuple pricing and selection, with the player being the data tuple and the utility referring to the test accuracy of the ML model. The DV experiments are conducted on *Iris* and *Adult*. For both two datasets, the task randomly splits data samples in the dataset into two portions, one with 80% samples for model training and another with the rest 20% samples for model testing. <u>The linear model is chosen as the model learned in the task for the *Iris* dataset, while for the *wine* dataset, the decision tree is chosen.</u> The sampling strategy for every method is random sampling.

### Data Preparation

Firstly, perform the data preparation operation for two datasets, which divides training dataset and test dataset for the ratio of $80\%/20\%$ by default. The processed training and test data are stored in the directories `examples/data/iris0` and `examples/data/adult0`.

```sh
python -u data_preparation.py --dataset=iris --num_classes=3 --num_trainDatasets=1 --data_allocation=0  --data_size_group=1 --group_size=1 

python -u data_preparation.py --dataset=wine --num_classes=3 --num_trainDatasets=1 --data_allocation=0  --data_size_group=1 --group_size=1
```

### Task Running

1.   Using `MC` method to calculate SVs for all tuples in the dataset.

     ```shell
     nohup python -u data_valuation.py --task=DV --model_name=Linear --ep=30 --bs=16 --lr=0.01 --dataset=iris --num_class=3 --num_feature=4 --data_allocation=0 --test_metric=tst_accuracy --method=MC --sampling_strategy=random --convergence_threshold=0.01 --manual_seed=42 --num_parallelThreads=20 --log_file=logs/DV_Iris_MC.log > logs/DV_Iris_MC.log  &
     
     nohup python -u data_valuation.py --task=DV --model_name=Tree --tree_maxDepth=10 --ep=30 --bs=16 --lr=0.01 --dataset=wine --num_class=3 --num_feature=4 --data_allocation=0 --test_metric=tst_accuracy --method=MC --sampling_strategy=random --convergence_threshold=0.01 --manual_seed=42 --num_parallelThreads=20 --log_file=logs/DV_wine_MC.log > logs/DV_wine_MC.log  &
     ```

2.   Using `MC` method with `truncation` and `gradient approximation` technology to calculate SVs for all tuples in the dataset.

     ```shell
     nohup python -u data_valuation.py --task=DV --model_name=Linear --ep=30 --bs=16 --lr=0.01 --dataset=iris --num_class=3 --num_feature=4 --data_allocation=0 --test_metric=tst_accuracy --method=MC --sampling_strategy=random --convergence_threshold=0.01 --truncation=True --truncationThreshold=0.01  --gradient_approximation=True --manual_seed=42 --num_parallelThreads=20 --log_file=logs/DV_Iris_TGMC.log > logs/DV_Iris_TGMC.log & 
     
     nohup python -u data_valuation.py --task=DV --model_name=Tree --tree_maxDepth=10 --ep=30 --bs=16 --lr=0.01 --dataset=wine --num_class=3 --num_feature=4 --data_allocation=0 --test_metric=tst_accuracy --method=MC --sampling_strategy=random --convergence_threshold=0.01 --truncation=True --truncationThreshold=0.01  --gradient_approximation=True --manual_seed=42 --num_parallelThreads=20 --log_file=logs/DV_wine_TGMC.log > logs/DV_wine_TGMC.log & 
     ```

3.   Using `CP` method to calculate SVs for all tuples in the dataset.

     ```sh
     nohup python -u data_valuation.py --task=DV --model_name=Linear --ep=30 --bs=16 --lr=0.01 --dataset=iris --num_class=3 --num_feature=4 --data_allocation=0 --test_metric=tst_accuracy --method=CP --num_measurement=100 --CP_epsilon=0.01 --convergence_threshold=0.01 --sampling_strategy=random --manual_seed=42 --num_parallelThreads=20 --log_file=../../logs/DV_Iris_CP.log > ../../logs/DV_Iris_CP.log &
     
     nohup python -u data_valuation.py --task=DV --model_name=Tree --tree_maxDepth=10 --ep=30 --bs=16 --lr=0.01 --dataset=wine --num_class=3 --num_feature=4 --data_allocation=0 --test_metric=tst_accuracy --method=CP --num_measurement=100 --CP_epsilon=0.01 --convergence_threshold=0.01 --sampling_strategy=random --manual_seed=42 --num_parallelThreads=20 --log_file=../../logs/DV_wine_CP.log > ../../logs/DV_wine_CP.log &
     ```

4.   Using `MLE` method to calculate SVs for all tuples in the dataset.

     ```sh
     nohup python -u data_valuation.py --task=DV --model_name=Linear --ep=30 --bs=16 --lr=0.01 --dataset=iris --num_class=3 --num_feature=4 --data_allocation=0 --test_metric=tst_accuracy --method=MLE --MLE_maxInterval=15000 --convergence_threshold=0.01 --sampling_strategy=random  --manual_seed=42 --num_parallelThreads=20 --log_file=logs/DV_Iris_MLE.log > logs/DV_Iris_MLE.log &
     
     nohup python -u data_valuation.py --task=DV --model_name=Tree --tree_maxDepth=10 --ep=30 --bs=16 --lr=0.01 --dataset=wine --num_class=3 --num_feature=4 --data_allocation=0 --test_metric=tst_accuracy --method=MLE --MLE_maxInterval=15000 --convergence_threshold=0.01 --sampling_strategy=random  --manual_seed=42 --num_parallelThreads=20 --log_file=logs/DV_wine_MLE.log > logs/DV_wine_MLE.log &
     ```

5.   Using `GT` method to calculate SVs for all tuples in the dataset.

     ```sh
     nohup python -u data_valuation.py --task=DV --model_name=Linear --ep=30 --bs=16 --lr=0.01 --dataset=iris --num_class=3 --num_feature=4 --data_allocation=0 --test_metric=tst_accuracy --method=GT --GT_epsilon=0.00001 --convergence_threshold=0.01 --sampling_strategy=random --manual_seed=42 --num_parallelThreads=20 --log_file=logs/DV_Iris_GT.log > logs/DV_Iris_GT_.log &
     
     nohup python -u data_valuation.py --task=DV --model_name=Tree --tree_maxDepth=10 --ep=30 --bs=16 --lr=0.01 --dataset=wine --num_class=3 --num_feature=4 --data_allocation=0 --test_metric=tst_accuracy --method=GT --GT_epsilon=0.00001 --convergence_threshold=0.01 --sampling_strategy=random --manual_seed=42 --num_parallelThreads=20 --log_file=logs/DV_wine_GT.log > logs/DV_wine_GT_.log &
     ```

## Federated Learning(FL) Task

In FL task, the player refers to the local model learned by each FL participant and the utility is defined as the test accuracy of the global model. 

### Data Preparation

 The FL experiments are conducted on MNIST(containing 60,000 training data and 10,000 test data) and CIFAR-10(containing 50,000 training data and 10,000 test data) with the small-size CNN models. The training samples in each dataset are distributed over 10 participants uniformly at random. We adopt the same way as GTG-Shapley to divide the MNIST dataset into a training dataset containing 54, 210 samples and a test dataset containing 8, 920 samples.

```shell
# split mnist dataset into 10 shards with the average number of samples in each shard equal to data_size_mean*multiplier = 500
python -u data_preparation.py --dataset=mnist --num_classes=10 --num_trainDatasets=10 --data_allocation=1 --data_size_group=1 --group_size=10 --data_size_mean=100  --multiplier=5

python -u data_preparation.py --dataset=cifar --num_classes=10 --num_trainDatasets=10 --data_allocation=1 --data_size_group=1 --group_size=10 --data_size_mean=100  --multiplier=5
```



## Feature Attribution(FA) Task