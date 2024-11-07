# Examples of Using SV4DA

The current folder `examples` contains sample codes and datasets related to using **SV4DA** for actual DA tasks. This README document will describe the specific usage examples provided by SV4DA and how users can run these examples themselves.

There are three specific tasks provided by **SV4DA** as examples. These tasks are as follows:

## Data Valuation(DV) Task

The DV task utilizes SV for data tuple pricing and selection, with the player being the data tuple and the utility referring to the test accuracy of the ML model. The DV experiments are conducted on *Iris* and *Adult*. For both two datasets, the task randomly splits data samples in the dataset into two portions, one with 80% samples for model training and another with the rest 20% samples for model testing. The linear model is chosen as the model learned in the task.

### Data Preparation

Firstly, perform the data preparation operation for two datasets, which divides training dataset and test dataset for the ratio of $80\%/20\%$ by default.

```sh
python -u data_preparation.py --dataset=iris --num_classes=3 --num_trainDatasets=1 --data_allocation=0  --data_size_group=1 --group_size=1 

python -u data_preparation.py --dataset=adult --num_classes=2 --num_trainDatasets=1 --data_allocation=0  --data_size_group=1 --group_size=1 
```

### Task Running



```sh
nohup python -u data_valuation.py --task=DV --model_name=Linear --ep=30 --bs=16 --lr=0.01 --dataset=iris --num_class=3 --num_feature=4 --data_allocation=0 --test_metric=tst_accuracy --method=MC --sampling_strategy=random --convergence_threshold=0.01 --manual_seed=42 --num_parallelThreads=20 --log_file=logs/DV_Iris_MC.log > logs/DV_Iris_MC.log  &

nohup python -u data_valuation.py --task=DV --model_name=Linear --ep=30 --bs=16 --lr=0.01 --dataset=iris --num_class=3 --num_feature=4 --data_allocation=0 --test_metric=tst_accuracy --method=MC --sampling_strategy=random --convergence_threshold=0.01 --tuple_to_set=12 --manual_seed=42 --num_parallelThreads=20 --log_file=logs/DSV_Iris_MC.log > logs/DSV_Iris_MC.log &  

nohup python -u data_valuation.py --task=DV --model_name=Linear --ep=30 --bs=16 --lr=0.01 --dataset=iris --num_class=3 --num_feature=4 --data_allocation=0 --test_metric=tst_accuracy --method=classical --sampling_strategy=random --convergence_threshold=0.01 --truncation=True --truncationThreshold=0.01  --gradient_approximation=True --manual_seed=42 --num_parallelThreads=20 --log_file=logs/DV_Iris_TGMC.log > logs/DV_Iris_TGMC.log &  

nohup python -u data_valuation.py --task=DV --model_name=Linear --ep=30 --bs=16 --lr=0.01 --dataset=iris --num_class=3 --num_feature=4 --data_allocation=0 --test_metric=tst_accuracy --method=CP --num_measurement=100 --CP_epsilon=0.01 --convergence_threshold=0.01 --sampling_strategy=random --manual_seed=42 --num_parallelThreads=20 --log_file=logs/DV_Iris_CP.log > logs/DV_Iris_CP.log &  

nohup python -u data_valuation.py --task=DV --model_name=Linear --ep=30 --bs=16 --lr=0.01 --dataset=iris --num_class=3 --num_feature=4 --data_allocation=0 --test_metric=tst_accuracy --method=MLE --MLE_maxInterval=15000 --convergence_threshold=0.01 --sampling_strategy=random  --manual_seed=42 --num_parallelThreads=20 --log_file=logs/DV_Iris_MLE.log > logs/DV_Iris_MLE.log & 

nohup python -u data_valuation.py --task=DV --model_name=Linear --ep=30 --bs=16 --lr=0.01 --dataset=iris --num_class=3 --num_feature=4 --data_allocation=0 --test_metric=tst_accuracy --method=GT --GT_epsilon=0.00001 --convergence_threshold=0.01 --sampling_strategy=random --manual_seed=42 --num_parallelThreads=20 --log_file=logs/DV_Iris_GT.log > logs/DV_Iris_GT_.log &  

```



## Federated Learning(FL) Task

## Feature Attribution(FA) Task