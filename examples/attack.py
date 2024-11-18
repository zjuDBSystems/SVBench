# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 09:34:10 2024

@author: admin
"""
from arguments import args_parser
import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score,accuracy_score
import sys, torch, math, os, time
from models.Nets import LinearAttackModel
from utils.dataset_helper import ImageDataset 
from ML_utils import DNNTrain, DNNTest
import torch.nn.functional as F
from Privacy_utils import privacy_protect


def combination(n, k):
    """计算组合数 C(n, k)"""
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))
 
def permutation(n, k):
    """计算排列数 P(n, k)"""
    return math.factorial(n) // math.factorial(n - k)


def SV_compute(task, sampleIdx, datasetIdx, SV_args):
    targetedPlayer_idx = len(datasetIdx)
    task.players.idxs = datasetIdx + [sampleIdx]
    print('trn data idxs:', task.players.idxs)
    sys.stdout.flush()
    SV = task.run()
    return SV[targetedPlayer_idx]

def MIA(maxIter, SV_args):
    '''
    The threat model we consider here is that 
    the attacker can compute the value of any data point among any datasets 
    (analogue to the setting of MIA against ML models where 
     the attacker can train models on any datasets he/she constructs).
    '''
    '''
    1. For each dataset we experiment on, we select 200 data points (members) 
    as the private dataset held by the central server. 
    2. We pick another 200 data points which serve as non-members. 
    3. Moreover, we leverage another 400 data points where 
    we can subsample “shadow dataset” in Algorithm 1. 
    4. We set the number of shadow datasets we sample as 32.
    5. The membership probability of every target sample 
       is determined by in_probability/out_probability;
    6. Adopting different thresholds can produce different confusion matrices 
       with different true positive rates and false positive rates, 
       which produce a ROC curve and its associated AUROC.
    '''
    #SV_args.scannedIter_maxNum=50
    task = Task(SV_args)  # DV-attack task
    dataIdx = list(task.players.idxs)
    
    shadowDataset = np.random.choice(
        dataIdx, int(len(dataIdx)/2), replace=False).tolist()
    QueryDataset = list(set(dataIdx)-set(shadowDataset))
    inMemberDataset = np.random.choice(
        QueryDataset, int(len(QueryDataset)/2), replace=False).tolist()
    print('shadowDataset: ',shadowDataset)
    print('QueryDataset: ',QueryDataset)
    print('inMemberDataset: ',inMemberDataset)
    
    queryResults = []
    queryLabels = []
    numSamples_in_Shadow = max(int(len(shadowDataset)/4), 
                               int(len(shadowDataset)/maxIter))
    for qidx, z in enumerate(QueryDataset):
        SV_in = []
        SV_out = []
        sampledShadowDataset = shadowDataset[:numSamples_in_Shadow]
        exclude_list = []
        for iter_ in range(maxIter):
            while "".join(map(str,sampledShadowDataset)) in exclude_list:
                sampledShadowDataset = np.random.choice(
                    shadowDataset, numSamples_in_Shadow, 
                    replace=False).tolist()
            exclude_list.append("".join(map(str,sampledShadowDataset)))
            
            SV_out.append(SV_compute(task, z, sampledShadowDataset, SV_args))
            sampledShadowDataset.append(z)
            SV_in.append(SV_compute(task, z, sampledShadowDataset, SV_args))
            sampledShadowDataset = sampledShadowDataset[:-1]
            print('Query sample %s/%s Iter %s/%s: SV_out %s, SV_in: %s'%(
                qidx, len(QueryDataset), iter_, maxIter,
                SV_out[-1], SV_in[-1]))
            if len(exclude_list) >= combination(
                    len(shadowDataset), numSamples_in_Shadow):
                break
            
        in_mean = np.mean(SV_in)
        out_mean = np.mean(SV_out)
        in_std = np.std(SV_in)
        if in_std ==0:
            in_std = 10**(-20)
        out_std = np.std(SV_out)
        if out_std ==0:
            out_std = 10**(-20)
        value = SV_compute(task, z, inMemberDataset, SV_args)
        
        in_probability = norm.pdf(value, loc=in_mean, scale=in_std)
        out_probability = norm.pdf(value, loc=out_mean, scale=out_std)
        if out_probability ==0:
            out_probability = 10**(-20)
        queryResults.append(in_probability/out_probability)
        queryLabels.append(int(z in inMemberDataset))
        print('Query sample %s/%s result: '%(qidx, len(QueryDataset)),
              value, queryLabels[-1], queryResults[-1], 
              in_probability, out_probability)
        
    try:
        AUROC = roc_auc_score(queryLabels, queryResults)
    except Exception as e :
        print(e)
        AUROC = 0.5
    print('MIA AUROC: ', AUROC)
    ACC = accuracy_score(
        queryLabels, [int(r>1) for r in queryResults])
    print('MIA ACC: ', ACC)
    return AUROC

def FIA(SV_args):
    # Algorithm 1: The Attack with an Auxiliary Dataset
    # 60% training set, 20% test set (auxiliary set for training MLP), 
    # 20% validation set (query set producing MAE)
    SV_args.dataNormalize = True
    #SV_args.scannedIter_maxNum=50
    task = Task(SV_args) # FA task
    auxiliary_index = np.random.choice(range(len(task.Tst)),
                                       int(len(task.Tst)/2),
                                       replace = False).tolist()
    validation_index = list(set(range(len(task.Tst)))-set(auxiliary_index))
    
    #compute SV
    task.run()
    auxiliary_SV = [list(task.testSampleFeatureSV[test_idx].values()) \
                    for test_idx in auxiliary_index]
    validation_SV = [list(task.testSampleFeatureSV[test_idx].values()) \
                     for test_idx in validation_index]
    
    auxiliary_data = ImageDataset(
        torch.FloatTensor(auxiliary_SV), 
        torch.FloatTensor(task.X_test[auxiliary_index]), 
        len(auxiliary_index), range(len(auxiliary_index)))
    validation_data = ImageDataset(
        torch.FloatTensor(validation_SV), 
        torch.FloatTensor(task.X_test[validation_index]), 
        len(validation_index), range(len(validation_index)))
    
    attackModel = LinearAttackModel(len(task.players))
    print('attackModel: ', attackModel)
    attackModel = DNNTrain(attackModel, auxiliary_data, 
                           epoch=100, batch_size=10, lr=0.005, 
                           loss_func=torch.nn.L1Loss(),
                           momentum=0, weight_decay=0, max_norm=0, 
                           validation_metric='tst_MAE', loss_print=True)
    
    MAE = DNNTest(attackModel, validation_data, test_bs=128, metric='tst_MAE',
                  pred_print=True) # mean absolute error
    print('FIA MAE: ', MAE)
    return MAE

def FIA_noAttackModelTrain(SV_args, random_mode='auxilliary'):
    # Algorithm 2: The Attack without an Auxiliary Dataset
    # M_c=30, tau=0.4, r=max S- min S, gagy=r/5
    #SV_args.scannedIter_maxNum=50
    SV_args.dataNormalize = True
    task = Task(SV_args) # FA task
    auxiliary_index = np.random.choice(range(len(task.Tst)),
                                       int(len(task.Tst)/2),
                                       replace = False).tolist()
    validation_index = list(set(range(len(task.Tst)))-set(auxiliary_index))
    # generate random data
    if random_mode == 'uniform':
        print('replace %s with random samples (uniform)...'%auxiliary_index)
        if 'torch' in str(type(task.model)):
            task.Trn.dataset[auxiliary_index] = torch.FloatTensor(
                np.random.rand(*task.Trn.dataset[auxiliary_index].shape))
        else:
            task.X_test[auxiliary_index] = np.random.rand(
                *task.X_test[auxiliary_index].shape)
    elif random_mode == 'normal':
        print('replace %s with random samples (normal)...'%auxiliary_index)
        if 'torch' in str(type(task.model)):
            task.Trn.dataset[auxiliary_index] = torch.FloatTensor(
                np.random.normal(
                    0.5,0.25, task.Trn.dataset[auxiliary_index].shape))
        else:
            task.X_test[auxiliary_index] = np.random.normal(
                0.5,0.25,task.X_test[auxiliary_index].shape)
    else:
        # use true data samples as the randomly-generated data sample
        pass
    task.randomSet = auxiliary_index
    print('validation_index:', validation_index, '\n',
          'auxiliary_index', auxiliary_index)
    
    # compute SV
    task.run()
    auxiliary_SV = [list(task.testSampleFeatureSV[test_idx].values()) \
                    for test_idx in auxiliary_index]
    validation_SV = [list(task.testSampleFeatureSV[test_idx].values()) \
                     for test_idx in validation_index]
        
    # start inference
    m_c=30
    r=np.array(auxiliary_SV).max()-np.array(auxiliary_SV).min()
    gagy = r/100
    tau=0.4
    print('number of reference samples:', m_c)
    print('threshold for feature SV difference:',r, gagy)
    print('threshold for feature max diff in references:',tau)
    predictions = np.zeros(task.X_test[validation_index].shape)
    num_unsuccess = 0
    for validation_data_idx in range(len(validation_SV)):
        for feature_idx in range(len(validation_SV[validation_data_idx])):
            
            diff_reference = dict()
            for reference_data_idx  in range(len(auxiliary_SV)):
                diff_reference[reference_data_idx]=np.abs(
                    validation_SV[validation_data_idx][feature_idx]-\
                    auxiliary_SV[reference_data_idx][feature_idx])
            diff_reference = dict(sorted(diff_reference.items(), 
                                         key=lambda item: item[1]))
            references = []
            for item  in diff_reference.items():
                if len(references)<m_c or item[1] < gagy:
                    references.append(
                        task.X_test[auxiliary_index[item[0]],feature_idx])
            if max(references)-min(references)>tau:
                #not to predict
                num_unsuccess += 1
                #print(validation_data_idx, feature_idx, ":", references)
            #else:
            predictions[validation_data_idx, feature_idx]=np.mean(references)
    MAE = F.l1_loss(torch.FloatTensor(predictions), 
                    torch.FloatTensor(task.X_test[validation_index]))
    print('FIA MAE: ', MAE)
    print('FIA SR: ', num_unsuccess, (validation_data_idx+1)*(feature_idx+1),
          1-num_unsuccess/((validation_data_idx+1)*(feature_idx+1)))
    return MAE



if __name__ == '__main__':    
    args = args_parser()
    if args.log_file!='':
        old_stdout = sys.stdout
        file = open(args.log_file, 'w')
        sys.stdout = file
    print('Experiment arguemtns: ', args)
    
    if args.task == 'DV':
        from data_valuation import Task 
    elif args.task == 'FA':
        from feature_attribution import Task 
    else:
        print('The task name may be wrong or ',
              'has not yet been considered in attack experiments!')
        sys.exit() 
        
    if args.attack_type == 'MIA':
        MIA(args.maxIter_in_MIA, args)
    elif args.attack_type == 'FIA':
        FIA(args)
    elif args.attack_type == 'FIA_withoutAuxilliary_uniform':
        FIA_noAttackModelTrain(args, random_mode='uniform')
    elif args.attack_type =='FIA_withoutAuxilliary_normal':
        FIA_noAttackModelTrain(args, random_mode='normal')
    elif args.attack_type =='FIA_noAttackModel':
        FIA_noAttackModelTrain(args, random_mode='auxilliary')
    else:
        print('The attack name may be wrong or ',
              'has not yet been considered in attack experiments!')
        sys.exit() 
    
    # Task terminated!
    sys.stdout.flush()
    if args.log_file!='':
        sys.stdout = old_stdout 
        file.close()
    sys.exit() 
