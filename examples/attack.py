# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 09:34:10 2024

@author: admin
"""
from arguments import args_parser
import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score,accuracy_score
from SV import Shapley
import sys, torch, math, os
from models.Nets import LinearAttackModel
from utils.dataset_helper import ImageDataset 
from ML_utils import DNNTrain, DNNTest
import torch.nn.functional as F
from Privacy_utils import privacy_protect

def SV_compute(task, sampleIdx, datasetIdx, SV_args):
    targetedPlayer_idx = len(datasetIdx)
    task.players.idxs = datasetIdx + [sampleIdx]
    SVtask = Shapley(players = task.players, 
                     taskUtilityFunc = task.utilityComputation, 
                     args = SV_args)#,
                     #targetedPlayer_idx = [targetedPlayer_idx])
    SVtask.CalSV()
    return SVtask.SV[targetedPlayer_idx]

def combination(n, k):
    """计算组合数 C(n, k)"""
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))
 
def permutation(n, k):
    """计算排列数 P(n, k)"""
    return math.factorial(n) // math.factorial(n - k)

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
    task = Task(SV_args)  # DV task
    dataIdx = list(task.players.idxs)
    
    replace_list = ['_recovery',
                    'withDP%s'%SV_args.privacy_protection_level,
                    'withQT%s'%SV_args.privacy_protection_level,
                    'withDR%s'%SV_args.privacy_protection_level,
                    '_seed10','_seed1']
    log_file = SV_args.log_file
    for replace_str in replace_list:
        log_file = log_file.replace(replace_str,"")
    if os.path.exists(log_file):
        SV_var = dict()
        Current_SV = []
        if os.path.exists(log_file):
            with open(log_file, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    if 'Current SV' in line:
                        Current_SV.append(eval(line.split('Current SV:')[-1]))
                    if 'SV of test sample' in line:
                        sample_idx=int(line.split("/")[0].split(" ")[-1])
                        testSampleFeatureSV[sample_idx] = eval("{"+line.split("{")[-1])
                        SV_var[sample_idx] = dict([
                            (feature_idx, \
                             np.var([sv[feature_idx] for sv in Current_SV])) \
                            for feature_idx in testSampleFeatureSV[sample_idx].keys()
                            ])
                        Current_SV=[]
        with open(log_file, 'r') as file:
            lines = file.readlines()
            querySampleSVin = dict()
            querySampleSVin_ref = dict()
            querySampleSVout = dict()
            querySampleSVout_ref = dict()
            queryValue = dict()
            queryValue_ref = dict()
            queryLabel = dict()
            resultantSV = []
            SV_var = dict()
            Current_SV = []
            for line in lines:
                ''' 
                if 'shadowDataset' in line:
                    shadowDataset = eval(line.split(":")[-1])
                elif 'QueryDataset' in line:
                    QueryDataset = eval(line.split(":")[-1])
                elif 'inMemberDataset' in line:
                    inMemberDataset = eval(line.split(":")[-1])
                el
                '''
                if 'Current SV' in line:
                    Current_SV.append(eval(line.split('Current SV:')[-1]))
                if 'Final Resultant SVs:' in line:
                    resultantSV.append(
                        eval(line.split('Final Resultant SVs:')[-1]))
                    
                    SV_var[len(resultantSV)-1] = dict([
                        (player_idx, \
                         np.var([sv[player_idx] for sv in Current_SV])) \
                        for player_idx in resultantSV[-1].keys()
                        ])
                    Current_SV=[]
                    
                elif 'SV_out' in line:
                    sample_idx=int(line.split("/")[0].split(" ")[-1])
                    SV_out = eval(line.split("SV_out")[-1].split(",")[0])
                    SV_in = eval(line.split(":")[-1])
                    
                    if sample_idx in querySampleSVout.keys():
                        querySampleSVout[sample_idx].append(SV_out)
                    else:
                        querySampleSVout[sample_idx] = [SV_out]
                    if sample_idx in querySampleSVout_ref.keys():
                        querySampleSVout_ref[sample_idx][
                            len(querySampleSVout[sample_idx])-1]=SV_var[0]
                        #resultantSV[0]
                            
                    else:
                        querySampleSVout_ref[sample_idx]={
                            len(querySampleSVout[sample_idx])-1:SV_var[0]
                            }
                    
                    if sample_idx in querySampleSVin.keys():
                        querySampleSVin[sample_idx].append(SV_in)
                    else:
                        querySampleSVin[sample_idx] = [SV_in]
                    if sample_idx in querySampleSVin_ref.keys():
                        querySampleSVin_ref[sample_idx][
                            len(querySampleSVout[sample_idx])-1]=SV_var[1]
                            
                    else:
                        querySampleSVin_ref[sample_idx]= {
                            len(querySampleSVout[sample_idx])-1:SV_var[1]
                            }
                    resultantSV = []
                    SV_var = dict()
                    
                elif "result:" in line:
                    results = line.split("result:")[-1].strip().split(" ")
                    queryValue[sample_idx] = eval(results[0])
                    queryLabel[sample_idx] = eval(results[1])
                    queryValue_ref[sample_idx] = SV_var[0]
                    resultantSV = []
                    SV_var = dict()
                    
        if len(queryValue) == len(dataIdx)-int(len(dataIdx)/2):
            if SV_args.privacy_protection_measure != None:
                # add privacy protection
                sortResult_change = []
                for qidx in queryValue.keys():
                    print('(bef PPM) sample %s\'s SV:'%qidx, 
                          querySampleSVin[qidx],querySampleSVout[qidx],
                          queryValue[qidx])
                    
                    for sv_idx in range(len(querySampleSVout[qidx])):
                        sortResult_bef = np.argsort(
                            list(querySampleSVout_ref[qidx][sv_idx].values())).tolist()
                        sortResult_bef = [sortResult_bef.index(k) \
                                          for k in querySampleSVout_ref[qidx][sv_idx].keys()]
                        
                        SV_with_protect = privacy_protect(
                            SV_args.privacy_protection_measure, 
                            SV_args.privacy_protection_level,
                            querySampleSVout_ref[qidx][sv_idx])
                        querySampleSVout[qidx][sv_idx] = SV_with_protect[len(SV_with_protect)-1]
                    
                        sortResult_aft = np.argsort(
                            list(SV_with_protect.values())).tolist()
                        sortResult_aft = [sortResult_aft.index(k) \
                                          for k in SV_with_protect.keys()]
                        sortResult_change.append(
                            np.mean(
                                np.abs(np.array(sortResult_aft)-\
                                       np.array(sortResult_bef))/\
                                    len(SV_with_protect)
                                    )
                            #abs(sortResult_aft[len(SV_with_protect)-1]-\
                            #    sortResult_bef[len(SV_with_protect)-1])/\
                            #    len(SV_with_protect)
                                )
                            
                    for sv_idx in range(len(querySampleSVin[qidx])):
                        sortResult_bef = np.argsort(
                            list(querySampleSVin_ref[qidx][sv_idx].values())).tolist()
                        sortResult_bef = [sortResult_bef.index(k) \
                                          for k in querySampleSVin_ref[qidx][sv_idx].keys()]
                        
                        SV_with_protect = privacy_protect(
                            SV_args.privacy_protection_measure, 
                            SV_args.privacy_protection_level,
                            querySampleSVin_ref[qidx][sv_idx])
                        querySampleSVin[qidx][sv_idx] = SV_with_protect[len(SV_with_protect)-1]
                        
                        sortResult_aft = np.argsort(
                            list(SV_with_protect.values())).tolist()
                        sortResult_aft = [sortResult_aft.index(k) \
                                          for k in SV_with_protect.keys()]
                        sortResult_change.append(
                            np.mean(
                                np.abs(np.array(sortResult_aft)-\
                                       np.array(sortResult_bef))/\
                                    len(SV_with_protect)
                                    )
                            #abs(sortResult_aft[len(SV_with_protect)-1]-\
                            #    sortResult_bef[len(SV_with_protect)-1])/\
                            #    len(SV_with_protect)
                                )
                    
                    sortResult_bef = np.argsort(
                        list(queryValue_ref[qidx].values())).tolist()
                    sortResult_bef = [sortResult_bef.index(k) \
                                      for k in queryValue_ref[qidx].keys()]
                    
                    SV_with_protect = privacy_protect(
                        SV_args.privacy_protection_measure, 
                        SV_args.privacy_protection_level,
                        queryValue_ref[qidx])
                    queryValue[qidx] = SV_with_protect[len(SV_with_protect)-1]
                    
                    sortResult_aft = np.argsort(
                        list(SV_with_protect.values())).tolist()
                    sortResult_aft = [sortResult_aft.index(k) \
                                      for k in SV_with_protect.keys()]
                    sortResult_change.append(
                        np.mean(
                            np.abs(np.array(sortResult_aft)-\
                                   np.array(sortResult_bef))/\
                                len(SV_with_protect)
                                )
                        #abs(sortResult_aft[len(SV_with_protect)-1]-\
                        #    sortResult_bef[len(SV_with_protect)-1])/\
                        #    len(SV_with_protect)
                            )
                    print('(aft PPM) sample %s\'s SV:'%qidx, 
                          querySampleSVin[qidx],querySampleSVout[qidx],
                          queryValue[qidx])
                sortResult_change = np.mean(sortResult_change)
                print('sortResult_change: ',sortResult_change)
                
            queryResults = []
            queryLabels = []
            for qidx in queryValue.keys():
                in_mean = np.mean(querySampleSVin[qidx])
                out_mean = np.mean(querySampleSVout[qidx])
                in_std = np.std(querySampleSVin[qidx])
                if in_std ==0:
                    in_std = 10**(-20)
                out_std = np.std(querySampleSVout[qidx])
                if out_std ==0:
                    out_std = 10**(-20)
                value = queryValue[qidx]
                
                in_probability = norm.pdf(value, loc=in_mean, scale=in_std)
                out_probability = norm.pdf(value, loc=out_mean, scale=out_std)
                if out_probability ==0:
                    out_probability = 10**(-20)
                queryResults.append(in_probability/out_probability)
                queryLabels.append(queryLabel[qidx])
                print('Query sample %s/%s result: '%(qidx, len(queryValue)),
                      value, queryLabels[-1], queryResults[-1], 
                      in_probability, in_mean, in_std,
                      out_probability, out_mean, out_std)
                
            print(queryLabels, queryResults)
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
    SV_args.scannedIter_maxNum=50
    task = Task(SV_args) # FA task
    auxiliary_index = np.random.choice(range(len(task.Tst)),
                                       int(len(task.Tst)/2),
                                       replace = False).tolist()
    validation_index = list(set(range(len(task.Tst)))-set(auxiliary_index))
    replace_list = ['_recovery',
                    'withDP%s'%SV_args.privacy_protection_level,
                    'withQT%s'%SV_args.privacy_protection_level,
                    'withDR%s'%SV_args.privacy_protection_level,
                    '_seed10','_seed1']
    log_file = SV_args.log_file
    for replace_str in replace_list:
        log_file = log_file.replace(replace_str,"")
    if not os.path.exists(log_file) or log_file == SV_args.log_file:
        log_file = log_file.replace('FIA',"FA")
        
    testSampleFeatureSV = dict()
    SV_var = dict()
    Current_SV = []
    if os.path.exists(log_file):
        with open(log_file, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if 'Current SV' in line:
                    Current_SV.append(eval(line.split('Current SV:')[-1]))
                if 'SV of test sample' in line:
                    sample_idx=int(line.split("/")[0].split(" ")[-1])
                    testSampleFeatureSV[sample_idx] = eval("{"+line.split("{")[-1])
                    SV_var[sample_idx] = dict([
                        (feature_idx, \
                         np.var([sv[feature_idx] for sv in Current_SV])) \
                        for feature_idx in testSampleFeatureSV[sample_idx].keys()
                        ])
                    Current_SV=[]
    if len(testSampleFeatureSV)==len(task.Tst):
        print('read existing testSampleFeatureSV...')
        task.trainModel()
        if SV_args.privacy_protection_measure != None:
            sortResult_change = []
            for test_idx in testSampleFeatureSV.keys():
                print('(bef PPM) sample %s\'s SV:'%test_idx, 
                      testSampleFeatureSV[test_idx])
                sortResult_bef = np.argsort(
                    list(testSampleFeatureSV[test_idx].values())).tolist()
                sortResult_bef = [sortResult_bef.index(k) \
                                  for k in testSampleFeatureSV[test_idx].keys()]
                
                testSampleFeatureSV[test_idx] = privacy_protect(
                    SV_args.privacy_protection_measure, 
                    SV_args.privacy_protection_level,
                    testSampleFeatureSV[test_idx], 
                    SV_var[test_idx])
                print('(aft PPM) sample %s\'s SV:'%test_idx, 
                      testSampleFeatureSV[test_idx])
                sortResult_aft = np.argsort(
                    list(testSampleFeatureSV[test_idx].values())).tolist()
                sortResult_aft = [sortResult_aft.index(k) \
                                  for k in testSampleFeatureSV[test_idx].keys()]
                
                sortResult_change.append(
                    np.mean(
                        np.abs(np.array(sortResult_aft)-\
                               np.array(sortResult_bef))/\
                            len(testSampleFeatureSV[test_idx])
                            ))
                print('(aft PPM) sample %s\'s SV sortition:'%test_idx, 
                      sortResult_change[-1],
                      sortResult_bef, sortResult_aft)
                
            sortResult_change = np.mean(sortResult_change)
            print('sortResult_change: ',sortResult_change)
        task.testSampleFeatureSV = testSampleFeatureSV
    else:
        task.run()
    
    auxiliary_SV = [list(task.testSampleFeatureSV[test_idx].values()) for test_idx in auxiliary_index]
    validation_SV = [list(task.testSampleFeatureSV[test_idx].values()) for test_idx in validation_index]
    if task.model_name in ['KNN', 'Tree']:
        auxiliary_data = ImageDataset(
            torch.FloatTensor(auxiliary_SV), 
            torch.FloatTensor(task.X_test[auxiliary_index]), 
            len(auxiliary_index), range(len(auxiliary_index)))
        validation_data = ImageDataset(
            torch.FloatTensor(validation_SV), 
            torch.FloatTensor(task.X_test[validation_index]), 
            len(validation_index), range(len(validation_index)))
    else:
        pass
        #auxiliary_data = copy.deepcopy(task.Tst)
        #auxiliary_data.idxs = [task.Tst.idxs[idx] for idx in auxiliary_index]
        #validation_data = copy.deepcopy(task.Tst)
        #validation_data.idxs = [task.Tst.idxs[idx] for idx in validation_index]
        
    attackModel = LinearAttackModel(len(task.players))
    print('attackModel: ', attackModel)
    attackModel = DNNTrain(attackModel, auxiliary_data, 
                           epoch=500, batch_size=50, lr=0.01, 
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
    SV_args.scannedIter_maxNum=50
    task = Task(SV_args) # FA task
    auxiliary_index = np.random.choice(range(len(task.Tst)),
                                       int(len(task.Tst)/2),
                                       replace = False).tolist()
    if random_mode == 'uniform':
        if task.model_name in ['KNN', 'Tree']:
            print('replace %s with random samples (uniform)...'%auxiliary_index)
            task.X_test[auxiliary_index] = np.random.rand(
                *task.X_test[auxiliary_index].shape)
        else:
            pass
    elif random_mode == 'normal':
        if task.model_name in ['KNN', 'Tree']:
            print('replace %s with random samples (normal)...'%auxiliary_index)
            task.X_test[auxiliary_index] = np.random.normal(
                0.5,0.25,task.X_test[auxiliary_index].shape)
        else:
            pass
    else:
        pass
    validation_index = list(set(range(len(task.Tst)))-set(auxiliary_index))
    replace_list = ['_noAMTwithAUX',
                    'withDP%s'%SV_args.privacy_protection_level,
                    'withQT%s'%SV_args.privacy_protection_level,
                    'withDR%s'%SV_args.privacy_protection_level,
                    '_seed10','_seed1']
    log_file = SV_args.log_file
    for replace_str in replace_list:
        log_file = log_file.replace(replace_str,"")
    if not os.path.exists(log_file) or log_file == SV_args.log_file:
        log_file = log_file.replace('FIA',"FA")
        
    testSampleFeatureSV = dict()
    SV_var = dict()
    Current_SV = []
    if os.path.exists(log_file):
        with open(log_file, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if 'Current SV' in line:
                    Current_SV.append(eval(line.split('Current SV:')[-1]))
                if 'SV of test sample' in line:
                    sample_idx=int(line.split("/")[0].split(" ")[-1])
                    testSampleFeatureSV[sample_idx] = eval("{"+line.split("{")[-1])
                    SV_var[sample_idx] = dict([
                        (feature_idx, \
                         np.var([sv[feature_idx] for sv in Current_SV])) \
                        for feature_idx in testSampleFeatureSV[sample_idx].keys()
                        ])
                    Current_SV=[]
                
    if len(testSampleFeatureSV)==len(task.Tst):
        print('read existing testSampleFeatureSV...')
        task.trainModel()
        if SV_args.privacy_protection_measure != None:
            sortResult_change = []
            for test_idx in testSampleFeatureSV.keys():
                print('(bef PPM) sample %s\'s SV:'%test_idx, 
                      testSampleFeatureSV[test_idx])
                sortResult_bef = np.argsort(
                    list(testSampleFeatureSV[test_idx].values())).tolist()
                sortResult_bef = [sortResult_bef.index(k) \
                                  for k in testSampleFeatureSV[test_idx].keys()]
                
                testSampleFeatureSV[test_idx] = privacy_protect(
                    SV_args.privacy_protection_measure, 
                    SV_args.privacy_protection_level,
                    testSampleFeatureSV[test_idx], 
                    SV_var[test_idx])
                print('(aft PPM) sample %s\'s SV:'%test_idx, 
                      testSampleFeatureSV[test_idx])
                
                sortResult_aft = np.argsort(
                    list(testSampleFeatureSV[test_idx].values())).tolist()
                sortResult_aft = [sortResult_aft.index(k) \
                                  for k in testSampleFeatureSV[test_idx].keys()]
                
                sortResult_change.append(
                    np.mean(
                        np.abs(np.array(sortResult_aft)-\
                               np.array(sortResult_bef))/\
                            len(testSampleFeatureSV[test_idx])
                            ))
                print('(aft PPM) sample %s\'s SV sortition:'%test_idx, 
                      sortResult_change[-1],
                      sortResult_bef, sortResult_aft)
                
            sortResult_change = np.mean(sortResult_change)
            print('sortResult_change: ',sortResult_change)
        task.testSampleFeatureSV = testSampleFeatureSV
    else:
        task.run()
    
    auxiliary_SV = [list(task.testSampleFeatureSV[test_idx].values()) \
                    for test_idx in auxiliary_index]
    validation_SV = [list(task.testSampleFeatureSV[test_idx].values()) \
                     for test_idx in validation_index]
    
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