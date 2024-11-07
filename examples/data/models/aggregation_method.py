#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import numpy as np
import torch
import sys
sys.path.append("..")
from functools import reduce

def model_dist_norm(model, target_params):
    squared_sum = 0
    param_dict = model#dict(model.state_dict())
    for name, layer in param_dict.items():
        if 'running_' in name or '_tracked' in name:
            continue
        squared_sum += torch.sum(
                torch.pow(layer.data - \
                          target_params[name].clone().detach().requires_grad_(False).data, 2))
    return torch.sqrt(squared_sum).item()

def Mutual_Divergence(w_glob, w, device):
    #print('\n calculating mutual weight_diverge...')
    tmp = {}
    idx_keys = list(w.keys())
    param_keys = list(w[idx_keys[0]].keys())
    for i in idx_keys:
        client_vector = torch.Tensor([]).to(device)
        for key in param_keys:
            client_vector = torch.cat(
                (client_vector, 
                 (w_glob[key].detach().reshape(-1) - \
                  w[i][key].detach().reshape(-1)).to(device))
                    )
        tmp[i] = client_vector
    
    mutual_div = {}
    for idxi in idx_keys:
        sum_ = torch.zeros(len(w))
        for j, idxj in enumerate(idx_keys):
            sum_[j] = torch.norm(tmp[idxi] - tmp[idxj])**2#/np.sqrt(len(tmp[i]))
        mutual_div[idxi] = sum_
    #weight = torch.softmax(torch.tensor(mutual_div).to(device),-1)
    #print('mutual_div: ', mutual_div)#
    return mutual_div#weight

def SimpleAvg(w_locals):
    #net_glob = copy.deepcopy(list(w_locals.values())[0])
    weight = 1/len(w_locals)
    parameter_keys = list(w_locals.values())[0].keys()
    idx_keys = list(w_locals.keys())
    net_glob = dict([(k,None) for k in parameter_keys])
    
    for k in parameter_keys:
        #flag = False
        for idx in idx_keys:
            if type(net_glob[k]) == type(None):
                net_glob[k] = weight * w_locals[idx][k].clone()
                #flag = True
            else:
                net_glob[k] += weight * w_locals[idx][k].clone()
         
    return net_glob

def WeightedAvg(w_locals, p_k):
    parameter_keys = list(w_locals.values())[0].keys()
    idx_keys = list(w_locals.keys())
    net_glob = dict([(k,None) for k in parameter_keys])
    
    sum_pk = sum([p_k[idx] for idx in idx_keys])
    
    for k in parameter_keys:
        #flag = False
        for idx in idx_keys:
            if type(net_glob[k]) == type(None):
                net_glob[k] = p_k[idx]/sum_pk * w_locals[idx][k].clone()
                #net_glob[k] = copy.deepcopy(p_k[idx]/sum_pk * w_locals[idx][k])
                #flag = True
            else:
                net_glob[k] += p_k[idx]/sum_pk * w_locals[idx][k].clone()
                
    return net_glob

def Krum(w_glob, w, device, k,m):
    mutual_div = Mutual_Divergence(w_glob, w, device)
    #print(mutual_div)
    score = dict()
    for i in w.keys():
        score[i] = torch.sum(torch.sort(mutual_div[i])[0][:k + 1]) # +1 exclude itself 
    #print(score)
    
    keys = list(score.keys())
    selected_idx = np.argsort(list(score.values()))[:m]
    #print('selected_idx: ', selected_idx)
    selected_parameters = {}
    for i in selected_idx:
        selected_parameters[keys[i]] = w[keys[i]]
        
    return selected_parameters

def K_norm(w_glob, w, device, m=1):
    model_norms  = dict()
    idx_keys = list(w.keys())
    for i in idx_keys:
        model_norms[i] = model_dist_norm(w[i], w_glob)
    
    #print(model_norms)
    selected_idx = np.argsort(list(model_norms.values()))[:m]
    #print('selected_idx: ', selected_idx)
    selected_parameters = {}
    keys = list(model_norms.keys())
    for i in selected_idx:
        selected_parameters[keys[i]] = w[keys[i]]
        
    return selected_parameters


def FedMedian(w_locals, device):
    parameter_keys = list(w_locals.values())[0].keys()
    idx_keys = list(w_locals.keys())
    net_glob = dict([(k,None) for k in parameter_keys])
    
    for k in parameter_keys:
        temp = torch.zeros([len(idx_keys)] + \
                           list(w_locals[idx_keys[0]][k].shape)).to(device)
        for no,i in enumerate(idx_keys):
            temp[no]= w_locals[i][k]
        net_glob[k] = temp.median(dim=0)[0].clone() 
    return net_glob

def FedTrimmedMean(w_locals, device, trim_num = 1):
    #print('trim_num: ',trim_num)
    parameter_keys = list(w_locals.values())[0].keys()
    idx_keys = list(w_locals.keys())
    net_glob = dict([(k,None) for k in parameter_keys])
    
    for k in parameter_keys:
        shape = w_locals[idx_keys[0]][k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(idx_keys), total_num).to(device)
        for no,i in enumerate(idx_keys):
            y_list[no] = torch.reshape(w_locals[i][k], (-1,))
        y = torch.t(y_list)
        y_sorted = y.sort()[0]
        if trim_num > 0:
            result = y_sorted[:, trim_num:-trim_num]
        else:
            result = y_sorted
        result = result.mean(dim=-1)
        assert total_num == len(result)
        net_glob[k] = torch.reshape(result, shape).clone()
    return net_glob

