import torch
from torch.utils.data import Dataset
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, dataset, labels, original_len, idxs, attacker= False, 
                 poison_labels=[], after_poison_labels=[]):
        self.dataset = dataset
        self.labels = labels
        self.idxs = list(idxs)
        self.attacker = attacker
        self.poison_labels = poison_labels
        self.after_poison_labels = after_poison_labels
        self.original_len = original_len
        
    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        
        if type(self.labels) == type(None):
            data, true_label = self.dataset[self.idxs[item]%self.original_len]
        else:
            data = self.dataset[self.idxs[item]%self.original_len]
            true_label = self.labels[self.idxs[item]%self.original_len]
        
        if self.idxs[item] > self.original_len-1:
            #print('add noise...')
            data += torch.normal(0,0.05, data.shape)
            
        if self.attacker and int(true_label) in self.poison_labels:
            label = torch.tensor(
                    self.after_poison_labels[
                            self.poison_labels.index(int(true_label))
                            ]
                    )
            
        else:
            label = true_label
        return data, label, true_label

class TextDataset(Dataset):
    def __init__(self, dataset, labels, original_len, idxs, attacker= False, 
                 poison_sentences=[], after_poison_sentences=[]):
        #self.dataset = dataset
        #self.labels = labels
        #print('len_: ',len(idxs))
        self.dataset = dataset
        self.labels = labels
        self.original_len = original_len
        self.idxs = list(idxs)
        self.attacker = attacker
        self.poison_sentences = poison_sentences
        self.after_poison_sentences = after_poison_sentences
        
    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        #tmp = self.idxs[item]%self.original_len
        data = self.dataset[self.idxs[item]%self.original_len]
        true_label = self.labels[self.idxs[item]%self.original_len]
        
        if self.idxs[item] > self.original_len-1:
            #print('add noise...')
            if len(data.shape)<=1:
                data[np.random.choice(range(80), 1, replace=True)] = 1
            else:
                data[:,np.random.choice(range(80), 1, replace=True)] = 1
            
            
        if self.attacker and int(true_label) in self.poison_sentences:
            label = torch.tensor(
                    self.after_poison_sentences[
                            self.poison_sentences.index(int(true_label))
                            ]
                    )
        else:
            label = true_label
        
        return data, label, true_label
  