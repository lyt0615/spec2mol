import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from sklearn.model_selection import train_test_split as tts
# from torchvision import transforms
import torch.nn.functional as F
from functools import lru_cache
from transformers import AutoTokenizer

# train_transform = transforms.Compose([torch.nn.Identity()])#ToFloatTensor()
# eval_transform = transforms.Compose([torch.nn.Identity()])#ToFloatTensor()

class MyDataset(Dataset):
    """create dataset"""

    def __init__(self, X, y, transform=None, pool_dim=None, cache_size=10, to_smarts=False):
        X = self.stack(X)
        y = self.stack(y)
        self.data = X
        self.label = [{'smiles':i,
                      'label': smiels_to_subs_smarts(i) if to_smarts else i} for i in y] #self.to_label(y)
        self.transform = transform
        self.pool_dim = pool_dim
        self.cache_size = cache_size
        self.loadbuffer = lru_cache(maxsize=self.cache_size)(self.__getitem__)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        X = self.data[item]
        y = self.label[item]
        if self.pool_dim:
            X = F.adaptive_avg_pool1d(X, self.pool_dim)
        return X, y
    
    def __getstate__(self):
        state = self.__dict__.copy()
        state['loadbuffer'] = self.__getitem__
        return state
    
    def __setstate__(self, state):
        state['loadbuffer'] = lru_cache(maxsize=100)(state['loadbuffer'])
        self.__dict__.update(state)
    
    def stack(self, input):
        try:
            return torch.FloatTensor(np.vstack(input))
        except:
            return input
        
    def to_label(self, input):
        if not type(input) == list:
            return torch.FloatTensor(input) if input.ndim == 2 else torch.LongTensor(input)
        else:
            return input


def collate_fn(batch):
    tokenizer = AutoTokenizer.from_pretrained("models/moltokenizer")
    x = [item[0] for item in batch]
    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=1).unsqueeze(1)
    smiles = [item[1]['smiles'] for item in batch]
    y = [item[1]['label'] for item in batch]
    y = tokenizer(y, padding=True, truncation=True, return_tensors="pt")
    return x, {'smiles': smiles, 'label': y}
    

def smiels_to_subs_smarts(data_y):
    from rdkit import Chem
    from .subs_smarts import subs_smarts
    pattern = [Chem.MolFromSmarts(s) for s in subs_smarts]
    # print(data_y)
    # for i in range(len(data_y)):
    smarts_seq = ''
    mol = Chem.MolFromSmiles(data_y)
    for j in range(len(pattern)):
        if mol.HasSubstructMatch(pattern[j]):
            smarts_seq += subs_smarts[j]
            if j != len(pattern)-1: smarts_seq += '<d>'  ## add delimiter
        # data_y = np.array(smarts_seq)
    return smarts_seq

        
def make_trainloader(ds, batch_size=16, num_workers=0, train_size=0.8, seed=42, mode='train', to_smarts=False):
    
    data = pd.read_pickle(f'datasets/{ds}/{mode}.pkl')
    data_x = data['spectrum']
    if not 'pretrain' in mode:
        data_y = data['smiles'].values
    else:
        pass
    # else: raise ValueError("Argument 'STRATEGY['mode'] should be chhosen among ""train, pretrain, test and tune"".")
    ids = np.arange(len(data_x))
    # transform_train = bacteria_train_transform if ds == 'Bacteria' else train_transform
    # transform_eval = bacteria_eval_transform if ds == 'Bacteria' else eval_transform
    stratify = None if 'pretrain' in mode else data_y
    assert 'train' in mode or mode == 'test'
    if 'train' in mode:
        if train_size is None:
            data_train = pd.read_pickle(f'datasets/{ds}/train.pkl')
            data_val = pd.read_pickle(f'datasets/{ds}/eval.pkl')
            train_x = data_train['spectrum'].values
            train_y = data_train['smiles'].values
            val_x = data_val['spectrum'].values
            val_y = data_val['smiles'].values
            trainset = MyDataset(train_x, train_y, to_smarts=to_smarts)
            valset = MyDataset(val_x, val_y, to_smarts=to_smarts)
        else:
            data = pd.read_pickle(f'datasets/{ds}/{mode}.pkl')
            data_x = data['spectrum'].values
            train_id, val_id = tts(ids, shuffle=False, train_size=train_size, random_state=seed, stratify=stratify)
            if mode == 'train':
                data_y = data['smiles'].values
                trainset = MyDataset(data_x[train_id], data_y[train_id], to_smarts=to_smarts)
                valset = MyDataset(data_x[val_id], data_y[val_id], to_smarts=to_smarts)
            else: 
                data_x = torch.FloatTensor(np.vstack(data_x)).unsqueeze(1)
                trainset = TensorDataset(data_x[train_id])
                valset = TensorDataset(data_x[val_id])
    if mode == 'test':
        test_data = pd.read_pickle(f'datasets/{ds}/test.pkl')
        test_x = test_data['spectrum'].values
        test_y = test_data['smiles'].values
        trainset = MyDataset(data_x, data_y, to_smarts=to_smarts)
        valset = MyDataset(test_x, test_y, to_smarts=to_smarts)
    collate_func = collate_fn if mode == 'train' else None
    trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, 
                             shuffle=False, pin_memory=True, sampler=DistributedSampler(trainset), collate_fn=collate_func)
    valloader = DataLoader(valset, batch_size=batch_size, num_workers=num_workers, 
                           shuffle=False, pin_memory=True, sampler=DistributedSampler(valset), collate_fn=collate_func)
    return trainloader, valloader


def make_testloader(ds, batch_size=128, num_workers=0, pool_dim=256, collate=True, num=None, to_smarts=False):
    data = pd.read_pickle(f'datasets/{ds}/test.pkl')
    data_x = data['spectrum'].values[:num]
    data_y = data['smiles'].values[:num]
    # transform_eval = bacteria_eval_transform if ds == 'Bacteria' else eval_transform
    collate_func = collate_fn if collate else None
    testset = MyDataset(data_x, data_y, to_smarts=to_smarts)
    return DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_func)

