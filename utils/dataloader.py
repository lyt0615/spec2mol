import numpy as np
import torch
import pandas as pd

from utils.transform import ToFloatTensor, bacteria_train_transform, bacteria_valid_transform
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split as tts
from torchvision import transforms
import torch.nn.functional as F

train_transform = transforms.Compose([ToFloatTensor()])
valid_transform = transforms.Compose([ToFloatTensor()])


class MyDataset(Dataset):
    """create dataset"""

    def __init__(self, X, y, transform=None, pool_dim=None):
        X, y = np.vstack(X), np.vstack(y)
        self.data = X
        self.transform = transform
        self.label = torch.FloatTensor(y) if y.ndim == 2 else torch.LongTensor(y)
        self.pool_dim = pool_dim

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        X = self.data[item, :]
        X = self.transform(X)
        y = self.label[item]
        if self.pool_dim:
            X = F.adaptive_avg_pool1d(X, self.pool_dim)
        return X, y


def make_trainloader(ds, batch_size=16, num_workers=0, train_size=0.8, seed=42, tune=False, pool_dim=None):

    if tune:
        data = pd.read_pickle(f'datasets/{ds}/tune.pkl')
        data_x = data['spectrum'].values
        data_y = data['smiles'].values

    else:
        data = pd.read_pickle(f'datasets/{ds}/train.pkl')
        data_x = data['spectrum'].values
        data_y = data['smiles'].values

    ids = np.arange(len(data_y))

    transform_train = bacteria_train_transform if ds == 'Bacteria' else train_transform
    transform_valid = bacteria_valid_transform if ds == 'Bacteria' else valid_transform

    stratify = None if ds == 'FunctionalGroups' else data_y

    if train_size is None:
        data_train = pd.read_pickle(f'datasets/{ds}/train.pkl')
        data_val = pd.read_pickle(f'datasets/{ds}/eval.pkl')
        train_x = data_train['spectrum'].values
        train_y = data_train['smiles'].values
        val_x = data_val['spectrum'].values
        val_y = data_val['smiles'].values
        trainset = MyDataset(train_x, train_y, transform=transform_train, )
        valset = MyDataset(val_x, val_y, transform=transform_valid, )

    elif train_size:
        datat = pd.read_pickle(f'datasets/{ds}/train.pkl')
        data_x = datat['spectrum'].values
        data_y = datat['smiles'].values
        train_id, val_id = tts(ids, shuffle=False, train_size=train_size, random_state=seed, stratify=stratify)
        trainset = MyDataset(data_x[train_id], data_y[train_id], transform=transform_train, )
        valset = MyDataset(data_x[val_id], data_y[val_id], transform=transform_valid, )

    else:
        test_data = pd.read_pickle(f'datasets/{ds}/test.pkl')
        test_x = test_data['spectrum'].values
        test_y = test_data['smiles'].values

        trainset = MyDataset(data_x, data_y, transform=transform_train, )
        valset = MyDataset(test_x, test_y, transform=transform_valid, )

    trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)
    valloader = DataLoader(valset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)

    return trainloader, valloader


def make_testloader(ds, batch_size=128, num_workers=0, pool_dim=256):
    data = pd.read_pickle(f'datasets/{ds}/test.pkl')
    data_x = data['spectrum'].values
    data_y = data['smiles'].values

    transform_valid = bacteria_valid_transform if ds == 'Bacteria' else valid_transform

    testset = MyDataset(data_x, data_y, transform=transform_valid, )
    return DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

