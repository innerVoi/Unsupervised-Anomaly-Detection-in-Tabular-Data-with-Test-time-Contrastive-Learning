import os
import numpy as np
from scipy import io
import torch
from torch.utils.data import Dataset


class MatDataset(Dataset):
    def __init__(self, dataset_name: str, data_dim: int, data_dir: str, mode: str = 'train'):
        super(MatDataset, self).__init__()
        path = os.path.join(data_dir, dataset_name + '.mat')
        data = io.loadmat(path)
        samples = data['X']
        labels = ((data['y']).astype(int)).reshape(-1)

        inliers = samples[labels == 0]
        outliers = samples[labels == 1]
        train_data, train_label, test_data, test_label = train_test_split(inliers, outliers)

        if mode == 'train':
            self.data = torch.Tensor(train_data)
            self.targets =torch.Tensor(train_label)
        else:
            self.data = torch.Tensor(test_data)
            self.targets = torch.Tensor(test_label)

    def __getitem__(self, item):
        return self.data[item], self.targets[item]

    def __len__(self):
        return len(self.data)
    

class NpzDataset(Dataset):
    def __init__(self, dataset_name: str, data_dim: int, data_dir: str, mode: str = 'train'):
        super(NpzDataset, self).__init__()
        path = os.path.join(data_dir, dataset_name+'.npz')
        data=np.load(path)  
        samples = data['X']
        labels = ((data['y']).astype(int)).reshape(-1)

        inliers = samples[labels == 0]
        outliers = samples[labels == 1]
        train_data, train_label, test_data, test_label = train_test_split(inliers, outliers)
        if mode == 'train':
            self.data = torch.Tensor(train_data)
            self.targets =torch.Tensor(train_label)
        else:
            self.data = torch.Tensor(test_data)
            self.targets = torch.Tensor(test_label)

    def __getitem__(self, item):
        return self.data[item], self.targets[item]

    def __len__(self):
        return len(self.data)

    
def train_test_split(inliers, outliers):
    num_split = len(inliers) // 2
    train_data = inliers[:num_split]
    train_label = np.zeros(num_split)
    test_data = np.concatenate([inliers[num_split:], outliers], 0)

    test_label = np.zeros(test_data.shape[0])
    test_label[num_split:] = 1
    return train_data, train_label, test_data, test_label
