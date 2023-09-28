import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

def get_train_val_data_loaders(batch_size, valid_size, transforms, num_workers):
    """
    Returns the training and validation data loaders.
    """
    # Download the training and validation datasets
    trainval_data = datasets.CIFAR10('data', train=True, download=True, transform=transforms)
    
    # Compute how many items we will reserve for the validation set
    n_tot = len(trainval_data)
    split = int(np.floor(valid_size * n_tot))
    
    # Compute the indices for the training set and for the validation set
    torch.manual_seed(42)
    shuffled_indices = torch.randperm(n_tot)
    train_idx, valid_idx = shuffled_indices[split:], shuffled_indices[:split]
    
    # Define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(indices=train_idx)
    valid_sampler = SubsetRandomSampler(indices=valid_idx)
    
    # Prepare dataloaders
    train_loader = DataLoader(trainval_data, batch_size, shuffle=True, sampler=train_sampler, num_workers=num_workers)
    valid_loader = DataLoader(trainval_data, batch_size, shuffle=False, sampler=valid_sampler, num_workers=num_workers)
    
    return train_loader, valid_loader


def get_test_data_loader(batch_size, transforms, num_workers):
    """
    Returns the test data loader. 
    """
    test_data = datasets.CIFAR10("data", train=False, transform=transforms, download=True)
    test_loader = DataLoader(test_data, batch_size, shuffle=False, num_workers=num_workers)
    
    return test_loader
