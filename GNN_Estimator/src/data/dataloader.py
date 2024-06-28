#!/usr/bin/env python
# -*- coding: utf-8 -*-


# %%
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split, Subset
import torch

# %%

def get_train_test_loader(dataset: InMemoryDataset, config):
    """
    {Description}

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    train_size = int(len(dataset) * config['train_ratio'])
    test_size = len(dataset) - train_size
    print(f"get_train_test_loader: train_size = {train_size}. test_size = {test_size}.")
    torch.manual_seed(42)

    # train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_dataset = Subset(dataset, range(train_size))
    test_dataset = Subset(dataset, range(train_size, train_size + test_size))

    print(f"get_train_test_loader: len(train_dataset) = {len(train_dataset)}. len(test_dataset) = {len(test_dataset)}.")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=config['shuffle'],
        num_workers=config['num_workers']
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=config['shuffle'],
        num_workers=config['num_workers']
    )

    return train_loader, test_loader

