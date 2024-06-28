#!/usr/bin/env python
# -*- coding: utf-8 -*-


# %%
from utilities.global_config import *
from training.optimizer import get_optimizer, get_lr_scheduler

# %%
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, GIN, SAGEConv
import torch.nn.functional as F
import time
from utilities import utils
# %%


class GNNModel(nn.Module):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, workload, hidden_size, embedding_dims, layer_type, dropout_rate, decay):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        super().__init__()

        # self.workload = workload
        input_dims = utils.get_feature_number(workload)
        self.dropout_rate = dropout_rate
        self.decay = decay
        self.training_num = None

        if layer_type == "gcn":
            self.conv1 = GCNConv(input_dims, hidden_size)
            self.conv2 = GCNConv(hidden_size, embedding_dims)
        elif layer_type == "gat":
            # self.conv1 = GATConv((-1, -1), hidden_size)
            self.conv1 = GATConv(input_dims, hidden_size)
            self.conv2 = GATConv(hidden_size, embedding_dims)
        elif layer_type == "gin":
            self.conv1 = GIN(input_dims, hidden_size)
            self.conv2 = GIN(hidden_size, embedding_dims)
        elif layer_type == "sage":
            self.conv1 = SAGEConv((-1, -1), hidden_size)
            self.conv2 = SAGEConv(hidden_size, embedding_dims)
        else:
            raise ValueError(f"GNNModel.__init__: unsupported layer_type({layer_type})")

        self.linear = nn.Linear(embedding_dims, 1)

    def set_training_num(self, in_num):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        self.training_num = in_num

    def forward(self, x, edge_index):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        # 加入dropout
        x = F.dropout(x, p=self.dropout_rate)

        # 考虑加入Pooling层
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=self.dropout_rate)
        x = self.linear(x)
        return x.view(-1)


    def load_params(self, in_path):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        self.dropout_rate, self.decay, self.training_num = utils.load_pickle(in_path)

    def dump_params(self, out_path):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        utils.dump_pickle((self.dropout_rate, \
            self.decay, self.training_num), out_path)
        
# %%
