#!/usr/bin/env python
# -*- coding: utf-8 -*-

# %%

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from utilities import utils
# %%

class GCNModel(nn.Module):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, workload, hidden_channels_1, embedding_dims):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        # feature_num_path = "/home/jinly/GNN_Estimator/data/feature_number.json"
        # feature_num_dict = utils.load_json(feature_num_path)
        # input_dims = feature_num_dict[workload]
        input_dims = utils.get_feature_number(workload)

        super().__init__()
        torch.manual_seed(1234567)

        self.conv1 = GCNConv(input_dims, hidden_channels_1)
        self.conv2 = GCNConv(hidden_channels_1, embedding_dims)
        self.linear = nn.Linear(embedding_dims, 1)


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
        x = x.relu()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        # x = x.relu()
        # x = F.dropout(x, p=0.2, training=self.training)
        # x = self.conv3(x, edge_index)

        # return x
        out = self.linear(x)
        return out.view(-1)

# %%
