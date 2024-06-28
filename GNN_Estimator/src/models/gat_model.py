#!/usr/bin/env python
# -*- coding: utf-8 -*-


# %%

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GATv2Conv
import torch.nn.functional as F

from utilities import utils
# %%

class GATModel(nn.Module):
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
        super().__init__()
        torch.manual_seed(1234567)

        input_dims = utils.get_feature_number(workload)

        # self.conv1 = GATConv((-1, -1), hidden_channels_1)
        # self.conv2 = GATConv((-1, -1), embedding_dims)
        # self.conv1 = GATConv(input_dims, hidden_channels_1)
        # self.conv2 = GATConv(hidden_channels_1, embedding_dims)
        self.conv1 = GATv2Conv(input_dims, hidden_channels_1)
        self.conv2 = GATv2Conv(hidden_channels_1, embedding_dims)

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
        # x = x.relu()      
        x = F.leaky_relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        # x = x.relu()
        # x = F.dropout(x, p=0.2, training=self.training)
        # x = self.conv3(x, edge_index)
        out = self.linear(x)
        return out.view(-1)



