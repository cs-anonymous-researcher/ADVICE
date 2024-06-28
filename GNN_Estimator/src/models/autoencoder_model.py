#!/usr/bin/env python
# -*- coding: utf-8 -*-


# %%
import torch
from torch import nn
import torch_geometric
from torch_geometric.nn import SAGEConv
# %%

class NodeEncoder(torch.nn.Module):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, hidden_channels, out_channels):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

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
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


# %%

class EdgeDecoder(torch.nn.Module):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, hidden_channels):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        super().__init__()
        self.lin1 = torch.nn.Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, 1)

    def forward(self, z, edge_index):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        z_src, z_dst = z[edge_index[0]], z[edge_index[1]]
        z = torch.cat([z_src, z_dst], dim = -1)
        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)



# %%

class GNNAutoEncoder(nn.Module):
    """
    基于图神经网络自编码器

    Members:
        field1:
        field2:
    """

    def __init__(self, encode_hidden, encode_out):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        super().__init__()
        self.encoder = NodeEncoder(encode_hidden, encode_out)
        self.decoder = EdgeDecoder(encode_out)

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
        z = self.encoder(x, edge_index)
        out = self.decoder(z, edge_index) 
        return out


# %%
