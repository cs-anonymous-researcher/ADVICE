#!/usr/bin/env python
# -*- coding: utf-8 -*-


# %%
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
import torch.nn.functional as F
import numpy as np

class GaussianGNN(nn.Module):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, hidden_channels, embedding_dims):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        super().__init__()
        torch.manual_seed(1234567)

        self.conv1 = GATConv((-1, -1), hidden_channels)
        self.conv2 = GATConv(hidden_channels, embedding_dims)
        self.linear = nn.Linear(embedding_dims, 2)

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
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        out = self.linear(x)

        mean, variance = torch.split(out, 1, dim=1)
        variance = F.softplus(variance) + 1e-6
        return mean, variance


# # %%

# class GaussianGraphModel(pl.LightningModule):
#     """
#     {Description}

#     Members:
#         field1:
#         field2:
#     """

#     def __init__(self, hidden_channels, embedding_dims):
#         """
#         {Description}

#         Args:
#             arg1:
#             arg2:
#         """
#         super().__init__()
#         self.model = GaussianGNN(hidden_channels, embedding_dims)

#     def forward(self, x, edge_index):
#         """
#         {Description}

#         Args:
#             arg1:
#             arg2:
#         Returns:
#             return1:
#             return2:
#         """
#         return self.model(x, edge_index)


#     def training_step(self, batch, batch_idx):
#         """
#         {Description}

#         Args:
#             arg1:
#             arg2:
#         Returns:
#             return1:
#             return2:
#         """
#         model = self.model
#         data = batch
#         data.x.requires_grad = True
#         mean, var = self.model(data.x, data.edge_index)
#         loss_for_adv = F.gaussian_nll_loss(mean, data.y, var)
#         grad = torch.autograd.grad(loss_for_adv, data.x, retain_graph=False)[0]

#         x_new = data.x.detach()
#         y_new = data.y.detach()

#         mean_new = mean.detach()
#         var_new = var.detach()
#         loss_for_adv.detach_()
#         perturbed_data = fgsm_attack(x_new, 0.01 * data_range, grad) # 8 is range of data
#         mean_adv, var_adv = self.model(perturbed_data)
#         loss = F.gaussian_nll_loss(mean_new, y_new, var_new) + \
#             F.gaussian_nll_loss(mean_adv, y_new, var_adv)
#         self.log('train_loss', loss)
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
#         return loss
    

#     def configure_optimizers(self,):
#         """
#         {Description}
    
#         Args:
#             arg1:
#             arg2:
#         Returns:
#             return1:
#             return2:
#         """
#         optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
#         return optimizer
    

# %%

def create_model_list(hidden_channels, embedding_dims, model_num):
    """
    创建模型列表
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    return [GaussianGNN(hidden_channels, embedding_dims) for _ in range(model_num)]

# %%
