#!/usr/bin/env python
# -*- coding: utf-8 -*-


# %%

import torch
from torch.nn import Module
from torch_geometric.loader import DataLoader
import torch.nn as nn

# %%

def launch_evaluate_process(model: Module, test_loader: DataLoader):
    """
    {Description}
    
    Args:
        model:
        test_loader:
    Returns:
        res1:
        res2:
    """

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device=device)
    criterion = nn.MSELoss()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            
            loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)

            _, pred = torch.max(output, 1)
            correct += torch.sum(pred == target).item()
            total += target.size(0)

    avg_loss = test_loss / total
    accuracy = correct / total * 100
    
    model.train()  # 将模型设置回训练模式

    return avg_loss, accuracy
# %%
