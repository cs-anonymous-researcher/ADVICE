#!/usr/bin/env python
# -*- coding: utf-8 -*-

# %%

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
import time
import numpy as np
from sklearn.metrics import r2_score


from data import query_dataset
from models import gcn_model, gat_model
from data.dataloader import get_train_test_loader
from training.optimizer import get_optimizer, get_lr_scheduler

from utilities.global_config import *
    
# %%
def launch_train_process():
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        model: 
        train_loader: 
        test_loader:
    """
    dataset = query_dataset.QueryGraphDataset()
    train_loader, test_loader = get_train_test_loader(\
        dataset=dataset, config=data_split_config)

    num_features = dataset.get_node_feature_number()
    model = gcn_model.GCNModel(num_features=num_features, \
        hidden_channels_1=200, hidden_channels_2=100, embedding_dims=50)
    
    optimizer = get_optimizer(model, optimizer_config)
    scheduler = get_lr_scheduler(optimizer, scheduler_config)
    
    gpu_id = model_spec['gpu_id']
    torch.cuda.set_device(gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()

    for epoch in range(data_split_config['num_epochs']):
        model.train()
        start_time = time.time()
        
        for i, data in enumerate(train_loader):
            # inputs, labels = inputs.to(device), labels.to(device)
            data = data.to(device)

            # 前向传播
            outputs = model(data.x, data.edge_index)
            
            # print(f"outputs = {type(outputs)}. data.y = {type(data.y)}. data.train_mask = {type(data.train_mask)}")
            # 计算损失
            loss = criterion(outputs[data.train_mask], data.y[data.train_mask])

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 打印进度
            if (i+1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{data_split_config['num_epochs']}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # 调整学习率
        if scheduler:
            scheduler.step()

        # 记录和打印相关信息
        end_time = time.time()
        print(f"Epoch [{epoch+1}/{data_split_config['num_epochs']}], Time: {end_time - start_time:.2f}s")

    return model, train_loader, test_loader

# %%

def launch_train_with_eval(model = "gcn", workload = "stats", signature = "0"):
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        model: 
        train_loader: 
        test_loader:
    """
    gpu_id = model_spec['gpu_id']
    torch.cuda.set_device(gpu_id)
    file_content = f"{workload}_0.pkl"
    dataset = query_dataset.QueryGraphDataset(workload=workload, \
        file_content = file_content, signature=signature)
    
    train_loader, test_loader = get_train_test_loader(\
        dataset=dataset, config=data_split_config)

    num_features = dataset.get_node_feature_number()

    if model == "gcn":
        model = gcn_model.GCNModel(num_features=num_features, \
            hidden_channels_1=200, hidden_channels_2=100, embedding_dims=50)
    elif model == "gat":
        model = gat_model.GATModel(num_features=num_features, \
            hidden_channels_1=200, hidden_channels_2=100, embedding_dims=50)
    else:
        raise ValueError(f"launch_train_with_eval: Unsupported model({model})")
    
    optimizer = get_optimizer(model=model, config=optimizer_config)
    scheduler = get_lr_scheduler(optimizer, scheduler_config)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()

    for epoch in range(data_split_config['num_epochs']):
        model.train()
        start_time = time.time()
        
        for i, data in enumerate(train_loader):
            # inputs, labels = inputs.to(device), labels.to(device)
            data = data.to(device)

            # 前向传播
            outputs = model(data.x, data.edge_index)
            
            # print(f"outputs = {type(outputs)}. data.y = {type(data.y)}. data.train_mask = {type(data.train_mask)}")
            # 计算损失
            loss = criterion(outputs[data.train_mask], data.y[data.train_mask])

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 打印进度
            if (i+1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{data_split_config['num_epochs']}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # 调整学习率
        if scheduler:
            scheduler.step()

        # 记录和打印相关信息
        end_time = time.time()
        print(f"Epoch [{epoch+1}/{data_split_config['num_epochs']}], Time: {end_time - start_time:.2f}s")

        model.eval()
        test_loss = 0.0
        all_predictions, all_targets = [], []

        for i, data in enumerate(test_loader):
            data = data.to(device)
            with torch.no_grad():
                test_outputs = model(data.x, data.edge_index)
                # print(f"{test_outputs[data.test_mask]}")
                # print(f"{data.y[data.test_mask].detach().tolist()}")
                test_loss += criterion(test_outputs[data.test_mask], data.y[data.test_mask]).item()
                all_predictions.append(test_outputs[data.test_mask].cpu().numpy())
                all_targets.append(data.y[data.test_mask].cpu().numpy())
        
        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)

        # 计算R2分数
        r2 = r2_score(all_targets, all_predictions)
        avg_test_loss = test_loss / len(test_loader)
        print(f"Epoch [{epoch+1}/{data_split_config['num_epochs']}], Test Loss: {avg_test_loss:.4f}"\
              f", R2 Score: {r2: .4f}")

    torch.save(model.state_dict(), 'model.pth')

    return model, train_loader, test_loader
