#!/usr/bin/env python
# -*- coding: utf-8 -*-


# %%
from utilities.global_config import *
import time
import numpy as np
from sklearn.metrics import r2_score

from models import dropout_bayes_model
from training.optimizer import get_optimizer, get_lr_scheduler
import torch
import torch.nn as nn

# %%

def sample_on_prob(mean_tensor: torch.Tensor, std_tensor: torch.Tensor, num):
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    mean_list, std_list = mean_tensor.tolist(), std_tensor.tolist()
    result_list = []

    for mean, std in zip(mean_list, std_list):
        local_vals = np.random.normal(mean, std, num)
        result_list.append(local_vals)

    return result_list


def pack_prob(mean_tensor: torch.Tensor, std_tensor: torch.Tensor):
    """
    将概率参数进行封装

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    mean_list, std_list = mean_tensor.tolist(), std_tensor.tolist()
    result_list = []

    for mean, std in zip(mean_list, std_list):
        result_list.append((mean, std))

    return result_list

# %%

def dropout_model_train(model: dropout_bayes_model.GNNModel, train_loader, test_loader, out_path):
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    # model.set_training_num(len(train_loader) * train_loader.)

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
                test_loss += criterion(test_outputs[data.test_mask], data.y[data.test_mask]).item()
                all_predictions.append(test_outputs[data.test_mask].cpu().numpy())
                all_targets.append(data.y[data.test_mask].cpu().numpy())
        
        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)

        # 计算R2分数
        try:
            r2 = r2_score(all_targets, all_predictions)
            avg_test_loss = test_loss / len(test_loader)
            print(f"Epoch [{epoch+1}/{data_split_config['num_epochs']}], Test Loss: {avg_test_loss:.4f}"\
                f", R2 Score: {r2: .4f}")
        except ValueError as e:
            print(e)
            pass

    # print(model)
    # 导出相关的参数
    params_path = out_path.replace(".pth", "_params.pkl")
    torch.save(model.state_dict(), out_path)
    model.dump_params(params_path)
    # torch.load()
    return model, train_loader, test_loader

def dropout_model_set(model: nn.Module):
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    for a in model.children():
        if a.children():
            dropout_model_set(a)
        if isinstance(a, nn.Dropout):
            a.train()

def inference_with_uncertainty(model, test_data, data_num, num_samples = 50, l2 = 0.01):
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        test_mean:
        test_std:
    """
    dropout_model_set(model)    # 将模型的dropout设置为开启模式

    with torch.no_grad():
        outputs = np.vstack([model(test_data.x, test_data.edge_index).\
            cpu().detach().numpy() for i in range(num_samples)])
        
    test_mean = outputs.mean(axis=0)
    test_variance = outputs.var(axis=0)
    tau = l2 * (1. - model.dropout_rate) / (2. * data_num * model.decay)

    test_variance += (1. / tau)
    test_std = np.sqrt(test_variance)
    
    nan_mask = np.isnan(test_mean) | np.isnan(test_std)

    test_mean = test_mean[~nan_mask]
    test_std = test_std[~nan_mask]

    # print(f"inference_with_uncertainty: outputs.shape = {outputs.shape}. "\
    #       f"test_mean = {test_mean}. test_std = {test_std}.")
    return test_mean, test_std

# %%
