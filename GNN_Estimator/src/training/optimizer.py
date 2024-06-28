#!/usr/bin/env python
# -*- coding: utf-8 -*-


# %%
import torch.optim as optim


# %%

def get_optimizer(model, config):
    """
    {Description}

    Args:
        model (torch.nn.Module): 要优化的模型
        config (dict): 包含优化器设置的配置字典
    Returns:
        torch.optim.Optimizer: 配置好的优化器
    """
    optimizer_type = config.get('optimizer_type', 'SGD')
    learning_rate = config.get('learning_rate', 0.0001)
    weight_decay = config.get('weight_decay', 0)
    momentum = config.get('momentum', 0)

    if optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Invalid optimizer type: {optimizer_type}")

    return optimizer



# %%
def get_lr_scheduler(optimizer, config):
    """
    创建学习率调度器。

    参数：
        optimizer (torch.optim.Optimizer): 已创建的优化器。
        config (dict): 包含学习率调度器设置的配置字典。

    返回：
        torch.optim.lr_scheduler._LRScheduler: 配置好的学习率调度器。
    """
    scheduler_type = config.get('scheduler_type', None)
    step_size = config.get('step_size', 10)
    gamma = config.get('gamma', 0.1)

    if scheduler_type == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=step_size, eta_min=0)
    elif scheduler_type is None:
        return None
    else:
        raise ValueError(f"Invalid scheduler type: {scheduler_type}")

    return scheduler

# %%
