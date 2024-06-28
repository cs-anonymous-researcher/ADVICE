#!/usr/bin/env python
# -*- coding: utf-8 -*-


# %%

from utilities.global_config import *
import time
import numpy as np
from sklearn.metrics import r2_score

from models import deep_ensemble_model
from training.optimizer import get_optimizer, get_lr_scheduler

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
# import pytorch_lightning as pl
from utilities import utils
from copy import deepcopy
import torch.nn.functional as F


# %%
data_range = 1.0
def fgsm_attack(image, epsilon, data_grad):
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon*sign_data_grad
    return perturbed_image

# %%

def ensemble_model_train(model_list: list[deep_ensemble_model.GaussianGNN], 
    train_loader: DataLoader, test_loader: DataLoader, output_path: str):
    """
    {Description}

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    for model in model_list:
        optimizer = get_optimizer(model=model, config=optimizer_config)
        scheduler = get_lr_scheduler(optimizer, scheduler_config)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        for epoch in range(data_split_config['num_epochs']):
            model.train()   
            for i, data in enumerate(train_loader):
                data = data.to(device)
                x, y = data.x, data.y
                x.requires_grad = True
                # 前向传播
                mean, var = model(x, data.edge_index)

                loss_for_adv = F.gaussian_nll_loss(mean, y, var)
                grad = torch.autograd.grad(loss_for_adv, x, retain_graph=False)[0]
                x, y = x.detach(), y.detach()
                mean, var = mean.detach(), var.detach()
                loss_for_adv.detach_()

                perturbed_data = fgsm_attack(x, 0.01 * data_range, grad) # 8 is range of data
                # print(f"ensemble_model_train: x.shape = {x.shape}. perturbed_data.shape = {perturbed_data.shape}.")

                mean_adv, var_adv = model(perturbed_data, data.edge_index)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 100)

                loss = F.gaussian_nll_loss(mean, y, var) + F.gaussian_nll_loss(mean_adv, y, var_adv)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # 调整学习率
        if scheduler:
            scheduler.step()

    state_dict_list = []
    for model in model_list:
        state_local = deepcopy(model.state_dict())
        state_dict_list.append(state_local)

    utils.dump_pickle(state_dict_list, output_path)


def ensemble_model_load(model_list: list[deep_ensemble_model.GaussianGNN], model_path):
    """
    加载ensemble模型

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    state_dict_list = utils.load_pickle(model_path)
    for model, state_dict in zip(model_list, state_dict_list):
        # 训练单个模型
        model.load_state_dict(state_dict)
    return model_list

def ensemble_model_inference(model_list: list[deep_ensemble_model.GaussianGNN], test_dataset):
    """
    {Description}

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    means, variances = [], []
    for model in model_list:
        model.eval()
        preds, sigmas = model(test_dataset.x, test_dataset.edge_index)
        means.append(preds)
        variances.append(sigmas)

    means = torch.tensor(means)
    mean = means.mean()
    variance = (variances +  means.pow(2)).mean(0) - mean.pow(2)
    std = np.sqrt(variance.numpy())
    return mean, std

# %%
