#!/usr/bin/env python
# -*- coding: utf-8 -*-

# %%
from utilities.global_config import *

from data import query_dataset
from data.dataloader import get_train_test_loader
from training.optimizer import get_optimizer, get_lr_scheduler
from models.autoencoder_model import GNNAutoEncoder
import torch, time
import torch.nn as nn
import numpy as np
from sklearn.metrics import r2_score

# %%

def launch_train_with_eval(workload = "stats", signature = "0"):
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    model = GNNAutoEncoder(50, 100)
    # model
    gpu_id = model_spec['gpu_id']
    torch.cuda.set_device(gpu_id)
    file_content = f"{workload}_0.pkl"
    dataset = query_dataset.QueryGraphDataset(workload=workload, \
        file_content = file_content, signature=signature)
    
    train_loader, test_loader = get_train_test_loader(\
        dataset=dataset, config=data_split_config)

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
            
            # print(f"outputs = {type(outputs)}. data.y = {type(data.edge_label)}. "
            #       f"data.train_mask = {type(data.edge_train_mask)}")
            
            # 计算损失
            loss = criterion(outputs[data.edge_train_mask], data.edge_label[data.edge_train_mask])

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
            mask = data.edge_test_mask
            data = data.to(device)
            with torch.no_grad():
                test_outputs = model(data.x, data.edge_index)
                test_loss += criterion(test_outputs[mask], data.edge_label[mask]).item()
                all_predictions.append(test_outputs[mask].cpu().numpy())
                all_targets.append(data.edge_label[mask].cpu().numpy())

        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)

        print(f"epoch {epoch}: all_predictions = {all_predictions[:5]}. all_targets = {all_targets[:5]}.")

        # 计算R2分数
        r2 = r2_score(all_targets, all_predictions)
        avg_test_loss = test_loss / len(test_loader)
        print(f"Epoch [{epoch+1}/{data_split_config['num_epochs']}], Test Loss: {avg_test_loss:.4f}"\
              f", R2 Score: {r2: .4f}")

    torch.save(model.state_dict(), 'model.pth')
    return model, train_loader, test_loader
