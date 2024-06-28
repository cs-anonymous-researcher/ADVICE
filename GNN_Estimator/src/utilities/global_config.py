#!/usr/bin/env python
# -*- coding: utf-8 -*-

# %%
from os.path import join as p_join

# %%

"""
{
    "description": {
        "overall": "项目的全局信息配置",
        "data_root": "",
        "meta_info_dir": ""
    },
    "data_root": "",
    "meta_info_dir": ""
}
"""

project_dir = "/home/lianyuan/Research/GNN_Estimator/"
 
# 在线数据的目录
# online_data_dir = "/home/jinly/GNN_Estimator/online_data"
online_data_dir = p_join(project_dir, "online_data")

# 数据根目录
# data_root = "/home/jinly/GNN_Estimator/data"
data_root = p_join(project_dir, "data")

# 配置信息目录
# meta_info_dir = "/home/jinly/GNN_Estimator/config"
meta_info_dir = p_join(project_dir, "config")

# 源代码目录
source_code_dir = p_join(project_dir, "src")

# 中间结果输出路径
output_dir = p_join(project_dir, "output")

# %%
import json

hyper_params_path = p_join(project_dir, "data", "hyper_parameters.json")

with open(hyper_params_path, "r") as f_in:
    hyper_params_dict = json.load(f_in)

# %%
python_path = "/home/lianyuan/anaconda3/envs/GNN_Estimator/bin/python"

model_spec = {
    "gpu_id": 0
}

data_split_config = {
    "num_epochs": 50,       # 训练批次
    "batch_size": 64,
    "train_ratio": 0.8,
    "shuffle": False,
    "num_workers": 10
}

optimizer_config = {
    "optimizer_type": "Adam",
    "learning_rate": 0.01
}

scheduler_config = {
    "scheduler_type": "StepLR"
}

# %%

import warnings
# 忽略所有警告
warnings.filterwarnings("ignore")

# %% 数据库的相关配置

port_number = 6432

JOB_config = {
    "user": "lianyuan",
    "host": "localhost",
    "password": "",
    "database": "imdbload", 
    "port": str(port_number)
}

STATS_config = {
    "user": "lianyuan",
    "host": "localhost",
    "password": "",
    "database": "stats", 
    "port": str(port_number)
}

DSB_config = {
    "user": "lianyuan",
    "host": "localhost",
    "password": "",
    "database": "dsb_5g", 
    "port": str(port_number)
}

workload_conn_option = {
    "job": JOB_config,
    "stats": STATS_config,
    "dsb": DSB_config
}

# %%
