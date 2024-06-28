#!/usr/bin/env python
# -*- coding: utf-8 -*-


# %%
import numpy as np
import pandas as pd
from os.path import join as p_join
from matplotlib import pyplot as plt
import json, os, time, re, math
import pickle
from copy import deepcopy

# %% 添加src路径

import os, sys

src_path = "/home/lianyuan/Research/CE_Evaluator/src"
sys.path.append(src_path)    # 添加parent directory

# %%

import argparse
import pickle
from plan import plan_template
from data_interaction import data_management, mv_management
from query import query_exploration
from grid_manipulation import grid_preprocess

# %% 处理输入输出
def read_from_pickle(in_path):
    """
    {Description}

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    with open(in_path, "rb") as f_in:
        result = pickle.load(f_in)
    return result

def dump_to_pickle(out_obj, out_path):
    """
    {Description}

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    with open(out_path, "wb") as f_out:
        pickle.dump(out_obj, f_out)
    return True


# %%

def init_template_plan(input_args, output_path, split_budget = 100, dynamic_config = {}):
    """
    完成单个template_plan的初始化
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    out_obj = None

    if len(input_args) == 4:
        # 变量的含义分别是数据集、查询元信息、选择的列、基数估计错误模式
        workload, query_meta, selected_columns, mode = input_args
        print(f"init_template_plan: input_args[0] = {workload}.")
        print(f"init_template_plan: input_args[1] = {query_meta}.")
        print(f"init_template_plan: input_args[2] = {selected_columns}.")
        print(f"init_template_plan: input_args[3] = {mode}.")
        extra_info = {}
    elif len(input_args) == 5:
        workload, query_meta, selected_columns, mode, cond_bound_dict = input_args
        print(f"init_template_plan: input_args[0] = {workload}.")
        print(f"init_template_plan: input_args[1] = {query_meta}.")
        print(f"init_template_plan: input_args[2] = {selected_columns}.")
        print(f"init_template_plan: input_args[3] = {mode}.")
        print(f"init_template_plan: input_args[4] = {cond_bound_dict}.")
        extra_info = {
            "cond_bound": cond_bound_dict       # 构造时的迭代信息
        }
    else:
        raise ValueError(f"init_template_plan: len(input_args) = {len(input_args)}.")

    print(f"init_template_plan: workload = {workload}. mode = {mode}.")
    assert mode in ("over-estimation", "under-estimation")
    
    data_manager = data_management.DataManager(wkld_name=workload)
    mv_manager = mv_management.MaterializedViewManager(workload=workload)
    bins_builder = grid_preprocess.BinsBuilder(workload = workload, data_manager_ref=data_manager, mv_manager_ref=mv_manager)
    # 构建bins_dict的过程中支持自适应的split预算
    bins_dict = bins_builder.construct_bins_dict(selected_columns=selected_columns, split_budget=split_budget)
    reverse_dict = bins_builder.construct_reverse_dict(bins_dict=bins_dict)
    marginal_dict = bins_builder.construct_marginal_dict(bins_dict=bins_dict)

    query_ctrl = query_exploration.QueryController(workload=workload)
    
    table_builder = grid_preprocess.get_table_builder_by_workload(
        workload=workload, data_manager=data_manager, mv_manager=mv_manager, dynamic_config=dynamic_config
    )

    template_plan = plan_template.TemplatePlan(workload=workload, query_meta=query_meta, mode=mode, 
        bins_dict=bins_dict, ce_handler=None, reverse_dict=reverse_dict, marginal_dict=marginal_dict, 
        query_ctrl_ref=query_ctrl, bins_builder=bins_builder, table_builder_ref=table_builder, 
        extra_info=extra_info, split_budget=split_budget)
    
    # 在default_grid_plan的基础上创建额外的grid_plan
    plan_num = 4
    template_plan.create_extra_grid_plans(plan_num)
    grid_plan_id = template_plan.select_grid_plan(mode = "max_grid")
    template_plan.bind_grid_plan(grid_plan_id)

    # template_plan.grid_info_adjust()      # 调整grid_info的信息
    # template_plan.create_default_grid_plan()
    template_plan.release_df_memory()       # 释放用不到的内存
    template_plan.clean_table_buffer()      # 清理表的缓存
    
    out_obj = template_plan
    
    dump_to_pickle(out_obj=out_obj, out_path=output_path)

# %%
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', default="template_plan", choices = ["template_plan"])         # 模式
parser.add_argument('-i', '--input_path')   # 输入变量的路径
parser.add_argument('-o', '--output_path')       # 输出变量的路径
parser.add_argument('-c', '--config', default="{}")
parser.add_argument('-s', '--split_budget', default=100)

# %%

if __name__ == "__main__":
    args = parser.parse_args()
    # print("args.mode = {}.".format(args.mode))
    if args.mode == "template_plan":
        input_args = read_from_pickle(in_path=args.input_path)
        output_path = args.output_path
        dynamic_config = eval(args.config)
        split_budget = int(args.split_budget)
        init_template_plan(input_args=input_args, output_path=output_path, \
            split_budget=split_budget, dynamic_config=dynamic_config)
    else:
        error_str = f"basic_execution_unit: Unsupported mode = {args.mode}"
        raise ValueError(error_str)
    
# %%
