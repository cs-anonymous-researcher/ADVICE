#!/usr/bin/env python
# -*- coding: utf-8 -*-

# %%

import numpy as np
import pandas as pd
from os.path import join as p_join
from matplotlib import pyplot as plt
import json, os, time, re, math
import pickle

# %%
from grid_manipulation import condition_exploration, grid_construction, grid_base
from data_interaction import data_management, mv_management
from query import query_construction
# %%



def get_inner_corr_instance(workload_name, table_name, selected_columns = None, \
    column_num = None, split_budget = 100, data_manager = None):
    """
    {Description}

    Args:
        workload_name:
        table_name:
        selected_columns:
        column_num:
        split_budget:
        data_manager:
    Returns:
        column_list:
        bins_list:
        value_cnt_arr:
        marginal_value_arr:
    """

    time1 = time.time()
    if data_manager is None:
        # 如果data_manager没有作为入参，临时创建一个信息
        data_manager = data_management.DataManager(wkld_name=workload_name)

    local_explorer = condition_exploration.TableExplorer(\
        dm_ref = data_manager, workload = workload_name, table_name = table_name)
    time2 = time.time()

    if selected_columns is None:
        if column_num is not None:
            # 选择作为条件的列
            total_num = len(local_explorer.column_order)
            selected_columns = local_explorer.column_order[:min(total_num, column_num)]      
        else:
            raise ValueError("Lack input: selected_columns is None and column_num is None")

    print("table = {}. selected_columns = {}.".format(table_name, selected_columns))
    # split_size = int(np.ceil(np.sqrt(split_budget)))    # 这里的split_budget还需要再处理

    if isinstance(split_budget, int):
        split_size = int(np.ceil(split_budget ** (1 / len(selected_columns))))
    elif isinstance(split_budget, (list, tuple)):
        split_size = split_budget

    data_arr, bins_list = grid_base.process_single_dataframe(data_df = local_explorer.data_df, \
        column_list=selected_columns, split_size=split_size)
    
    time3 = time.time()
    distinct_list, marginal_list, value_cnt_arr = \
        grid_construction.make_grid_data(data_arr = data_arr, input_bins = bins_list)
    time4 = time.time()
    """
    value_cnt_arr: 实际grid中每个格点的值
    marginal_value_arr: grid矩阵的边缘分布
    bins_merge: 每一维grid array中的实际值列表集合
    """
    marginal_value_arr = grid_base.make_marginal_value_arr(marginal_list)
    column_list = [(table_name, col) for col in selected_columns]    # 这里的column_list会用在后面的QueryBuilder上面
    time5 = time.time()

    print("delta1 = {:.2f}. delta2 = {:.2f}. delta3 = {:.2f}. delta4 = {:.2f}".format(\
        time2 - time1, time3 - time2, time4 - time3, time5 - time4))
    return column_list, bins_list, value_cnt_arr, marginal_value_arr



def get_external_instance(external_df, selected_columns, split_budget = 100, alias_mapping = {}):
    """
    外部的dataframe进行划分
    
    Args:
        external_df:
        selected_columns:
        split_budget:
        alias_mapping: 别名映射
    Returns:
        column_list: 
        bins_list: 
        value_cnt_arr:
        marginal_value_arr:
    """
    if isinstance(split_budget, int):
        split_size = int(np.ceil(split_budget ** (1 / len(selected_columns))))
    elif isinstance(split_budget, (list, tuple)):
        split_size = split_budget

    data_arr, bins_list = grid_base.process_single_dataframe(data_df = external_df, \
        column_list=selected_columns, split_size=split_size)    # 这一步感觉需要进行拆分，以支持多表的情况
    
    distinct_list, marginal_list, value_cnt_arr = \
        grid_construction.make_grid_data(data_arr = data_arr, input_bins = bins_list)
    """
    value_cnt_arr: 实际grid中每个格点的值
    marginal_value_arr: grid矩阵的边缘分布
    bins_merge: 每一维grid array中的实际值列表集合
    """
    marginal_value_arr = grid_base.make_marginal_value_arr(marginal_list)
    # column_list = [(table_name, col) for col in selected_columns]    # 这里的column_list会用在后面的QueryBuilder上面

    print("alias_mapping = {}.".format(alias_mapping))

    def parse_col(col_str):
        curr_alias, curr_schema = "", ""
        for k, v in alias_mapping.items():
            if col_str.startswith(v) and len(curr_alias) < len(v):
                curr_alias, curr_schema = v, k

        col_name = col_str[len(curr_alias) + 1: ]
        print("curr_schema = {}. col_name = {}.".format(curr_schema, col_name))
        return curr_schema, col_name

    # 这里的selected_columns的格式是{alias}_{column}，需要转化成原来的格式
    column_list = [parse_col(col_str) for col_str in selected_columns]    
    return column_list, bins_list, value_cnt_arr, marginal_value_arr

# %%

def add_table_on_meta(existing_meta, dm_ref:data_management.DataManager, table_name:str):
    """
    {Description}
    
    Args:
        existing_meta:
        dm_ref:
        table_name:
    Returns:
        out_meta:
    """
    table_meta = dm_ref.load_table_meta(tbl_name = table_name)
    out_meta = mv_management.meta_merge(left_meta=existing_meta, right_meta=table_meta)
    return out_meta

# %%

def infer_true_cardinality(query_meta, value_cnt_arr, column_order, reverse_dict, alias_reverse):
    """
    根据grid内容推断查询真实的基数

    Args:
        query_meta: 查询的元信息
        value_cnt_arr: 
        column_order: 
        reverse_dict:
        alias_reverse: 别名映射的反向字典
    Returns:
        true_card: 查询对应的真实基数
        return2:
    """
    true_card = 1

    def get_column_values(query_meta, column_order):
        # 获得指定列对应的值
        # 但是这里好像没处理为空的情况
        value_dict = {}
        _, filter_list = query_meta

        for item in filter_list:
            alias, column_name, lower_bound, upper_bound = item
            schema_name = alias_reverse[alias]
            # col_compound = "{}_{}".format(alias, column_name)
            # if col_compound in column_order:
            if (schema_name, column_name) in column_order:
                value_dict[(schema_name, column_name)] = \
                    lower_bound, upper_bound

        return value_dict


    def bound_location(lower_bound, upper_bound, column, reverse_dict):
        # 确定bins下最终的bound index
        lower_bound -= 1
        lower_idx, upper_idx = 0, 0
        reverse_local = reverse_dict[column]
        
        if lower_bound is not None:
            lower_idx = reverse_local[lower_bound]
        else:
            lower_idx = 0

        if upper_bound is not None:
            upper_idx = reverse_local[upper_bound]
        else:
            upper_idx = max(reverse_local.values())   # 选择为最大值

        return lower_idx, upper_idx
    
    value_dict = get_column_values(query_meta=query_meta, column_order=column_order)
    # print("infer_true_cardinality: value_dict.keys() = {}.".format(value_dict.keys()))
    # print("infer_true_cardinality: query_meta = {}. column_order = {}".\
    #       format(query_meta, column_order))
    
    # print("infer_true_cardinality: value_dict = {}.".format(value_dict))

    # 局部的索引
    local_index = []
    # column_order是compound names，需要进行转化
    for col in column_order:
        # 还原普通的schema_name以及column_name
        # schema_name, column_name = query_construction.parse_compound_column_name(\
        #     col, alias_reverse = alias_reverse)
        schema_name, column_name = col
        lower_bound, upper_bound = value_dict[(schema_name, column_name)]
        lower_idx, upper_idx = bound_location(lower_bound=lower_bound, \
            upper_bound=upper_bound, column=(schema_name, column_name), reverse_dict=reverse_dict)
        
        # print("lower_bound = {}. upper_bound = {}. lower_idx = {}. upper_idx = {}.".
        #       format(lower_bound, upper_bound, lower_idx, upper_idx))
        if lower_idx == -1 and upper_idx == -1:
            local_index.append(Ellipsis)
        else:
            local_index.append(slice(lower_idx, upper_idx))
    # 真实基数的求和
    # print("local_index = {}.".format(local_index))
    # print("selected value_cnt_arr = {}.".format(value_cnt_arr[tuple(local_index)]))
    true_card = np.sum(value_cnt_arr[tuple(local_index)])
    
    return true_card


# %%

# # %%
# class GridManager(object):
#     """
#     基于查询模板的格点管理器

#     Members:
#         query_template:
#         field2:
#     """

#     def __init__(self, template_meta):
#         """
#         {Description}

#         Args:
#             template_meta: 查询模板的元信息
#             arg2:
#         """
#         self.template_meta = template_meta

#     def func_name1(self,):
#         """
#         {Description}

#         Args:
#             arg1:
#             arg2:
#         Returns:
#             return1:
#             return2:
#         """
#         pass


#     def func_name2(self,):
#         """
#         {Description}

#         Args:
#             arg1:
#             arg2:
#         Returns:
#             return1:
#             return2:
#         """
#         pass