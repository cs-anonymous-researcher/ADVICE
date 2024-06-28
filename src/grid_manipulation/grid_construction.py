#!/usr/bin/env python
# -*- coding: utf-8 -*-

# %%

import numpy as np
import pandas as pd
from os.path import join as p_join
from matplotlib import pyplot as plt
import json, os, time, re, math
import pickle
import hashlib

# 新的函数
from multiprocessing import Pool
from functools import reduce
from copy import copy, deepcopy

# %%
from scipy.stats import rankdata
from copy import copy
from scipy.ndimage import convolve
from utility.utils import general_cache_dump, general_cache_load, objects_signature
from utility import utils
 
normalized_arr = None   # 全局变量

def kernel_create(arr_shape, size = 3):
    # 核的创建
    input_dims = len(arr_shape)
    size_list = []
    for num in arr_shape:
        if num >= size:
            size_list.append(size)
        else:
            size_list.append(1)

    kernel = np.ones(size_list)
    kernel = kernel / kernel.size
    return kernel


def get_data_hash(data_arr: np.ndarray, split_size_list: list):
    """
    {Description}
    
    Args:
        data_arr:
        split_size:
    Returns:
        hash_val:
    """

    data_view = np.ascontiguousarray(data_arr).view(np.uint8)
    data_hash = hashlib.sha1(data_view).hexdigest()
    # 重写split_hash的生成方法，采用迭代的策略
    h = hashlib.blake2b()
    for item in split_size_list:
        h.update(bytes(item))
    split_hash = h.hexdigest()
    input_hash = data_hash[:8] + split_hash[:8]
    return input_hash

@utils.timing_decorator
def construct_data_info(data_arr:np.ndarray, bins_list, cache_path = "../tmp/info"):
    """
    make_grid_data的辅助函数，用于构建关键信息，
    添加cache减少重复计算的开销
    
    Args:
        data_arr:
        split_size_list:
    Returns:
        distinct_list:
        marginal_list:
    """

    column_num = data_arr.shape[1]    
    assert len(data_arr.shape) == 2     # 
    input_len = data_arr.shape[0]

    distinct_list, marginal_list = [], []   # 获得每一维上的边缘分布
    for col_idx in range(column_num):
        local_distinct, local_marginal = np.unique(np.digitize(\
            data_arr[..., col_idx], bins_list[col_idx], right=True), return_counts = True)
        # 由于bins_list不是随时生成的，需要补充缺失的部分
        # local_distinct的预期值为0, 1, 2, ..., n-1
        
        # print("len(local_distinct) = {}. len(local_marginal) = {}. len(bins_list[col_idx]) = {}"\
        #     .format(len(local_distinct), len(local_marginal), len(bins_list[col_idx])))
        if len(bins_list[col_idx]) - 1 != len(local_distinct):
            complement_distinct = range(1, len(bins_list[col_idx]))    # 补全的distinct
            complement_marginal = []    # 补全的marginal

            curr_idx = 0
            for idx, val in enumerate(complement_distinct):
                if curr_idx != len(local_distinct) and local_distinct[curr_idx] == val:
                    complement_marginal.append(local_marginal[curr_idx])
                    curr_idx += 1
                elif curr_idx == len(local_distinct) or local_distinct[curr_idx] > val:
                    complement_marginal.append(0)
                elif local_distinct[curr_idx] < val:
                    raise ValueError("local_distinct[curr_idx] = {}. val = {}".\
                        format(local_distinct[curr_idx], val))
            
            # print("make list complement. len(complement_distinct) = {}. \
            #     len(complement_marginal) = {}.".format(len(complement_distinct), len(complement_marginal)))
            distinct_list.append(deepcopy(complement_distinct))
            marginal_list.append(deepcopy(complement_marginal))

        else:
            distinct_list.append(local_distinct)
            marginal_list.append(local_marginal)

    # print(distinct_list)
    return distinct_list, marginal_list




def explore_dim(curr_data, current_dim, prev_path, value_cnt_arr):
    """
    {Description}
    
    Args:
        data_arr: 数据矩阵
        current_dim: 当前需要处理的维度
        prev_path: 过去的路径
    Returns:
        None
    """
    if current_dim == column_num:
        # 到达终点，直接退出
        value_cnt_arr[tuple(prev_path)] += len(curr_data)    # 结果赋值
        return len(prev_path)

    # print("curr_data.shape = {}".format(curr_data.shape))
    curr_data = curr_data[curr_data[:, 0].argsort()] # 按照第一维排序
    curr_distinct, curr_marginal = np.unique(np.digitize(curr_data[..., 0], \
        bins_list[current_dim], right=True), return_counts = True)

    # 这里的local_distinct和global_distinct之间还存在差异，idx这里不是单调incremental的
    global_distinct = distinct_list[current_dim]
    idx_list = []

    local_idx = 0
    # 匹配全局索引和局部的索引
    for global_idx, val in enumerate(global_distinct):
        if local_idx == len(curr_distinct):
            # 超出range，直接退出
            break
        if val == curr_distinct[local_idx]:
            idx_list.append(global_idx)
            local_idx += 1
        elif val < curr_distinct[local_idx]:
            continue
        else:
            raise ValueError("global_val > local_val")

    curr_pos_list = list(np.cumsum(curr_marginal))
    curr_idx_pairs = list(zip([0] + curr_pos_list[:-1], curr_pos_list))

    # 确定start_idx, end_idx这样的pairs
    for idx, (start, end) in zip(idx_list, curr_idx_pairs):
        new_path = copy(prev_path)
        new_path.append(idx)
        explore_dim(curr_data[start: end, 1:], \
                    current_dim + 1, new_path, value_cnt_arr)
    return True

column_num = 0                  # 总的列数目
global_data_arr = np.array([])         # 处理后的数据矩阵

marginal_list = []
bins_list = []
distinct_list = []

def func_entry(pair):
    start, end = pair
    value_cnt_arr = np.zeros([len(i) for i in marginal_list])    # 这里变成了多维数组，只和marginal_list个数相关
    res = explore_dim(global_data_arr[start: end, ...], current_dim = 0, \
        prev_path = [], value_cnt_arr=value_cnt_arr)
    return value_cnt_arr

@utils.timing_decorator
def make_grid_data(data_arr:np.ndarray, input_bins, process_num = 40):
    """
    构造网格数据，每一个dimension都按值大小进行等深的划分
    20230204: 采用多线程加速explore_dim的过程，让总的探索时间从20s优化到5s
    
    Args:
        data_arr:
        input_bins:
    Returns:
        distinct_list: 
        marginal_list: 
        value_cnt_arr:
    """
    global column_num
    column_num = data_arr.shape[1]

    # 获得数据的相关处理后信息
    global marginal_list, bins_list, distinct_list
    bins_list = input_bins
    # t0 = time.time()
    distinct_list, marginal_list = construct_data_info(data_arr, bins_list)

    # 验证distinct_list, marginal_list的合法性
    for i, j, k in zip(distinct_list, marginal_list, bins_list):
        len1, len2, len3 = len(i), len(j), len(k)
        if len1 != len2:
            raise ValueError("len1 != len2. len1 = {}. len2 = {}.".format(len1, len2))
        elif len2 != len3 - 1:
            raise ValueError("len2 != len3 - 1. len2 = {}. len3 = {}.".format(len2, len3))
        else:
            continue
    
    # t1 = time.time()
    # 设置全局数据
    global global_data_arr
    global_data_arr = data_arr

    # t2 = time.time()
    # 通过多进程来解决此问题
    pos_list = np.linspace(0, data_arr.shape[0], num = process_num + 1, dtype=int)
    pair_list = list(zip(pos_list[:-1], pos_list[1:]))
    
    # t3 = time.time()
    with Pool(processes = process_num) as p:
        res = p.map(func_entry, pair_list)
    value_cnt_arr = reduce(np.add, res, np.zeros_like(res[0]))
    # t4 = time.time()

    # print("make_grid_data: delta0 = {:.2f}. delta1 = {:.2f}. delta2 = {:.2f}. delta3 = {:.2f}.".\
    #       format(t1 - t0, t2 - t1, t3 - t2, t4 - t1))
    return distinct_list, marginal_list, value_cnt_arr


def select_abnormal_grid(value_cnt_arr, marginal_list, num = 100, mode = "under-estimation"):
    """
    选择异常的网格点，直接从value_cnt_arr中寻找结果
    
    Args:
        value_cnt_arr:
        marginal_list:
        num:
        mode:
    Returns:
        res_idx_list:
    """
    marginal_value_arr = make_marginal_value_arr(marginal_list) # 预处理，值放缩
    
    if mode == "under-estimation":
        ratio_arr = value_cnt_arr / marginal_value_arr          # 期望比例矩阵
    elif mode == "over-estimation":
        ratio_arr = marginal_value_arr / (value_cnt_arr + 1)
    else:
        raise ValueError("Unsupported select_abnormal_grid mode: {}".format(mode))

    res_idx_list = np.unravel_index(np.argsort(ratio_arr, \
        axis = None)[::-1], ratio_arr.shape)
    res_idx_list = list(zip(*res_idx_list))
    return res_idx_list[:num]

def make_marginal_value_arr(marginal_list):
    marginal_copy = deepcopy(marginal_list)
    for idx in range(1, len(marginal_copy)):                # 预处理，值放缩
        marginal_copy[idx] = marginal_copy[idx] / np.sum(marginal_copy[idx])
    marginal_value_arr = np.prod(np.ix_(*marginal_copy))  
    return marginal_value_arr


# %%

"""
优化步骤
1. distinct value少的情况(<split_size)，需要特殊处理
2. skew distribution的情况下，需要修改expected value，实际上是一个expected matrix
"""


def abnormal_grid_in_array(data_arr:np.ndarray, input_bins, num = 50):
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    distinct_list, marginal_list, value_cnt_arr = \
        make_grid_data(data_arr, input_bins = input_bins)
    t1 = time.time()
    grid_res = select_abnormal_grid(value_cnt_arr, marginal_list, num = num)
    t2 = time.time()
    print("select_abnormal_grid: {}".format(t2 - t1))
    return grid_res, marginal_list, value_cnt_arr
