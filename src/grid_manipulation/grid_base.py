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

from copy import copy, deepcopy
from utility.utils import point2range, point2range_batch
from utility import utils
from query import query_construction
from data_interaction.mv_management import \
    meta_schema_append, meta_filter_append, meta_copy


# %%

def process_single_dataframe(data_df, column_list, split_size):
    """
    处理单个的dataframe，column_list表示选择的条件
    
    Args:
        data_df:
        column_list:
        split_size:
    Returns:
        data_arr:
        bins_list:
    """
    bins_list = []
    if isinstance(split_size, int):
        split_size_list = [split_size for _ in column_list]
    elif isinstance(split_size, list):
        if len(column_list) != len(split_size):
            print("len(column_list) = {}. len(split_size) = {}".\
                format(len(column_list), len(split_size)))
            raise ValueError("split_size length mismatch")
        else:
            split_size_list = split_size
    else:
        raise TypeError("unsupported split_size type")

    data_df = data_df[column_list].dropna(how='any')    # 直接进行预处理了
    for col, split in zip(column_list, split_size_list):
        local_bins = construct_bins(data_df[col], split)
        bins_list.append(local_bins)

    data_arr = dataframe2ndarray(data_df, column_list)
    return data_arr, bins_list


def make_marginal_value_arr(marginal_list, full_join_size = -1):
    """
    {Description}
    
    Args:
        marginal_list:
        full_join_size:
    Returns:
        marginal_value_arr:
    """
    if len(marginal_list) == 1:
        # 如果是一维的情况，直接特判返回结果
        return marginal_list[0]

    if full_join_size == -1:
        # 这是单表的情况
        full_join_size = np.sum(marginal_list[0])           # 边缘分布的和
    marginal_copy = deepcopy(marginal_list)
    for idx in range(0, len(marginal_copy)):                # 预处理，值放缩
        marginal_copy[idx] = marginal_copy[idx] / np.sum(marginal_copy[idx])
    marginal_value_arr = np.prod(np.ix_(*marginal_copy)) * full_join_size
    return marginal_value_arr


def dataframe2ndarray(data_df, column_list):
    """
    处理单个的dataframe，column_list表示选择的条件
    
    Args:
        data_df:
        column_list:
    Returns:
        data_arr:
    """
    data_df = data_df[column_list].dropna(how='any')
    return data_df.values


def process_to_join_dataframes():
    """
    处理待连接的dataframes
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """


def construct_marginal(data_series: pd.Series, input_bins):
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    data_series = data_series.dropna()
    
    res_distinct, res_marginal = np.unique(np.digitize(\
        data_series.values, input_bins, right=True), return_counts=True)
    if len(res_marginal) + 1 != len(input_bins):
        # 如果结果长度不匹配，直接raise error
        print("res_distinct = {}.".format(res_distinct))
        print("res_marginal = {}.".format(res_marginal))
        print("input_bins = {}.".format(input_bins))

        raise ValueError("construct_marginal: len(res_marginal) = {}. len(input_bins) = {}.".\
                         format(len(res_marginal), len(input_bins)))
    return res_marginal


def construct_bins(data_series: pd.Series, bin_num: int):
    """
    构造bins，确保res_bins数永远比value_cnt_arr维度大1

    bins和value_cnt_arr的关系: bins中的值比value_cnt_arr对应的维度大1，
    后续会调用np.digitize(..., right=True)这样的函数，对于每个bins是左闭右开的。
    为此，最左边的value需要-1。
    因此，lower_bound需要调整+1，最左边的情况下也是需要+1。
    
    在对应到value_cnt_arr时，需要保证left_idx < right_idx，否则结果无法由
    value_cnt_arr算出来，并且计算时right_idx要-1。

    Args:
        data_series:
        bin_num:
    Returns:
        res_bins:
        res_marginal:
    """
    
    # 去掉nan的值
    data_series = data_series.dropna()
    data_series = data_series.astype(int)   # 转换成整型series

    # 获得唯一的值
    unique_values = np.sort(data_series.unique())
    if len(unique_values) > bin_num:
        sorted_df = data_series.sort_values()   # 排序
        idx_list = np.linspace(0, len(sorted_df) - 1, bin_num + 1, dtype=int)
        value_list = [sorted_df.iloc[i] for i in idx_list]
        res_bins = np.unique(value_list)
        
        res_bins[0] -= 1    # 第一个值调整，之后考虑自适应的生成bins
    else:
        # 选取所有的值
        res_bins = [unique_values[0] - 1, ] + list(unique_values)   # 添加一个前置值

    return res_bins


# %% bridge用于连接value_pair_list和query_meta


class QueryBuilder(object):
    """
    查询的构建者，支持PlaceHolder，可以根据输入列表生成一批的查询

    Members:
        field1:
        field2:
    """

    def __init__(self, existing_meta, workload = "job"):
        """
        {Description}

        Args:
            existing_meta:
            column_placeholder:
        """
        self.workload = workload
        self.existing_meta = existing_meta
        self.column_placeholder = []
        self.abbr_mapping = query_construction.abbr_option[workload]    # 别名映射
        self.inverse_mapping = {}
        for k, v in self.abbr_mapping.items():
            self.inverse_mapping[v] = k
        # self.wrapper = query_construction.SingleWrapper(single_df, existing_meta)

    def get_default_meta(self, default_start = 0, default_end = 0):
        """
        {Description}
        
        Args:
            None
        Returns:
            query_default_meta:
        """
        default_val_pair_list = [(default_start, default_end) for _ in self.column_placeholder]
        return self.generate_meta(val_pair_list = default_val_pair_list)

    def get_query_wrapper(self, query_meta):
        """
        获得查询的包装者
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        # print("query_meta = {}".format(query_meta))
        
        if hasattr(self, "wrapper") == False:
            single_df = pd.DataFrame([])
            # self.wrapper = query_construction.SingleWrapper(single_df, query_meta)
            self.wrapper = query_construction.get_single_wrapper_instance(\
                single_df, query_meta, workload=self.workload)
        else:
            self.wrapper.replace_meta(new_meta = query_meta)
        
        return self.wrapper

    def set_existing_meta(self, in_meta):
        """
        {Description}
        
        Args:
            in_meta:
        Returns:
            None
        """
        self.existing_meta = in_meta

    def set_single_conditions(self, schema, column_list):
        """
        column_list这个顺序也是value_pair_list的顺序
        
        Args:
            schema:
            column_list:
        Returns:
            column_placeholder:
        """
        # 清空历史数据
        self.column_placeholder = []
        for col in column_list:
            self.column_placeholder.append(\
                (self.abbr_mapping[schema], col))
        return self.column_placeholder

    def set_join_conditions(self, schema_left, column_left,
            schema_right, column_right):
        """
        设置连接的条件，左右各一列
        
        Args:
            arg1:
            arg2:
        Returns:
            column_placeholder:
        """
        self.column_placeholder = []

        for col in column_left:
            self.column_placeholder.append(\
                (self.abbr_mapping[schema_left], col))

        for col in column_right:
            self.column_placeholder.append(\
                (self.abbr_mapping[schema_right], col))

    def set_multi_conditions(self, column_info_list):
        """
        {Description}
    
        Args:
            column_info_list: 列信息的list，需要注意的是在这里column需要按顺序排列，
                和val_pair_list中的顺序保持一致
        Returns:
            column_placeholder:
        """
        for schema, column in column_info_list:
            self.column_placeholder.append(\
                (self.abbr_mapping[schema], column))

    def geneate_existing_meta(self,):
        """
        生成返回已经存在的meta
        
        Args:
            None:
        Returns:
            query_meta:
        """
        return meta_copy(self.existing_meta)

    def geneate_existing_query(self,):
        """
        生成已经存在的query
    
        Args:
            None
        Returns:
            query_text:
        """
        curr_meta = self.existing_meta
        wrapper = self.get_query_wrapper(curr_meta)
        query_text = wrapper.generate_current_query()

        return query_text

    def generate_meta(self, val_pair_list):
        """
        获得查询的元信息
        
        Args:
            val_pair_list:
        Returns:
            query_meta:
        """
        curr_meta = meta_copy(self.existing_meta)
        extra_schema = set()
        for col_info, val_pair in zip(self.column_placeholder, val_pair_list):
            tbl_abbr, col_name = col_info
            lower_bound, upper_bound = val_pair
            # print("col_info = {}. val_pair = {}.".format(col_info, val_pair))

            # 数据全部转int
            filter_list = [(tbl_abbr, col_name, int(lower_bound), int(upper_bound)), ]  
            curr_meta = meta_filter_append(curr_meta, filter_list)
            extra_schema.add(self.inverse_mapping[tbl_abbr])
        # print("extra_schema = {}".format(extra_schema))
        curr_meta = meta_schema_append(curr_meta, extra_schema)
        return curr_meta

    def generate_batch_meta(self, val_pair_list_batch):
        """
        {Description}
        
        Args:
            val_pair_list_batch:
        Returns:
            meta_list:
        """
        meta_list = []
        for val_pair_list in val_pair_list_batch:
            meta_list.append(self.generate_meta(val_pair_list))
        return meta_list

    def generate_query(self, val_pair_list):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        curr_meta = self.generate_meta(val_pair_list)
        wrapper = self.get_query_wrapper(curr_meta)
        query_text = wrapper.generate_current_query()
        return query_text

    def generate_batch_queries(self, val_pair_list_batch):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        query_list = []
        for val_pair_list in val_pair_list_batch:
            query_list.append(self.generate_query(val_pair_list))
        return query_list

class ValueTranslator(object):
    """
    变量值的还原

    Members:
        field1:
        field2:
    """

    def __init__(self, bins_list):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        self.bins_list = bins_list

    def convert2origin(self, pair_list, dtype = "dummy"):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        res_list = []
        if dtype == "dummy":
            func = lambda a:a
        elif dtype == "integer":
            func = lambda a:int(a)
        elif dtype == "float":
            func = lambda a:float(a)
        else:
            raise ValueError("Unrecognize dtype: {}".format(dtype))

        for local_pair, local_bins in zip(pair_list, self.bins_list):
            # print("local_pair = {}. local_bins = {}.".format(local_pair, local_bins))
            i, j = local_pair
            local_lower = local_bins[i]
            # if i != 0:
            local_lower += 1    # 考虑"<"的问题，下界自增1 
            local_upper = local_bins[j]
            res_list.append((local_lower, local_upper))

        return res_list

    def convert2origin_batch(self, pair_list_batch, dtype = "dummy"):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        return [self.convert2origin(pair_list, dtype = dtype) for \
            pair_list in pair_list_batch]


class PostProcessor(object):
    """
    计算出val_cnt_arr后的处理

    Members:
        field1:
        field2:
    """

    def __init__(self, value_cnt_arr, marginal_value_arr, bins_list):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        self.value_cnt_arr = value_cnt_arr
        self.marginal_value_arr = marginal_value_arr
        self.bins_list = bins_list
        self.ratio_arr = value_cnt_arr / (marginal_value_arr + 1)
        self.translator = ValueTranslator(bins_list)

    def index_conversion_batch(self, uni_index_list):
        """
        一维索引到多维索引的转换
        
        Args:
            uni_index_list:
        Returns:
            multi_index_list:
        """
        ratio_arr = self.ratio_arr
        multi_index_list = np.unravel_index(uni_index_list, shape = ratio_arr.shape)
        multi_index_list = list(zip(*multi_index_list))
        return multi_index_list

    def select_grids_by_multi_index(self, index_list):
        """
        通过多维索引来选择结果集
        
        Args:
            index_list:
        Returns:
            val_pair_list_batch:
            label_list:
        """
        translator = self.translator

        test_pair_list_batch = utils.point2range_batch(index_list)
        label_list = utils.restore_origin_label(self.value_cnt_arr, index_list, mode="point")
        val_pair_list_batch = translator.convert2origin_batch(pair_list_batch = \
            test_pair_list_batch, dtype="integer")

        return val_pair_list_batch, label_list

    def select_grids_by_index(self, index_list):
        return self.select_grids_by_1d_index(index_list)
    
    def select_grids_by_1d_index(self, index_list):
        """
        {Description}
    
        Args:
            index_list:
        Returns:
            val_pair_list_batch:
            label_list:
        """
        multi_index_list = self.index_conversion_batch(index_list)
        return self.select_grids_by_multi_index(index_list = multi_index_list)
    

    def select_region_by_index(self, index_pair_list):
        """
        根据index_pair_list获得region信息的单例
        
        Args:
            index_pair_list:
        Returns:
            val_pair_list:
            label:
        """
        translator = self.translator
        # 获得查询对应的原始标签
        label = utils.restore_label_by_pair_list(self.value_cnt_arr, \
            index_pair_list)
        val_pair_list = translator.convert2origin(pair_list = \
            index_pair_list, dtype="integer")
        return val_pair_list, label


    def select_regions_by_index_batch(self, index_pair_list_batch):
        """
        批处理，根据索引选择目标的区域
        
        Args:
            index_list_batch:
        Returns:
            val_pair_list_batch:
            label_list:
        """
        translator = self.translator
        # 获得查询对应的原始标签
        label_list = utils.restore_origin_label(self.value_cnt_arr, index_pair_list_batch, mode="range")   
        val_pair_list_batch = translator.convert2origin_batch(pair_list_batch = \
            index_pair_list_batch, dtype="integer")

        return val_pair_list_batch, label_list


    def generate_random_grids(self, num = 200, non_zero = True):
        """
        生成一批随机的格点

        Args:
            num:
            non_zero:
        Returns:
            sample_res: 
            val_pair_list_batch: 
            label_list:
        """
        value_cnt_arr, marginal_value_arr = self.value_cnt_arr, self.marginal_value_arr
        translator = self.translator

        ratio_arr = value_cnt_arr / (marginal_value_arr + 1)
        # 在这里把高维的数据展开成一维的
        total_idx_list = np.ravel_multi_index(np.where(ratio_arr > 1e-3), ratio_arr.shape)    # 选择特定范围的数据

        if len(total_idx_list) > num:
            sample_res = np.random.choice(total_idx_list, size = num, replace = False)
        else:
            sample_res = total_idx_list

        param_list = np.unravel_index(sample_res, shape = ratio_arr.shape)
        param_list = list(zip(*param_list))

        test_pair_list_batch = utils.point2range_batch(param_list)
        label_list = utils.restore_origin_label(self.value_cnt_arr, param_list, mode="point")
        val_pair_list_batch = translator.convert2origin_batch(pair_list_batch = \
            test_pair_list_batch, dtype="integer")

        return sample_res, val_pair_list_batch, label_list

    def generate_random_regions(self,):
        """
        随机生成一批region
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        pass