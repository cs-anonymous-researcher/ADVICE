#!/usr/bin/env python
# -*- coding: utf-8 -*-

# %%

import numpy as np
import pandas as pd
from os.path import join as p_join
import scipy
from matplotlib import pyplot as plt
import json, os, time, re, math
import pickle

# %%

# import inner_correlation_multi as inner_correlation
from grid_manipulation import grid_construction
from data_interaction import data_management

# %%


class TableExplorer(object):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, dm_ref:data_management.DataManager, workload:str, table_name:str):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        self.dm_ref = dm_ref
        self.workload = workload
        self.table_name = table_name
        column_list, meta_dict = self.load_table_instance()
        self.column_feature_dict = {}
        column_order = self.analyze_candidate_columns(column_list, meta_dict)   # 列的重要性顺序
        self.column_order = column_order                                        # 保存顺序
        # self.search_graph = SearchGraph((meta_dict, column_order), self.data_df)   # 将column info传进

    def get_column_features(self, col_name):
        """
        获得一个列的相关特征，包括distinct value, kurtosis和skewness，
        之后考虑再补充，返回一个list
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        col_data = self.data_df[col_name].dropna(how='any').values
        # print("col_data = {}".format(col_data))
        # print("col_data name = {}. type = {}".format(col_name, col_data.dtype))
        if len(col_data) <= 1000:   # 数目过小直接退出
            return []
        # col_data = col_data.astype("Int64")
        # col_data = col_data.astype("Int64")
        col_data = col_data[~np.isnan(col_data)]
        
        domain_size = len(np.unique(col_data))
        kurtosis = scipy.stats.kurtosis(col_data)
        skewness = scipy.stats.skew(col_data)

        return [domain_size, kurtosis, skewness]

    def rank_all_candidates(self,):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        # column_feature_list = [(k, v) for k, v in self.column_feature_dict.items()]
        # 针对每一项进行排序，再把得分加总起来
        column_list, features_list = zip(*self.column_feature_dict.items())
        # print("column_list = {}".format(column_list))
        score_list = [0 for _ in column_list]

        def assign_res(feature_vec):
            # 根据特征大小顺序分配结果
            # print("score_list = {}".format(score_list))
            # print("feature_vec = {}".format(feature_vec))

            sorted_idx = np.argsort(feature_vec)
            for val, idx in enumerate(sorted_idx):
                score_list[idx] += val

        def sorted_list(data_list, weight_arr):
            return list(np.array(data_list)\
                [np.argsort(weight_arr)])

        for single_feature in list(zip(*features_list)):
            assign_res(single_feature)

        column_order = sorted_list(column_list, score_list)
        return column_order

    def analyze_candidate_columns(self, column_list, meta_dict):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """

        # print("meta_dict = {}".format(meta_dict))
        cand_list = []
        foreign_columns = []
        string_columns = []

        for src_col, ref_tbl, ref_col in meta_dict['foreign']:
            foreign_columns.append(src_col)

        for col, dtype in meta_dict['data_type']:
            if "character" in dtype: 
                string_columns.append(col)

        for col in column_list:
            if col == meta_dict['primary'] or col in \
                foreign_columns or col in string_columns:
                continue
            else:
                cand_list.append(col)

        # print("cand_list = {}".format(cand_list))

        ts = time.time()
        self.load_column_features(cand_list)
        te = time.time()
        # print("get all column features = {}".format(te - ts))

        self.column_order = self.rank_all_candidates()
        return self.column_order

    def load_column_features(self, cand_list):
        path_kw = "column_features"
        features_path = p_join(self.meta_dict["src_path"], \
            path_kw, "{}.pkl".format(self.table_name))

        # 如果没有parent路径的话则创建路径
        os.makedirs(name=p_join(self.meta_dict['src_path'], \
            path_kw), exist_ok=True)

        if self.meta_dict['reuse_features'] == False or \
            os.path.isfile(features_path) == False:
            for col in cand_list:
                feature_list = self.get_column_features(col)
                if len(feature_list) == 0:
                    # print("127127127")
                    continue
                self.column_feature_dict[col] = feature_list
            with open(features_path, "wb") as f_out:
                pickle.dump(self.column_feature_dict, f_out)
        else:
            with open(features_path, "rb") as f_in:
                self.column_feature_dict = pickle.load(f_in)

        return self.column_feature_dict
                
    def load_table_instance(self,):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        self.data_df = self.dm_ref.load_table(self.table_name)
        self.meta_dict = self.dm_ref.get_table_meta(self.table_name)
        # self.candidate_columns = self.analyze_candidate_columns(self.data_df.columns, self.meta_dict)
        return self.data_df.columns, self.meta_dict

    def make_array_data(self, column_list):
        """
        根据column list构造ndarray，去除了含有NULL的data tuple
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        return self.data_df[column_list].dropna(how='any').values

    def get_evaluation_result(self, data_arr, split_size_list):
        """
        首先获得abnormal region，
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        # result = inner_correlation.abnormal_grid_in_array(data_arr, split_size = split_size_list)

    def add_column(self, col_name):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        pass


    def delete_column(self, col_name):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        pass

