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
from collections import Counter, defaultdict
# %%

from query import query_exploration
from copy import deepcopy, copy
from data_interaction.mv_management import meta_filter_add, meta_copy

from query import query_construction
from data_interaction import data_management
from collections import defaultdict
from copy import copy, deepcopy
from utility import utils

from grid_manipulation import grid_base, grid_preprocess

# %%

class ConditionGenerator(object):
    """
    根据要求生成一些简单的条件，目前只支持单列的生成

    Members:
        field1:
        field2:
    """

    def __init__(self, in_df, in_meta, tbl_abbr, column):
        """
        {Description}

        Args:
            in_df:
            in_meta:
            column:
        """
        self.in_df = in_df
        self.in_meta = meta_copy(in_meta)
        self.column, self.tbl_abbr = column, tbl_abbr
        self.sorted_col_values = \
            np.sort(self.in_df[self.column].dropna().values.astype('Int64'))

    def get_conditional_meta(self, start_val, end_val):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        local_meta = meta_copy(self.in_meta)
        local_meta = meta_filter_add(local_meta, \
            (self.tbl_abbr, self.column, start_val, end_val))

        return local_meta

    def grid_condition(self, num):
        """
        生成网格状的条件

        Args:
            num:
        Returns:
            result_list: (meta, card)组成的列表
        """
        result_list = []
        bin_num = num
        sorted_values = self.sorted_col_values
        idx_list = np.linspace(0, len(sorted_values) - 1, bin_num + 1, dtype=int)
        value_list = [sorted_values[i] for i in idx_list]
        res_bins = np.unique(value_list)
        res_bins[0] -= 1    # 第一个值调整，之后考虑自适应的生成bins

        # 获得数据划分的结果
        res_distinct, res_marginal = np.unique(\
            np.digitize(sorted_values, res_bins, right=True), return_counts = True)
        
        for idx, cnt in enumerate(res_marginal):
            start_val, end_val = res_bins[idx] + 1, res_bins[idx + 1]
            local_meta = self.get_conditional_meta(int(start_val), int(end_val))
            result_list.append((local_meta, cnt))
        return result_list

    def random_condition(self, num, selectivity):
        """
        生成随机性的条件

        Args:
            num:
            selectivity:
        Returns:
            result_list: (meta, card)组成的列表
        """
        sorted_values = self.sorted_col_values

        target_size = int(selectivity * len(sorted_values))
        result_list = []
        for i in range(num):
            # 确定start和end
            start = np.random.randint(0, len(sorted_values) - target_size)
            end = start + target_size
            start_val, end_val = \
                sorted_values[start], sorted_values[end]
            start_idx, end_idx = np.searchsorted(sorted_values, start_val, side='left'), \
                np.searchsorted(sorted_values, end_val, side='right')   # 搜索值的位置

            local_meta = self.get_conditional_meta(int(start_val), int(end_val))
            result_list.append((local_meta, end_idx - start_idx))
            
        return result_list


# %%
"""
随机查询的生成器

由一个概率表控制生成谓词

每一个查询相当于一个概率表的采样，目前概率表包含如下的内容：

1. 每一个表所产生的行数（均匀生成、log分布生成）
2. 每一个表所选列的概率（两项分布）

{

}
"""

class QueryGenerator(object):
    """
    根据要求生成一批随机的查询，column的谓词通过bins_dict进行生成

    Members:
        field1:
        field2:
    """

    def __init__(self, schema_list, dm_ref: data_management.DataManager, bins_builder: grid_preprocess.BinsBuilder, *, \
                 table_prob_config = None, column_prob_config = None, workload = "job", split_budget = 100):    
        """
        {Description}

        Args:
            schema_list:
            dm_ref:
            bins_builder:
            prob_table_config:
            workload:
        """
        self.dm_ref = dm_ref
        self.bins_builder = bins_builder
        self.schema_list = schema_list
        self.split_budget = split_budget    # 全区划分的预算
        
        if workload == "job":
            self.abbr_mapping = query_construction.JOB_abbr
        else:
            self.abbr_mapping = query_construction.abbr_option[workload]

        self.alias_mapping = self.abbr_mapping
        self.alias_inverse = {}
        for k, v in self.alias_mapping.items():
            self.alias_inverse[v] = k

        self.table_dict = {}
        self.schema_info = self.get_schema_info(schema_list)
        self.workload = workload
        # 旧的predicate获取方式
        # self.sorted_columns = self.init_sorted_columns(schema_list)   # 排好序的列
        # self.load_all_tables(schema_list)                             # 加载所有的表

        # 新的基于bins_dict获取方式
        self.bins_dict = self.init_bins_dict(schema_list, split_budget=split_budget)

        # 每一个列被选择的概率以及一个列上谓词大小
        if table_prob_config is None:
            self.table_prob_dict = self.init_prob_table()      # 加载表下的列分布概率
        else:
            self.table_prob_dict = table_prob_config
        
        if table_prob_config is None:
            self.column_prob_dict = self.init_prob_column()   # 加载列的值分布概率
        else:
            self.column_prob_dict = column_prob_config

        # print("当前的概率表为: ")
        # pprint.pprint(self.prob_table)


    def set_column_spec(self, column_list):
        """
        设置column的具体情况
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        pass

    def init_bins_dict(self, schema_list, split_budget = 200):
        """
        初始化一整个bins_dict
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        selected_columns = []
        # 把所有有效的列都加进来，后续考虑cache加速
        for schema in schema_list:
            valid_columns = self.schema_info[schema]
            for column in valid_columns:
                selected_columns.append((schema, column))

        ts = time.time()
        bins_dict = self.bins_builder.construct_bins_dict(selected_columns=\
                        selected_columns, split_budget=split_budget)
        te = time.time()
        # print("init_bins_dict: delta_time = {}.".format(te - ts))

        return bins_dict

    def get_schema_info(self, schema_list):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        res_dict = {}
        for s in schema_list:
            res_dict[s] = self.dm_ref.get_valid_columns(s)
        return res_dict
    
    def init_prob_column(self):
        """
        加载列的相关概率
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        prob_column = {}
        dm_ref = self.dm_ref

        for s in self.schema_list:
            local_columns = dm_ref.get_valid_columns(s)
            for c in local_columns:
                bins_local = self.bins_dict[(s, c)]
                # 一开始根据bins_local的size，设置等概率分布
                prob_column[(s, c)] = {}
                total_size = len(bins_local) - 1
                for i in range(1, total_size):
                    # 设置每一列大小的概率
                    prob_column[(s, c)][i] = 1 / total_size

        return prob_column

    def init_prob_table(self):
        """
        加载概率表，主要的解析合法的列，规则包含数据类型
        以及非NULL的数目，针对column选择的概率，以及每个column下面
        TODO: 之后考虑会加上多个Column的配置情况

        Args:
            prob_table_config:
        Returns:
            return1:
            return2:
        """
        prob_table = {}
        dm_ref = self.dm_ref

        for s in self.schema_list:
            local_dict = {}
            local_columns = dm_ref.get_valid_columns(s)
            col_num = len(local_columns)
            try:
                col_prob = 1.0 / col_num
                for c in local_columns:
                    local_dict[c] = col_prob
                prob_table[s] = copy(local_dict)
            except Exception as e:
                print(f"func_name: meet ZeroDivisionError. schema = {s}. local_columns = {local_columns}.")
                raise e

                
        return prob_table

    def random_range(self, tbl_name, col_name, col_size):
        """
        随机选取一个列的某个范围，按照均匀分布
        
        Args:
            tbl_name:
            col_name:
            col_size: 大小范围 [0,self.split_budget]
        Returns:
            res1:
            res2:
        """
        # lower_bound和upper_bound在这里需要分开处理，由于两边都包含
        # 等值查询，否则结果会出错
        bins_local = self.bins_dict[(tbl_name, col_name)]   
        
        # 针对col_size进行调整，如果值过大就取模
        if col_size >= len(bins_local):
            col_size = 1 + (col_size % (len(bins_local) - 1))

        start = np.random.randint(0, len(bins_local) - col_size)
        end = start + col_size
        return bins_local[start], bins_local[end]


    def prob_list_resize(self, in_prob_list):
        """
        {Description}
        
        Args:
            in_prob_list:
            arg2:
        Returns:
            out_prob_list:
            res2:
        """
        expected_sum = 1.0
        actual_sum = np.sum(in_prob_list)
        resize_factor = expected_sum / actual_sum

        out_prob_list = [i * resize_factor for i in in_prob_list]
        return out_prob_list
    
    def prob_dict_resize(self, in_prob_dict):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        out_prob_dict = {}
        key_list, value_list = self.dict2list(in_dict=in_prob_dict)
        value_list = self.prob_list_resize(in_prob_list=value_list)
        for k, v in zip(key_list, value_list):
            out_prob_dict[k] = v

        return out_prob_dict

    def data_smooth(self, in_value_dict):
        """
        {Description}
    
        Args:
            in_value_dict:
            arg2:
        Returns:
            out_value_dict:
            return2:
        """
        print("data_smooth: in_value_dict = {}.".format(in_value_dict))
        out_value_dict = {}
        value_list = []
        for k in sorted(in_value_dict.keys()):
            value_list.append(in_value_dict[k])

        box_pts = 3
        box = np.ones(box_pts)/box_pts
        print("value_list = {}. box = {}.".format(value_list, box))
        value_list = np.convolve(value_list, box, mode="same")
        for idx, k in enumerate(sorted(in_value_dict.keys())):
            out_value_dict[k] = value_list[idx]
        return out_value_dict


    def generate_batch_queries(self, num, with_meta = True):
        """
        {Description}
        
        Args:
            num: 生成查询的数目
            with_meta: 是否返回元信息
        Returns:
            query_list: 查询文本列表
            meta_list(optional): 元信息列表
        """
        query_list = []
        for _ in range(num):
            curr_query = self.generate_query(with_meta = with_meta)
            query_list.append(curr_query)

        # print("query_list = {}.".format(query_list))  # 事实表明生成的查询没有问题
        if with_meta == True:
            return list(zip(*query_list))
        else:
            return query_list


    def process_invalid_char(self, query_text:str, invalid_set = ["'", '"'], mode = "remove"):
        """
        {Description}
        
        Args:
            query_text:
            invalid_set:
            mode:
        Returns:
            query_result:
            res2:
        """
        query_result = query_text
        empty_char = ""
        if mode == "remove":
            for invalid_char in invalid_set:
                query_result = query_result.replace(invalid_char, empty_char)

        elif mode == "warning":
            for invalid_char in invalid_set:
                if invalid_char in query_result:
                    raise ValueError("Invalid {} in query {}.".format(invalid_char, query_text))
                
        return query_result
            
    def generate_query(self, with_meta = True):
        """
        生成查询

        Args:
            with_meta:
        Returns:
            query_text:
            query_meta(optional): 查询的元信息
        """
        config_instance = self.parse_prob_status()
        # print("config_instance = {}".format(config_instance))
        query_text = ""
        schema_list = self.schema_list
        filter_list = []

        # 生成filter_list
        for s, sub_dict in config_instance.items():
            for col, val_pair in sub_dict.items():
                lower_bound, upper_bound = val_pair
                filter_list.append((self.abbr_mapping[s], col, lower_bound, upper_bound))   

        # single_df = pd.DataFrame()
        single_meta = schema_list, filter_list
        # wrapper = query_construction.SingleWrapper(single_df, single_meta)
        # query_text = wrapper.generate_current_query()
        query_text = query_construction.construct_origin_query(query_meta=single_meta, workload=self.workload)

        # 检测/删除非法的符号
        query_text = self.process_invalid_char(query_text=query_text)

        if with_meta == True:
            return query_text, single_meta
        else:
            return query_text
    
    def dict2list(self, in_dict: dict):
        pair_list = [(k, v) for k, v in in_dict.items()]
        key_list, value_list = zip(*pair_list)
        return key_list, value_list

    def parse_prob_status(self,):
        """
        解析prob表，生成一个实例所需要的参数
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """

        config_dict = {}
        prob_table = self.table_prob_dict
        prob_column = self.column_prob_dict

        dict2list = self.dict2list
        for s in self.schema_list:
            schema_config = {}
            schema_prob = prob_table[s]
            column_list, prob_list = dict2list(schema_prob)
            selected_column = np.random.choice(a = column_list, p = prob_list)      # 等概率选一个column
            # print("schema = {}. selected_column = {}".format(s, selected_column))

            # 在Column下选择一个Condition
            # self.prob_column = 
            local_dict = prob_column[(s, selected_column)]
            col_size_list, col_prob_list = dict2list(in_dict=local_dict)
            # target_size = np.random.randint(low = 0, high = col_len)
            # print("col_size_list = {}.".format(col_size_list))
            # print("col_prob_list = {}.".format(col_prob_list))
            # print("prob_sum = {}.".format(np.sum(col_prob_list)))
            col_prob_list = self.prob_list_resize(col_prob_list)
            target_size = np.random.choice(a=col_size_list, p=col_prob_list)    # 设置目标的大小

            col_start, col_end = self.random_range(\
                tbl_name = s, col_name = selected_column, col_size = target_size)       # 确定起始和结束位置
            
            # 调整col_start和col_end以适应bins的结果
            col_start, col_end = int(col_start + 1), col_end
            schema_config[selected_column] = (col_start, col_end)
            config_dict[s] = copy(schema_config)

        return config_dict



# %%

class ComplexQueryGenerator(QueryGenerator):
    """
    更加复杂的查询生成器

    Members:
        field1:
        field2:
    """

    def __init__(self, schema_list, dm_ref: data_management.DataManager, bins_builder: grid_preprocess.BinsBuilder, *, \
                 table_prob_config = None, column_prob_config = None, workload = "job", split_budget = 100):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        super(ComplexQueryGenerator, self).__init__(schema_list, dm_ref, bins_builder, table_prob_config=table_prob_config, \
                    column_prob_config=column_prob_config, workload=workload, split_budget=split_budget)


    def generate_complex_query(self, with_meta = True):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        config_instance = self.parse_prob_status()
        # print("config_instance = {}".format(config_instance))
        query_text = ""
        schema_list = self.schema_list
        filter_list = []

        # 生成filter_list
        for s, sub_dict in config_instance.items():
            for col, val_pair in sub_dict.items():
                lower_bound, upper_bound = val_pair
                filter_list.append((self.abbr_mapping[s], col, lower_bound, upper_bound))   

        # single_df = pd.DataFrame()
        single_meta = schema_list, filter_list
        query_text = query_construction.construct_origin_query(query_meta=single_meta, workload=self.workload)

        # 检测/删除非法的符号
        query_text = self.process_invalid_char(query_text=query_text)

        if with_meta == True:
            return query_text, single_meta
        else:
            return query_text

    def construct_single_condition(self, schema, column):
        """
        生成单表的条件

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        prob_column = self.column_prob_dict
        dict2list = self.dict2list
    
        # 在Column下选择一个Condition
        local_dict = prob_column[(schema, column)]
        col_size_list, col_prob_list = dict2list(in_dict=local_dict)

        col_prob_list = self.prob_list_resize(col_prob_list)
        target_size = np.random.choice(a=col_size_list, p=col_prob_list)    # 设置目标的大小

        col_start, col_end = self.random_range(\
            tbl_name = schema, col_name = column, col_size = target_size)       # 确定起始和结束位置
        
        # 调整col_start和col_end以适应bins的结果
        col_start, col_end = int(col_start + 1), col_end
        return col_start, col_end

    def parse_prob_status(self, column_num_prob = {0: 0.2, 1: 0.6, 2: 0.2}):
        """
        解析prob表，生成一个实例所需要的参数
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        config_dict = {}
        prob_table = self.table_prob_dict

        for s in self.schema_list:
            schema_config = {}
            schema_prob = prob_table[s]

            # selected_column = np.random.choice(a = column_list, p = prob_list)      # 等概率选一个column
            col_num = utils.prob_dict_infer(column_num_prob, out_num=1)
            all_columns = utils.prob_dict_infer(schema_prob, out_num=col_num)      # 等概率选一个column

            print(f"parse_prob_status: all_columns = {all_columns}")

            if isinstance(all_columns, str):
                all_columns = [all_columns, ]
            for selected_column in all_columns:
                col_start, col_end = self.construct_single_condition(\
                    s, selected_column)
                schema_config[selected_column] = (col_start, col_end)
            config_dict[s] = copy(schema_config)

        return config_dict




# %%

class JoinPrioriGenerator(object):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, query_gen: QueryGenerator):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        self.query_gen = query_gen
        self.data_smooth = query_gen.data_smooth
        self.prob_dict_resize = query_gen.prob_dict_resize
        self.generate_batch_queries = query_gen.generate_batch_queries

    def position2score(self, position_list):
        """
        从位置到分数
        
        Args:
            position_list:
            arg2:
        Returns:
            score_dict:
            res2:
        """
        print("position_list = {}.".format(position_list))
        pos_counter = Counter(position_list)
        score_dict = {}
        for k, v in pos_counter.items():
            score_dict[k] = v        
        return score_dict
    
    def update_prob_table(self, table, column_list, position_list, smooth_factor):
        """
        {Description}
        
        Args:
            table: 所选的table
            column_list: 所选column的列表
            position_list: join order中位置的列表[]
            smooth_factor:
        Returns:
            old_prob_dict:
            new_prob_dict:
        """
        old_prob_dict = self.query_gen.table_prob_dict[table]
        new_prob_dict = copy(old_prob_dict)

        column_score_dict = defaultdict(list)    #
        position_score = self.position2score(position_list=position_list)
        print("position_score = {}.".format(position_score))

        for col, pos in zip(column_list, position_list):
            column_score_dict[col].append(position_score[pos])
        
        # 对new_prob_dict进行更新
        for k, v in column_score_dict.items():
            new_prob_dict[k] += np.max(v)

        # smooth operation
        # new_prob_dict = self.data_smooth(in_value_dict=new_prob_dict)

        # resize operation
        new_prob_dict = self.query_gen.prob_dict_resize(in_prob_dict=new_prob_dict)

        return old_prob_dict, new_prob_dict

    def update_prob_column(self, table, column, col_size_list, position_list, smooth_factor):
        """
        更新概率表

        Args:
            table: 表名
            column: 列名
            col_size_list: column_size的大小
            position_list: Table在join order中的位置
            smooth_factor: 平滑系数，针对临近结果的更新
        Returns:
            old_prob_dict:
            new_prob_dict:
        """
        old_prob_dict = self.query_gen.column_prob_dict[(table, column)]
        new_prob_dict = deepcopy(old_prob_dict)
        # 
        print("new_prob_dict = {}.".format(new_prob_dict))
        position_score = self.position2score(position_list=position_list)
        col_size_dict = defaultdict(list)

        for col, pos in zip(col_size_list, position_list):
            col_size_dict[col].append(position_score[pos])

        # 将col_size_dict的结果加到new_prob_dict中
        for k, v in col_size_dict.items():
            new_prob_dict[k] += np.max(v)

        # 针对概率进行临域的smooth
        new_prob_dict = self.data_smooth(new_prob_dict)
        # resize结果
        new_prob_dict = self.prob_dict_resize(new_prob_dict)

        return old_prob_dict, new_prob_dict



    def multi_step_workload_generation(self, external_func, step_num = 5):
        """
        {Description}
    
        Args:
            external_func: 外部评测结果的函数，输入是query的信息，输出是flag和level_dict，
                flag代表是否是zigzag的，level_dict代表每一个表对应的位置
            step_num: 
        Returns:
            query_list: 
            meta_list: 
            result_list: 
            table_prob_dict:
            column_prob_dict:
        """
        
        for step in range(step_num):
            print("current step = {}.".format(step))
            query_list, meta_list = self.generate_batch_queries(num = 10, with_meta=True)
            result_list = external_func(query_list, meta_list)

            # 默认的值都是两个列表
            column_info_dict = defaultdict(lambda: ([], []))
            table_info_dict = defaultdict(lambda: ([], []))

            # 完成info_dict的补全
            for query, meta, result in zip(query_list, meta_list, result_list):
                flag, level_dict = result
                print("level_dict = {}.".format(level_dict))

                if flag == False:
                    # bushy型的结果，不纳入考虑
                    continue

                # # 枚举所有的table
                # for schema in meta[0]:
                #     table_info_dict[schema][0].append()     # 添加Column
                #     table_info_dict[schema][1].append()     # 添加Position

                # 枚举所有的column
                for alias, column, _, _ in meta[1]:
                    position = level_dict[alias]
                    schema = self.query_gen.alias_inverse[alias]

                    # 更新table_info_dict
                    table_info_dict[schema][0].append(column)       # 添加Column
                    table_info_dict[schema][1].append(position)     # 添加Position

                    # 更新column_info_dict
                    # 确定start_idx, end_idx的相关信息
                    start_idx, end_idx = 0, 0
                    col_size = end_idx - start_idx

                    column_info_dict[(schema, column)][0].append(col_size)
                    column_info_dict[(schema, column)][1].append(position)

            # 进行数据的更新
            for k, v in table_info_dict.items():
                table = k
                column_list, position_list = v
                self.query_gen.update_prob_table(table=table, column_list=column_list, \
                                       position_list=position_list, smooth_factor=1.0)
                
            for k, v in column_info_dict.items():
                table, column = k
                col_size_list, position_list = v
                self.query_gen.update_prob_column(table=table, column=column, col_size_list=col_size_list, \
                                        position_list=position_list, smooth_factor=1.0)  

        return query_list, meta_list, result_list, self.query_gen.table_prob_dict, self.query_gen.column_prob_dict


# %%

def get_query_generator_by_workload(schema_list = [], workload = "stats", external_dict = {}):
    data_manager = external_dict.get("data_manager", data_management.DataManager(wkld_name=workload))
    bins_builder = external_dict.get("bins_builder", grid_preprocess.BinsBuilder(\
            workload=workload, data_manager_ref=data_manager))
    res_generator = QueryGenerator(schema_list = schema_list, 
            dm_ref=data_manager, bins_builder=bins_builder, workload=workload)
    return res_generator

# %%
class ProgressiveGenerator(object):
    """
    渐进式的查询生成器

    Members:
        field1:
        field2:
    """

    def __init__(self,):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        pass

    def init_query(self,):
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


    def add_predicate(self, table_name):
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


    def analyze_potential_tables(self,):
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

# %%
