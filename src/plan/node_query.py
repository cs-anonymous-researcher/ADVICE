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
import parse

# %%

from grid_manipulation import grid_preprocess, condition_exploration, grid_construction
from query import ce_injection, query_construction, query_exploration
from utility.workload_parser import SQLParser
from utility import utils
from workload import physical_plan_info                     # 物理计划的相关信息
# from plan import node_extension
# from plan.node_extension import ExtensionInstance
from data_interaction.mv_management import meta_comparison, meta_copy, meta_schema_append,\
    meta_schema_add, meta_filter_add, meta_filter_append
from data_interaction import data_management, mv_management
from grid_manipulation.grid_analysis import split_segment   # 
from utility import workload_spec
from utility.utils import tuple_in

from plan.node_utils import dict_complement, dict_difference, invalid_evaluation,\
    dict_make, plan_evaluation_under_cardinality, plan_comparison, complement_true_card, complement_estimation_card
from utility import common_config

# %%

class BatchExecutor(object):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, ce_handler:ce_injection.BaseHandler):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        # print(f"BatchExecutor.__init__: ce_handler = {ce_handler}")

        self.ce_handler = ce_handler
        self.instance_list = []

    def add_instance(self, subquery_dict, single_table_dict):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        res_idx = len(self.instance_list)
        self.instance_list.append((subquery_dict, single_table_dict))
        return res_idx

    def execute_all_instance_list(self,):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            instance_result:
            return2:
        """
        query_list = []
        pair_idx_list = []

        for instance in self.instance_list:
            subquery_dict, single_table_dict = instance
            subquery_idx, single_table_idx = {}, {}
            curr_pos = len(query_list)

            # 处理subquery
            for idx, (k, v) in enumerate(subquery_dict.items()):
                subquery_idx[k] = idx + curr_pos
                query_list.append(v)
            
            # 处理single_table
            curr_pos = len(query_list)
            for idx, (k, v) in enumerate(single_table_dict.items()):
                single_table_idx[k] = idx + curr_pos
                query_list.append(v)

            pair_idx_list.append((subquery_idx, single_table_idx))

        card_result_list = self.ce_handler.get_cardinalities(query_list=query_list)

        instance_result = []

        for pair_idx in pair_idx_list:
            subquery_idx, single_table_idx = pair_idx
            subquery_res, single_table_res = {}, {}

            for k, idx in subquery_idx.items():
                subquery_res[k] = card_result_list[idx]

            for k, idx in single_table_idx.items():
                single_table_res[k] = card_result_list[idx]

            instance_result.append((subquery_res, single_table_res))

        # print(f"execute_all_instance_list: len(instance_result) = {len(instance_result)}")
        return instance_result
    
    def clean_cache(self,):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        self.instance_list = []

# %%

class QueryInstance(object):
    """
    单个查询的实例

    Members:
        field1:
        field2:
    """

    def __init__(self, workload, query_meta, data_manager_ref: data_management.DataManager, ce_handler:ce_injection.BaseHandler, \
        query_controller: query_exploration.QueryController, grid_preprocessor: grid_preprocess.MultiTableBuilder, \
        bins_builder: grid_preprocess.BinsBuilder, card_dict: dict = {}):
        """
        {Description}

        Args:
            workload:
            query_meta:
            data_manager_ref:
            ce_handler:
            query_controller: 
            grid_preprocessor: 
        """
        self.workload = workload
        # self.top_key = sorted([self.data_manager.tbl_abbr[i] for i in query_meta[0]])

        # 查询的文本信息和元信息
        self.query_text = query_construction.construct_origin_query(\
            query_meta=query_meta, workload=workload)

        # print("QueryInstance: self.query_text = {}.".format(self.query_text))
        # self.query_meta = query_meta

        self.estimation_card_dict = {}      # 估计子查询基数的字典
        self.true_card_dict = {}            # 真实子查询基数的字典

        self.estimation_single_table = {}
        self.true_single_table = {}

        self.data_manager = data_manager_ref
        self.ce_handler = ce_handler
        self.query_ctrl = query_controller

        # grid manipulation相关
        self.grid_preprocessor = grid_preprocessor

        # 
        self.bins_builder = bins_builder
        
        self.set_query_meta(new_meta=query_meta)

        # 全局bins_dict的信息
        self.global_bins_info = {}
        # 全局value_cnt_array的信息
        self.global_value_cnt_info = {}

        self.init_card_estimation()
        # 验证初始化的正确性
        # print("__init__: subquery_dict = {}. single_table_dict = {}.".\
        #     format(self.estimation_card_dict, self.estimation_single_table))
        # 当前物理计划的获取
        self.physical_plan_fetch()
        # print(self.physical_info.get_physical_spec())   # 打印物理信息
        # print("abbr_inverse = {}.".format(data_manager_ref.tbl_abbr))

    def true_card_integrity(self,) -> bool:
        """
        {Description}
        
        Args:
            None
        Returns:
            flag:
            missing_key_list:
        """
        missing_key_list = []

        # subquery_evaluation
        for k, v in self.estimation_card_dict.items():
            if k not in self.true_single_table or self.true_single_table[k] \
                is None or self.true_single_table[k] < 0:
                missing_key_list.append(k)
                # break
        
        # single_table_evaluation
        for k, v in self.estimation_single_table.items():
            if k not in self.true_single_table or self.true_single_table[k] \
                is None or self.true_single_table[k] < 0:
                missing_key_list.append(k)
                # break
        
        flag = len(missing_key_list) == 0
        return flag, missing_key_list
    
    def set_query_meta(self, new_meta):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        # 采用深复制的策略
        query_meta = (tuple(deepcopy(new_meta[0])), tuple(deepcopy(new_meta[1])))
        # meta信息规范化
        self.query_meta = mv_management.meta_standardization(query_meta)
        # 设置新的top_key
        self.top_key = sorted([self.data_manager.tbl_abbr[i] for i in query_meta[0]])

    def __str__(self,) -> str:
        """
        返回实例的字符串表达式
        
        Args:
            None
        Returns:
            res_str:
        """
        res_str = "QueryInstance.query_meta = {}".format(self.query_meta)
        return res_str


    def get_columns_from_schema(self, schema):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            column_list:
            res2:
        """
        alias = self.get_alias(table_name=schema)
        column_list = []

        for item in self.query_meta[1]:
            if item[0] == alias:
                column_list.append(item[1])
        return column_list

    def get_single_table_card(self, schema):
        """
        获得单表的相关基数
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        return self.estimation_single_table[self.get_alias(schema)]


    def get_columns_info(self, schema, column_list):
        """
        获得当前列的相关信息
    
        Args:
            schema:
            column_list:
        Returns:
            value_list: 当前的取值列表
            min_value: 最小的grid值
            max_value: 最大的grid值
        """
        value_list = []
        min_value, max_value = 100000.0, 0.0
        column_composite = []
        for col in column_list:
            column_composite.append((schema, col))

        # TODO: 修改hard code的代码
        bins_dict, reverse_dict = self.construct_bins_dict(column_list=column_composite, split_budget=100)
        marginal_dict = self.bins_builder.construct_marginal_dict(bins_dict=bins_dict)

        alias = self.get_alias(table_name=schema)
        for column in column_list:
            for item in self.query_meta[1]:
                if (alias, column) == (item[0], item[1]):
                    start_val, end_val = item[2], item[3]
                    # bins_local = bins_dict[(schema, column)]
                    reverse_local = reverse_dict[(schema, column)]
                    start_idx, end_idx = utils.predicate_location(reverse_local, start_val, end_val)
                    marginal_local = marginal_dict[(schema, column)]
                    value_list.append(np.sum(marginal_local[start_idx: end_idx]))
                    # 完成min_value和max_value的更新
                    min_value = np.min((min_value, np.min(marginal_local)))
                    max_value = np.max((max_value, np.sum(marginal_local)))

        return value_list, min_value, max_value

    def infer_clip_spec(self, range_dict_list: list):
        """
        推断需要进行clip的column已经具体的处理范围
    
        Args:
            range_dict_list:
            arg2:
        Returns:
            clip_column: 选择裁剪的列
            clip_point: 选择裁剪的起始点
        """
        coverage_dict = {}

        clip_column = ""
        clip_point = 0, 0

        # 确保curr_key的一致性
        global_key = ""
        for range_dict in range_dict_list:
            column_list = list(range_dict.keys())
            print("range_dict = {}. column_list = {}.".format(\
                range_dict, column_list))
            
            curr_key = self.column_list2key(column_list)
            bins_dict, reverse_dict = self.global_bins_info[curr_key]
            if global_key == "":
                global_key = curr_key
            elif global_key != curr_key:
                raise ValueError("global_key = {}. curr_key = {}.".format(global_key, curr_key))

        # 构造coverage_dict
        for k, v in bins_dict.items():
            coverage_dict[k] = np.zeros(len(v))

        for range_dict in range_dict_list:
            for col_name, (start_val, end_val) in range_dict.items():
                reverse_local = reverse_dict[col_name]
                start_idx, end_idx = reverse_local[start_val], reverse_local[end_val]
                coverage_dict[col_name][start_idx:end_idx] += 1

        # 构造ratio_dict，表示整个column有多少被覆盖了
        ratio_dict = {}
        
        for k, v in coverage_dict.items():
            ratio_dict[k] = (v > 1e-5).sum() / len(v)

        # 选择column
        selected_column, cover_val = "", 1.0
        for k, v in ratio_dict.items():
            if v < cover_val:
                selected_column = k
                cover_val = v
        
        # 从column中确定一个point，实际上感觉设置成
        # 函数好一点，后续待优化，目前采用greedy的策略，直接去选count最大的点
        selected_idx, max_val = 0, 0
        for idx, v in enumerate(coverage_dict[selected_column]):
            if v > max_val:
                selected_idx = idx
                max_val = v

        clip_column, clip_point = selected_column, selected_idx
        return clip_column, clip_point


    def construct_range_dict_list(self, query_meta_list, column_list):
        """
        {Description}
        
        Args:
            query_meta_list:
            column_list:
        Returns:
            range_dict_list: 范围字典的列表
        """
        range_dict_list = []
        alias_reverse = self.data_manager.abbr_inverse

        # print("construct_range_dict_list: query_meta_list = {}. column_list = {}.".\
        #       format(query_meta_list, column_list))
        
        def construct_single_range_dict(query_meta, column_list):
            schema_list, filter_list = query_meta
            range_dict = {}
            # print("construct_single_range_dict: column_list = {}.".format(column_list))
            for item in filter_list:
                curr_col = alias_reverse[item[0]], item[1]
                if curr_col in column_list:
                    # 填充range_dict
                    range_dict[curr_col] = (item[2], item[3])
            
            return range_dict

        for query_meta in query_meta_list:
            local_dict = construct_single_range_dict(\
                query_meta=query_meta, column_list=column_list)
            range_dict_list.append(local_dict)

        return range_dict_list

    def get_extra_schema_meta(self, query_meta):
        """
        获得额外的schema组成的meta信息，只考虑schema，不考虑filter
        
        Args:
            query_meta:
        Returns:
            extra_meta:
        """
        schema_origin = self.query_meta[0]
        schema_new = query_meta[0]
        schema_res = []
        for s in schema_new:
            if s not in schema_origin:
                schema_res.append(s)
        return schema_res, []

    def create_grid_on_meta(self, query_meta_list):
        """
        构造新的grid真实矩阵
        
        Args:
            query_meta_list: 一批查询的meta信息
        Returns:
            value_cnt_arr:
            column_order:
        """
        column_list = self.get_column_list_from_meta(query_meta=query_meta_list[0])
        # 获得bins_dict
        bins_dict, reverse_dict = self.get_global_bins_info(column_list=column_list)
        # 生成range_dict_list = []
        range_dict_list = self.construct_range_dict_list(query_meta_list=\
                query_meta_list, column_list=column_list)

        clip_column, clip_point = self.infer_clip_spec(range_dict_list)
        clip_bins = bins_dict[clip_column]

        new_table_meta = self.get_extra_schema_meta(query_meta=query_meta_list[0])
        # 打印建立joined_tables_obj的相关信息
        print("src_meta_list = {}.".format([self.query_meta, new_table_meta]))
        print("selected_columns = {}.".format(column_list))
        print("clip_column = {}.".format(clip_column))
        print("clip_point = {}.".format(clip_point))
        
        # 获得连接表的对象
        joined_tables_obj, column_order = self.grid_preprocessor.build_joined_tables_object(\
            src_meta_list = [self.query_meta, new_table_meta], selected_columns = column_list, \
            clip_column = clip_column, clip_bins = clip_bins, clip_point = clip_point)

        input_bins = []
        for col in column_order:
            input_bins.append(bins_dict[col])
        # 构建grid_array
        distinct_list, marginal_list, value_cnt_arr = grid_construction.\
            make_grid_data(data_arr=joined_tables_obj, input_bins=input_bins)

        return value_cnt_arr, column_order


    def get_column_list_from_meta(self, query_meta):
        """
        {Description}
    
        Args:
            query_meta:
            arg2:
        Returns:
            column_list:
            return2:
        """
        column_list = []
        _, filter_list = query_meta
        _, filter_src = self.query_meta

        alias_reverse = self.data_manager.abbr_inverse  # 别名反向映射
        print("alias_reveres = {}.".format(alias_reverse))

        for item in filter_list:
            if item not in filter_src:
                print("get_column_list_from_meta: item = {}.".format(item))
                alias, column, _, _ = item
                column_list.append((alias_reverse[alias], column))

        return column_list


    def infer_true_cardinality(self, query_meta):
        """
        通过query元信息推导真实的基数
        
        Args:
            query_meta:
            arg2:
        Returns:
            res1:
            res2:
        """
        column_list = self.get_column_list_from_meta(query_meta=query_meta)
        # 找到对应的bins_dict和reverse_dict
        bins_dict, reverse_dict = self.get_global_bins_info(column_list=column_list)
        # 
        restriction, column_order, value_cnt_arr = self.get_value_cnt_info(column_list=column_list)

        # 局部的索引
        local_index = []
        for col in column_order:
            lower_bound_idx, upper_bound_idx = self.bound_location(\
                query_meta=query_meta, column=col, reverse_dict=reverse_dict)
            if lower_bound_idx == -1 and upper_bound_idx == -1:
                local_index.append(Ellipsis)
            else:
                local_index.append(slice(lower_bound_idx, upper_bound_idx))
        # 真实基数的求和
        true_card = np.sum(value_cnt_arr[tuple(local_index)])
        
        return true_card


    def modify_by_query_meta(self, new_query_meta, time_limit=60000):
        """
        管理扩展实例
    
        Args:
            new_query_meta: 新的query_meta，包含整一个查询的元数据
        Returns:
            cost1: 最有的cost
            cost2: 实际的cost
        """
        print("modify_by_query_meta: old_query_meta = {}. new_query_meta = {}.".format(self.query_meta, new_query_meta))
        candidate_tables = []
        # 分析修改的table
        schema_new, filter_new = new_query_meta
        schema_old, filter_old = self.query_meta

        for s in schema_new:
            if s not in schema_old:
                # 有新的表被添加了
                candidate_tables.append(s)
            else:
                alias = self.get_alias(table_name=s)
                for item in filter_new:
                    if s in candidate_tables:
                        break

                    if item[0] == alias:
                        # 探索condition不匹配的情况
                        flag = False
                        for item1 in filter_old:
                            if item == item1:
                                flag = True
                        
                        if flag == False:
                            candidate_tables.append(s)
                            break

        candidate_alias = []

        for s in candidate_tables:
            candidate_alias.append(self.data_manager.tbl_abbr[s])

        # 根据table删除真实基数和估计基数的相关选项
  
        delete_key_list = []
        est_card = self.estimation_card_dict
        before_keys = est_card.keys()
        # 删除基数估计的结果
        for k in list(self.estimation_card_dict.keys()):
            if len(set(k).intersection(candidate_alias)) > 0:
                del self.estimation_card_dict[k]
                delete_key_list.append(k)

        for k in list(self.estimation_single_table.keys()):
            if k in candidate_alias:
                del self.estimation_single_table[k]
                delete_key_list.append(k)

        after_keys = est_card.keys()
        print(f"modify_by_query_meta: candidate_tables = {candidate_tables}. "
              f"before_keys = {before_keys}. after_keys = {after_keys}.")

        # 删除真实基数的结果
        for k in list(self.true_card_dict.keys()):
            if len(set(k).intersection(candidate_alias)) > 0:
                del self.true_card_dict[k]
        
        for k in list(self.true_single_table.keys()):
            if k in candidate_alias:
                del self.true_single_table[k]

        # 完成基数的补充，设置新的meta
        # self.query_meta = new_query_meta
        self.set_query_meta(new_meta=new_query_meta)

        self.query_text = query_construction.construct_origin_query(\
            query_meta=new_query_meta, workload=self.workload)

        # 补全估计基数
        self.complement_estimation_card()       # 新的方法优化效率

        query_parser = SQLParser(sql_text = self.query_text, workload=self.workload) 

        # 补全真实基数
        self.true_card_dict, self.true_single_table = complement_true_card(query_parser=query_parser, \
            subquery_true=self.true_card_dict, single_table_true=self.true_single_table, \
            subquery_estimation=self.estimation_card_dict, single_table_estimation=self.estimation_single_table, \
            get_cardinalities_func=self.query_ctrl.db_conn.get_cardinalities, time_limit=time_limit)
        
        # 判断所有基数是否都被获取
        flag = invalid_evaluation(self.true_card_dict)
        if flag == False:
            # 提前return结果
            cost1, cost2 = -1, -1
            return cost1, cost2
        
        assert len(self.true_card_dict) == len(self.estimation_card_dict)           # 基数数目确保相同
        assert len(self.true_single_table) == len(self.estimation_single_table)     # 
        
        # 获得新的plan相关信息
        plan1 = plan_evaluation_under_cardinality(workload=self.workload, query_ctrl=self.query_ctrl, \
            query_meta=self.query_meta, subquery_dict=self.true_card_dict, single_table_dict=self.true_single_table)
        plan2 = plan_evaluation_under_cardinality(workload=self.workload, query_ctrl=self.query_ctrl, \
            query_meta=self.query_meta, subquery_dict=self.estimation_card_dict, single_table_dict=self.estimation_single_table)

        # 设置外部数据库连接
        plan1.set_database_connector(self.query_ctrl.db_conn)
        plan2.set_database_connector(self.query_ctrl.db_conn)

        # 返回结果
        flag, cost1, cost2 = plan_comparison(plan1, plan2, subquery_dict=self.true_card_dict, single_table_dict=self.true_single_table)
        print("modify_by_query_meta: normal return. flag = {}. cost1 = {}. cost2 = {}.".format(flag, cost1, cost2))

        return cost1, cost2


    def complement_true_card(self, time_limit):
        """
        补全真实基数
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        query_parser = SQLParser(sql_text = self.query_text, workload=self.workload)

        # print(f"complement_true_card: subquery_true = {self.true_card_dict.keys()}")
        # print(f"complement_true_card: single_table_true = {self.true_single_table.keys()}")

        # print(f"complement_true_card: subquery_estimation = {self.estimation_card_dict.keys()}")
        # print(f"complement_true_card: single_table_estimation = {self.estimation_single_table.keys()}")

        subquery_true, single_table_true = complement_true_card(query_parser=query_parser, \
            subquery_estimation=self.estimation_card_dict, single_table_estimation=self.estimation_single_table, \
            subquery_true=self.true_card_dict, single_table_true=self.true_single_table, \
            get_cardinalities_func=self.query_ctrl.db_conn.get_cardinalities, time_limit=time_limit)

        self.true_card_dict = subquery_true
        self.true_single_table = single_table_true

        return self.true_card_dict, self.true_single_table
    
    def complement_estimation_card(self,):
        """
        补全估计的cardinality
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        self.query_ctrl.set_query_instance(query_text = \
            self.query_text, query_meta = self.query_meta)
        # subquery_repr_list = self.query_ctrl.get_all_sub_relations()  # 获得所有的子查询关系
        subquery_repr_list = self.data_manager.get_subqueries(alias_list=self.top_key)

        tbl_abbr = query_construction.abbr_option[self.workload]
        single_table_repr_list = [tbl_abbr[schema] for schema in self.query_meta[0]]

        subquery_ref = {}
        single_table_ref = {}

        for k in subquery_repr_list:
            subquery_ref[k] = 0

        for k in single_table_repr_list:
            single_table_ref[k] = 0

        query_parser = SQLParser(sql_text = self.query_text, workload=self.workload)
        subquery_estimation, single_table_estimation = complement_estimation_card(query_parser=query_parser, \
            subquery_estimation=self.estimation_card_dict, single_table_estimation=self.estimation_single_table, \
            subquery_ref=subquery_ref, single_table_ref=single_table_ref, get_cardinalities_func=self.ce_handler.get_cardinalities)

        # self.subquery_estimation = subquery_estimation
        # self.single_table_estimation = single_table_estimation
        self.estimation_card_dict = subquery_estimation
        self.estimation_single_table = single_table_estimation

        return subquery_estimation, single_table_estimation
    
    def init_card_estimation(self,):
        """
        当前查询历史基数估计初始化
        合并两次基数估计调用
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        # 多表子查询的结果
        self.query_ctrl.set_query_instance(query_text = \
            self.query_text, query_meta = self.query_meta)
        curr_parser = SQLParser(sql_text = self.query_text, workload=self.workload)

        workload_foreign_mapping = workload_spec.get_spec_foreign_mapping(workload_name = self.workload)
        curr_parser.load_workload_info(workload_foreign_mapping)     # 加载workload的信息

        subquery_repr_list = self.data_manager.get_subqueries(self.top_key)

        subquery_text_list = [curr_parser.construct_PK_FK_sub_query(alias_list = alias_list) \
            for alias_list in subquery_repr_list]

        # 单表的结果
        single_table_repr_list = [self.data_manager.tbl_abbr[schema] for schema in self.query_meta[0]]
        single_table_text_list = [curr_parser.get_single_table_query(alias) for alias \
            in single_table_repr_list]
        single_table_local = {}     # 单表的结果
        subquery_local = {}         # 子查询的结果

        # single_table_res_list = self.ce_handler.get_cardinalities(query_list = single_table_text_list)
        # subquery_res_list = self.ce_handler.get_cardinalities(query_list = subquery_text_list)
        total_res_list = self.ce_handler.get_cardinalities(\
            query_list = single_table_text_list + subquery_text_list)
        
        seg_pos = len(single_table_text_list)
        single_table_res_list, subquery_res_list = total_res_list[:seg_pos], total_res_list[seg_pos:]

        for k, v in zip(subquery_repr_list, subquery_res_list):
            subquery_local[k] = v
        self.dict_complement(old_dict = self.estimation_card_dict, new_dict = subquery_local)

        for k, v in zip(single_table_repr_list, single_table_res_list):
            single_table_local[k] = v
        self.dict_complement(old_dict = self.estimation_single_table, new_dict = single_table_local)

    def dict_complement(self, old_dict, new_dict):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        return dict_complement(old_dict, new_dict)

    def dict_difference(self, old_dict, new_dict):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        return dict_difference(old_dict, new_dict)

    def dict_make(self, key_list, value_list):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        return dict_make(key_list, value_list)

    # def add_estimation_card(self, card_dict, mode = "subquery"):
    def add_estimation_card(self, card_dict, mode):
        """
        添加估计的基数，支持的mode为["subquery", "single_table"]
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        if mode == "subquery":
            return self.dict_complement(old_dict = self.estimation_card_dict, new_dict = card_dict)
        elif mode == "single_table":
            return self.dict_complement(old_dict = self.estimation_single_table, new_dict = card_dict)

    def card_key_eval(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            error_list:
            return2:
        """
        pass

    # def add_true_card(self, card_dict, mode = "subquery"):
    def add_true_card(self, card_dict, mode):
        """
        添加真实的基数，支持的mode为["subquery", "single_table"]
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        if mode == "subquery":
            self.true_card_dict = self.dict_complement(old_dict = self.true_card_dict, new_dict = card_dict)
            return self.true_card_dict
        elif mode == "single_table":
            self.true_single_table = self.dict_complement(old_dict = self.true_single_table, new_dict = card_dict)
            return self.true_single_table
        else:
            raise ValueError("add_true_card: Unsupported mode({})".format(mode))

    # @utils.timing_decorator
    def fetch_candidate_tables(self, table_subset = None):
        """
        获得候选新增的表
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        table_set = set()
        workload_foreign_mapping = workload_spec.\
            get_spec_foreign_mapping(workload_name = self.workload)
        
        # print("workload_foreign_mapping = {}.".format(workload_foreign_mapping))
        # 这里是所有的表，但是实际上应该是一个子集
        if table_subset is None or len(table_subset) == 0:
            all_tables = workload_foreign_mapping.keys()
        else:
            all_tables = table_subset

        for table_i in all_tables:
            for table_j in all_tables:
                if table_i == table_j:
                    continue
                # print("table_i = {}. table_j = {}.".format(table_i, table_j))
                # 主键表没有出现怎么办？不一定是外键表的情况
                if table_i in workload_foreign_mapping.keys() and table_j in workload_foreign_mapping.keys():
                    _, ref_tbl_i, ref_col_i = workload_foreign_mapping[table_i]
                    _, ref_tbl_j, ref_col_j = workload_foreign_mapping[table_j]

                    # print("fetch_candidate_tables case 1: table_i = {}. ref_tbl_i = {}. table_j = {}. ref_tbl_i = {}.".\
                    #       format(table_i, ref_tbl_i, table_j, ref_tbl_j))
                    
                    if (ref_tbl_i, ref_col_i) == (ref_tbl_j, ref_col_j):
                        if table_i in self.query_meta[0] and table_j not in self.query_meta[0]:
                            table_set.add(table_j)

                    if ref_tbl_i == table_j:
                        if table_j in self.query_meta[0] and table_i not in self.query_meta[0]:
                            table_set.add(table_i)
                        if table_i in self.query_meta[0] and table_j not in self.query_meta[0]:
                            table_set.add(table_j)

                elif table_i in workload_foreign_mapping.keys():
                    _, ref_tbl_i, ref_col_i = workload_foreign_mapping[table_i]
                    # print("fetch_candidate_tables case 2: table_i = {}. ref_tbl_i = {}. table_j = {}.".\
                    #       format(table_i, ref_tbl_i, table_j))
                    if ref_tbl_i == table_j:
                        if table_i in self.query_meta[0] and table_j not in self.query_meta[0]:
                            table_set.add(table_j)
                        if table_j in self.query_meta[0] and table_i not in self.query_meta[0]:
                            table_set.add(table_i)
                # elif table_j in workload_foreign_mapping.keys():
                #     _, ref_tbl_j, ref_col_j = workload_foreign_mapping[table_j]
                #     if ref_tbl_j == table_i:
                #         if table_i in self.query_meta[0] and table_j not in self.query_meta[0]:
                #             table_set.add(table_j)
                            

        # 最后把集合转成列表
        return list(table_set)  

    def bound_location(self, query_meta, column, reverse_dict):
        """
        确定索引的上界和下界
        
        Args:
            query_meta:
            column:
            reverse_dict:
        Returns:
            lower_bound_idx:
            upper_bound_idx:
        """
        lower_bound_idx, upper_bound_idx = 0, 0
        reverse_local = reverse_dict[column]
        filter_list = query_meta[1]
        flag = False

        for item in filter_list:
            # 需要处理None的问题
            alias, col, lower_bound, upper_bound = item

            if alias == column[0] and col == column[1]:
                if lower_bound is not None:
                    lower_bound_idx = reverse_local[lower_bound]
                else:
                    lower_bound_idx = 0

                if upper_bound is not None:
                    upper_bound_idx = reverse_local[upper_bound]
                else:
                    upper_bound_idx = max(reverse_local.values())   # 选择为最大值
                flag = True     # 完成匹配，直接退出
                break

        if flag == False:
            lower_bound_idx, upper_bound_idx = -1, -1
        return lower_bound_idx, upper_bound_idx


    def single_subquery_tuning(self, query_meta, target_key, subquery_dict, \
                               single_table_dict, growth_factor = 2, mode = "under"):
        """
        针对单个subquery的调优
        
        Args:
            arg1:
            arg2:
        Returns:
            origin_cost:
            origin_card:
            final_cost:
            final_card:
        """
        origin_plan = self.plan_evaluation_under_cardinality(query_meta=query_meta,
            subquery_dict=subquery_dict, single_table_dict=single_table_dict)
        target_key = tuple(target_key)  # 转成元组
        print("target_key = {}. subquery_dict = {}.".format(target_key, subquery_dict))
        print("origin join_order: {}.".format(origin_plan.leading))

        origin_card = subquery_dict[target_key]
        final_card = int(max(origin_card, 1))
        origin_cost = origin_plan.plan_cost    # 最初的代价
        final_cost = origin_cost

        max_try_times = 10
        if mode == "under":
            pass
        elif mode == "over":
            pass

        for _ in range(max_try_times):
            final_card *= growth_factor
            print("final_card = {}.".format(final_card))
            subquery_dict[target_key] = final_card
            local_plan = self.plan_evaluation_under_cardinality(query_meta=query_meta,
                subquery_dict=subquery_dict, single_table_dict=single_table_dict)
            final_cost = local_plan.plan_cost

            flag, error_dict = physical_plan_info.physical_comparison(\
                (origin_plan.leading, origin_plan.join_ops, origin_plan.scan_ops), 
                (local_plan.leading, local_plan.join_ops, local_plan.scan_ops))
            # 如果出现了查询计划不等，直接退出
            if flag == False:
                print("Origin Plan:")
                origin_plan.show_plan()
                print("Current Plan:")
                local_plan.show_plan()
                break

        return origin_cost, origin_card, final_cost, final_card
        


    def extend_estimation(self, query_meta, new_table, subquery_dict, single_table_dict):
        """
        针对table拓展的状态估计，判断
    
        Args:
            query_meta: 
            new_table: 
            subquery_dict: 
            single_table_dict:
        Returns:
            flag: 两个物理计划是否等价
            plan_target: 目标的物理计划
            plan_actual: 实际的物理计划
            indicator_pair: 指示的值(value1, value2)，
                如果两个查询计划相同，其代表当前的cost以及可以接受的最大顶层card对应的cost。
                如果两个查询计划不同，其代表估计基数下两个Plan的Cost。
        """
        plan_actual = self.plan_evaluation_under_cardinality(query_meta=query_meta,
            subquery_dict=subquery_dict, single_table_dict=single_table_dict)
        leading_hint, scan_ops, join_ops = self.target_plan_info(new_table = new_table)

        plan_target = self.plan_evaluation_under_spec(query_meta = query_meta, \
            subquery_dict = subquery_dict, single_table_dict = single_table_dict, \
            leading_hint = leading_hint, scan_ops = scan_ops, join_ops = join_ops)

        flag_join = plan_actual.leading == plan_target.leading   # 仅比较join order的顺序

        # Join物理算子的比较
        flag_plan, error_dict = physical_plan_info.physical_comparison(\
            (plan_actual.leading, plan_actual.join_ops, plan_actual.scan_ops), 
            (plan_target.leading, plan_target.join_ops, plan_target.scan_ops))
        
        print("extend_estimation: new_table = {}. origin_meta = {}. below are two query plans(actual plan and target plan):".\
              format(new_table, self.query_meta[0]))
        plan_actual.show_plan()
        plan_target.show_plan()

        indicator_pair = None, None
        if flag_plan == True:
            # 
            target_key = self.top_key     # 目标子查询的key
            origin_cost, origin_card, final_cost, final_card = self.single_subquery_tuning(\
                query_meta=query_meta, target_key=target_key, subquery_dict=subquery_dict, single_table_dict=single_table_dict)
            # return final_cost - origin_cost
            print("flag = True. origin_cost = {}. final_cost = {}.".format(origin_cost, final_cost))
            indicator_pair = origin_cost, final_cost
        elif flag_plan == False:
            # 设置数据库连接
            plan_actual.set_database_connector(self.query_ctrl.db_conn)
            plan_target.set_database_connector(self.query_ctrl.db_conn)
            target_cost = plan_target.get_plan_cost(subquery_dict=subquery_dict, single_table_dict=single_table_dict)
            actual_cost = plan_actual.get_plan_cost(subquery_dict=subquery_dict, single_table_dict=single_table_dict)
            print("flag = False. target_cost = {}. actual_cost = {}.".format(target_cost, actual_cost))
            indicator_pair = actual_cost, target_cost

        return flag_plan, plan_target, plan_actual, indicator_pair
    

    @utils.timing_decorator
    def plan_verification(self, query_meta, new_table, subquery_dict, single_table_dict):
        """
        给定查询的元信息以及子查询、单表的基数信息，分析目标计划是否符合预期
        
        Args:
            query_meta:
            new_table:
            subquery_dict:
            single_table_dict:
        Returns:
            flag: 两个物理计划是否等价
            plan_target: 目标的物理计划
            plan_actual: 实际的物理计划
        """
        plan_actual = self.plan_evaluation_under_cardinality(query_meta=query_meta,
            subquery_dict=subquery_dict, single_table_dict=single_table_dict)
        leading_hint, scan_ops, join_ops = self.target_plan_info(new_table = new_table)
        plan_target = self.plan_evaluation_under_spec(query_meta = query_meta, \
            subquery_dict = subquery_dict, single_table_dict = single_table_dict, \
            leading_hint = leading_hint, scan_ops = scan_ops, join_ops = join_ops)

        mode = common_config.comparison_mode
        if mode == "leading":
            flag = plan_actual.leading == plan_target.leading   # 仅比较join order的顺序
        elif mode == "plan":
            # Join物理算子的比较
            flag, error_dict = physical_plan_info.physical_comparison(\
                (plan_actual.leading, plan_actual.join_ops, plan_actual.scan_ops), 
                (plan_target.leading, plan_target.join_ops, plan_actual.scan_ops))
        else:
            raise ValueError(f"plan_verification: mode = {mode}.")
        
        # if flag == False:
        #     print(f"plan_verification: new_table = {new_table}. plan_actual.leading = {plan_actual.leading}. plan_target.leading = {plan_target.leading}")

        return flag, plan_target, plan_actual

    def random_range_on_bins(self, bins, length = None):
        """
        在bins上进行随机的采样
        
        Args:
            bins:
            length:
        Returns:
            start_val:
            end_val:
        """
        total = len(bins)
        # 在一个bins上随机选择一段长度
        if length is not None:
            start, end = 0, 0
            start = np.random.randint(0, total - length)    # 随机选择起始位置
            end = start + length
            return bins[start] + 1, bins[end]
        else:
            start, end = np.random.randint(0, total), np.random.randint(0, total)
            start, end = min(start, end), max(start, end)
            return bins[start] + 1, bins[end]

    @utils.timing_decorator
    def random_on_columns_batch(self, bins_dict: dict, num):
        """
        {Description}
        
        Args:
            bins_dict:
            num:
        Returns:
            result_batch:
        """
        # 根据bins_dict生成一批随机的值
        result_batch = []
        for _ in range(num):
            result_range_dict = {}
            for k, v in bins_dict.items():
                result_range_dict[k] = self.random_range_on_bins(v)
            result_batch.append(result_range_dict)
        return result_batch

    def selection_on_bins(self, bins_dict, data_ratio):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        # 在bins上进行选择一个multi-range
        total_size = 1
        total_dim = 0
        for k, v in bins_dict.items():
            total_size *= (len(v) - 1)
            total_dim += 1
        segments = split_segment(np.log(total_size * data_ratio), n = total_dim)    # 随机划分成若干个区块
        bins_length_list = [int(math.exp(s)) for s in segments]
        idx = 0
        result_range_dict = {}
        for k, v in bins_dict.items():
            local_length = bins_length_list[idx]
            idx += 1
            result_range_dict[k] = self.random_range_on_bins(v, local_length)
        return result_range_dict


    def add_new_table_meta(self, curr_query_meta, new_table, column_range_dict):
        """
        向query_meta中添加新表的信息，返回新的元信息
        
        Args:
            curr_query_meta:
            new_table:
            column_range_dict:
        Returns:
            res_query_meta:
        """
        # 拷贝已有的元信息
        res_query_meta = meta_copy(in_meta = curr_query_meta)
        new_filter_list = []
        alias_name = self.data_manager.tbl_abbr[new_table]     # 获得表的别名
        for k, v in column_range_dict.items():
            new_filter_list.append((alias_name, k[1], v[0], v[1]))

        # 添加schema
        res_query_meta = meta_schema_add(res_query_meta, new_table)
        # 添加filter
        res_query_meta = meta_filter_append(res_query_meta, new_filter_list)
        return res_query_meta


    def extend_new_table(self, table_name, column_info):
        """
        添加一个新的表，观察结果

        Args:
            table_name:
            column_info:
        Returns:
            curr_query_meta: 当前查询的原信息
            return2:
        """
        curr_query_meta = meta_copy(self.query_meta)
        column_num = column_info["num"]         # 选择列的数目
        data_ratio = column_info["ratio"]       # 选择表数据的百分比
        split_budget = column_info["budget"]    # 划分的大小
        column_list = self.predicate_selection(table_name = table_name, column_num = column_num)

        bins_dict, reverse_dict = self.construct_bins_dict(column_list = column_list, split_budget = split_budget)
        result_range_dict = self.selection_on_bins(bins_dict, data_ratio)   

        def add_new_meta(curr_query_meta, result_range_dict):
            # 
            new_filter_list = []
            alias_name = self.data_manager.tbl_abbr[table_name]     # 获得表的别名
            for k, v in result_range_dict.items():
                new_filter_list.append((alias_name, k[1], v[0], v[1]))

            return meta_filter_append(curr_query_meta, new_filter_list)

        # print("before meta = {}.".format(curr_query_meta))
        # 添加新的schema
        curr_query_meta = meta_schema_add(curr_query_meta, schema = table_name)
        # 添加新的filter
        curr_query_meta = add_new_meta(curr_query_meta, result_range_dict)
        # print("after meta = {}.".format(curr_query_meta))
        return curr_query_meta
    
    @utils.timing_decorator
    def predicate_selection(self, table_name, column_num, mode = "greedy"):
        """
        从table中选取特定的谓词特征
        
        Args:
            table_name:
            column_num:
        Returns:
            column_list:
        """
        # 
        assert mode in ("greedy", "random")
        column_list = []
        # TableExplorer初始化可能会比较的慢
        table_explorer = condition_exploration.TableExplorer(dm_ref = \
            self.data_manager, workload = self.workload, table_name = table_name)

        if mode == "greedy":
            if column_num > len(table_explorer.column_order):
                column_list = table_explorer.column_order
            else:
                column_list = table_explorer.column_order[:column_num]
        elif mode == "random":
            column_list = np.random.choice(table_explorer.column_order, \
                                           column_num, replace=False)

        # 补充table_name
        column_list = [(table_name, column_name) for column_name in column_list]
        return column_list

    @utils.timing_decorator
    def get_query_cardinalities(self, query_meta):
        """
        获得指定查询的基数值
        
        Args:
            query_meta:
            arg2:
        Returns:
            subquery_dict:
            single_table_dict:
        """
        query_text = query_construction.construct_origin_query(\
            query_meta, workload = self.workload)
        current_parser = SQLParser(sql_text = query_text, workload=self.workload)

        current_parser.load_external_alias_info(alias_external = \
            query_construction.abbr_option[self.workload])  # 加载外部的别名
        workload_foreign_mapping = workload_spec.get_spec_foreign_mapping(workload_name = self.workload)
        current_parser.load_workload_info(workload_foreign_mapping)     # 加载workload的信息

        # print("current_parser.join_mapping = {}".format(current_parser.join_mapping))
        self.query_ctrl.set_query_instance(query_text = query_text, query_meta = query_meta)
        # sub_repr_list = self.query_ctrl.get_all_sub_relations()
        # sub_repr_list = self.data_manager.get_subqueries(alias_list=self.top_key)
        curr_key = [self.get_alias(s) for s in query_meta[0]]
        sub_repr_list = self.data_manager.get_subqueries(alias_list=curr_key)   # 优化表达式

        # print(f"get_query_cardinalities: sub_repr_list = {sub_repr_list}. query_meta = {query_meta[0]}.")

        subquery_repr_left, subquery_list = self.get_missing_subquery(sub_repr_list, current_parser)
        # print("sub_repr_left = {}.".format(sub_repr_left))
        # print("subquery_list = {}.".format(subquery_list))

        single_repr_left, single_table_list = self.get_missing_single_table(query_meta = \
            query_meta, sql_parser = current_parser)
        
        # print("single_repr_list = {}.".format(single_repr_list))
        # print("single_table_list = {}.".format(single_table_list))
        subquery_dict, single_table_dict = self.construct_full_cardinalities(
            subquery_repr_left, subquery_list, single_repr_left, single_table_list
        )

        return subquery_dict, single_table_dict
    

    def get_query_card_dict(self, query_meta):
        """
        {Description}
    
        Args:
            query_meta:
            arg2:
        Returns:
            return1:
            return2:
        """
        query_text = query_construction.construct_origin_query(\
            query_meta, workload = self.workload)
        current_parser = SQLParser(sql_text = query_text, workload=self.workload)

        workload_foreign_mapping = workload_spec.get_spec_foreign_mapping(workload_name = self.workload)
        current_parser.load_workload_info(workload_foreign_mapping)     # 加载workload的信息

        self.query_ctrl.set_query_instance(query_text = query_text, query_meta = query_meta)

        curr_key = [self.get_alias(s) for s in query_meta[0]]
        sub_repr_list = self.data_manager.get_subqueries(alias_list=curr_key)   # 优化表达式
 
        subquery_repr_left, subquery_list = self.get_missing_subquery(sub_repr_list, current_parser)
        single_repr_left, single_table_list = self.get_missing_single_table(query_meta = \
            query_meta, sql_parser = current_parser)
        
        subquery_dict = {k: v for k, v in zip(subquery_repr_left, subquery_list)}
        single_table_dict = {k: v for k, v in zip(single_repr_left, single_table_list)}
        
        return subquery_dict, single_table_dict

    def load_query_card_dict(self, query_meta, subquery_external, single_table_external):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            subquery_full:
            single_table_full:
        """
        subquery_full = utils.dict_merge(self.estimation_card_dict, subquery_external)
        single_table_full = utils.dict_merge(self.estimation_single_table, single_table_external)

        return subquery_full, single_table_full

    def batch_result_summary(self, instance_result_list):
        """
        分析一批的测试结果
        
        Args:
            instance_result_list:
        Returns:
            result_summary_(list/dict):
        """
        if isinstance(instance_result_list, (list,)):
            result_summary_list = []

            for origin_result, adjust_result_list in instance_result_list:
                result_summary_list.append(self.instance_result_summary(\
                    origin_result, adjust_result_list))
            return result_summary_list
        elif isinstance(instance_result_list, dict):
            result_summary_dict = {}
            for k, v in instance_result_list.items():
                result_summary_dict[k] = self.batch_result_summary(v)
            return result_summary_dict
        else:
            raise TypeError("Unsupported instance_result_list type: {}".\
                format(type(instance_result_list)))

    def instance_result_summary(self, origin_result, adjust_result_list):
        """
        针对一批结果进行分析总结
        
        Args:
            origin_result:
            adjust_result_list:
        Returns:
            valid_origin:
            valid_count:
            result_summary:
        """
        valid_count = 0
        flag, plan1, plan2 = origin_result
        valid_origin = flag
        valid_count += int(flag)
        result_summary = []
        if flag == True:
            result_summary.append((flag, plan1, plan2))

        for adjust_result in adjust_result_list:
            flag, plan1, plan2 = adjust_result
            if flag == True:
                result_summary.append((flag, plan1, plan2))

        return valid_origin, valid_count, result_summary

    def set_current_query(self, query_meta):
        """
        设置当前的查询
        
        Args:
            query_meta:
            arg2:
        Returns:
            query_text:
            res2:
        """
        query_text = query_construction.construct_origin_query(\
            query_meta, workload = self.workload)
        self.query_ctrl.set_query_instance(query_text = query_text, query_meta = query_meta)
        return query_text

    def plan_evaluation_under_cardinality(self, query_meta, subquery_dict, single_table_dict) -> physical_plan_info.PhysicalPlan:
        """ 
        {Description}
    
        Args:
            query_meta:
            subquery_dict:
            single_table_dict:
        Returns:
            return1:
            return2:
        """
        query_text = query_construction.construct_origin_query(\
            query_meta, workload = self.workload)
        # print("plan_evaluation_under_cardinality: query_text = {}.".format(query_text))
        # print("plan_evaluation_under_cardinality")

        self.query_ctrl.set_query_instance(query_text = query_text, query_meta = query_meta)
        res_plan = self.query_ctrl.get_plan_by_external_card(subquery_dict, single_table_dict)  # 获得注入基数的查询

        curr_physical_info = physical_plan_info.PhysicalPlan(query_text = \
            query_text, plan_dict = res_plan)   # 物理信息
        return curr_physical_info


    def plan_evaluation_under_spec(self, query_meta, subquery_dict, single_table_dict, \
        leading_hint, scan_ops, join_ops) -> physical_plan_info.PhysicalPlan:
        """
        在完整信息下对于查询计划的评估
        
        Args:
            query_meta:
            subquery_dict:
            single_table_dict:
            leading_hint:
            scan_ops:
            join_ops:
        Returns:
            res1:
            res2:
        """
        query_text = self.set_current_query(query_meta = query_meta)
        res_plan = self.query_ctrl.get_plan_by_external_info(subquery_dict = subquery_dict, \
            single_table_dict=single_table_dict, join_leading=leading_hint, scan_ops=scan_ops, join_ops=join_ops)

        curr_physical_info = physical_plan_info.PhysicalPlan(query_text = \
            query_text, plan_dict = res_plan)   # 物理信息
        return curr_physical_info


    def get_value_cnt_info(self, column_list):
        """
        {Description}
        
        Args:
            column_list:
            arg2:
        Returns:
            restriction:
            column_order:
            value_cnt_arr:
        """
        curr_key = self.column_list2key(column_list=column_list)
        restriction, column_order, value_cnt_array = self.global_value_cnt_info[curr_key]

        return restriction, column_order, value_cnt_array

    def get_global_bins_info(self, column_list):
        """
        {Description}
        
        Args:
            column_list:
        Returns:
            bins_dict:
            reverse_dict:
        """
        curr_key = self.column_list2key(column_list=column_list)
        bins_dict, reverse_dict = self.global_bins_info[curr_key]

        return bins_dict, reverse_dict


    def update_global_bins_info(self, column_list, bins_dict, reverse_dict):
        """
        {Description}
    
        Args:
            column_list:
            bins_dict:
            reverse_dict
        Returns:
            global_bins_info:
            return2:
        """
        # 
        curr_key = self.column_list2key(column_list)
        self.global_bins_info[curr_key] = (bins_dict, reverse_dict)
        return self.global_bins_info

    def update_global_value_cnt_info(self, column_list, restriction, value_cnt_array, column_order):
        """
        根据全局的value_cnt_array的信息
        
        Args:
            column_list:
            restriction:
            value_cnt_array:
            column_order: 列的顺序
        Returns:
            res1:
            res2:
        """
        curr_key = self.column_list2key(column_list=column_list)
        self.global_value_cnt_info[curr_key] = restriction, column_order, value_cnt_array

        return self.global_value_cnt_info


    def column_list2key(self, column_list):
        """
        将column_list转成元组key
    
        Args:
            column_list:
        Returns:
            key_tuple:
        """
        return tuple(sorted(column_list))

    # @utils.timing_decorator
    def construct_bins_dict(self, column_list, split_budget):
        """
        {Description}
    
        Args:
            column_list:
            split_budget:
        Returns:
            bins_dict:
            reverse_dict:
        """
        # 创建桶字典
        bins_dict = self.bins_builder.construct_bins_dict(selected_columns = column_list, \
            split_budget = split_budget)
        # 创建桶的inverse index
        reverse_dict = {}

        for column, bins_list in bins_dict.items():
            reverse_local = {}
            for idx, val in enumerate(bins_list):
                reverse_local[val] = idx
            reverse_dict[column] = reverse_local

        self.update_global_bins_info(column_list, bins_dict, reverse_dict)
        return bins_dict, reverse_dict


    def plan_evaluation(self, query_meta):
        """
        导入估计基数以后获得具体的查询计划

        Args:
            query_meta:
            arg2:
        Returns:
            curr_physical_info:
            return2:
        """
        query_text = query_construction.construct_origin_query(\
            query_meta, workload = self.workload)
        # print("plan_evaluation: query_text = {}.".format(query_text))

        self.query_ctrl.set_query_instance(query_text = query_text, query_meta = query_meta)
        # 获得估计的基数
        subquery_dict, single_table_dict = self.get_query_cardinalities(query_meta = query_meta)    
        res_plan = self.query_ctrl.get_plan_by_external_card(subquery_dict, single_table_dict)  # 获得注入基数的查询

        curr_physical_info = physical_plan_info.PhysicalPlan(query_text = \
            query_text, plan_dict = res_plan)   # 物理信息
        return curr_physical_info

    def physical_plan_fetch(self,):
        """
        获得当前查询对应的物理计划
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        query_text = self.query_text
        self.query_ctrl.set_query_instance(query_text = query_text, query_meta = self.query_meta)
        res_plan = self.query_ctrl.get_plan_by_external_card(subquery_dict = \
            self.estimation_card_dict, single_table_dict = self.estimation_single_table)

        self.physical_info = physical_plan_info.PhysicalPlan(query_text = \
            query_text, plan_dict = res_plan)   # 物理信息

        # print("打印查询实例的物理计划: ")
        # self.physical_info.show_plan()          # 展示物理查询计划
        return self.physical_info
    

    def get_alias(self, table_name):
        """
        获得表的别名
        
        Args:
            table_name:
        Returns:
            alias_name:
        """
        return self.data_manager.tbl_abbr[table_name]

    def target_plan_info(self, new_table, join_type = "nestloop"):
        """
        目标查询计划的信息
    
        Args:
            new_table: 新添加的table
            join_type: 新添加表连接使用的方法
        Returns:
            leading_hint:
            scan_ops:
            join_ops:
        """
        # print("target_plan_info: {}".format(self.physical_info.get_physical_spec()))   # 打印物理信息
        new_alias = self.get_alias(new_table)
        alias_list = [self.get_alias(t) for t in self.query_meta[0]]
        alias_list.append(new_alias)    # alias_list准备好

        leading_hint, scan_ops, join_ops = self.physical_info.get_physical_spec()
        # print("leading_hint = {}.".format(leading_hint))
        # print("scan_ops = {}.".format(scan_ops))
        # print("join_ops = {}.".format(join_ops))

        # 设置最后一层的join type
        if join_type is not None:
            join_ops[" ".join(alias_list)] = "Nested Loop"

        # 在leading_hint的基础上添加一个新的表
        LEADING_TEMPLATE = "Leading({elements})"

        def parse_leading(in_str):
            result = parse.parse(LEADING_TEMPLATE, in_str)
            return result

        parse_res = parse_leading(in_str = leading_hint)
        # print("target_plan_info: leading_hint = {}. parse_res = {}.".format(leading_hint, parse_res))
        old_order_elements = parse_res.named['elements']
        leading_hint = LEADING_TEMPLATE.format(elements = "({} {})".format(\
            old_order_elements, new_alias))
        return leading_hint, scan_ops, join_ops
        
    @utils.timing_decorator
    def construct_full_cardinalities(self, sub_repr_list, subquery_list, single_repr_list, single_table_list):
        """
        构建查询所需的完整基数
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        print(f"construct_full_cardinalities: ce_handler = {self.ce_handler}")

        subquery_dict = deepcopy(self.estimation_card_dict)
        single_table_dict = deepcopy(self.estimation_single_table)

        # print("subquery_list = {}.".format(subquery_list))
        subquery_cards = self.ce_handler.get_cardinalities(query_list = subquery_list)
        subquery_increment = self.dict_make(sub_repr_list, subquery_cards)
        subquery_dict = self.dict_complement(old_dict = subquery_dict, new_dict = subquery_increment)

        # print("single_table_list = {}.".format(single_table_list))
        single_table_cards = self.ce_handler.get_cardinalities(query_list = single_table_list)
        single_table_increment = self.dict_make(single_repr_list, single_table_cards)
        single_table_dict = self.dict_complement(old_dict = single_table_dict, new_dict = single_table_increment)

        # print()
        return subquery_dict, single_table_dict

    def get_missing_single_table(self, query_meta, sql_parser: SQLParser):
        """
        获得确实的单表基数
    
        Args:
            query_meta:
            sql_parser:
        Returns:
            single_repr_list:
            single_table_list:
        """
        single_repr_list = []
        single_table_list = []

        for schema in query_meta[0]:
            # 
            alias_name = self.data_manager.tbl_abbr[schema]  # schema的别名
            if alias_name not in self.estimation_single_table.keys():
                single_repr_list.append(alias_name)
                single_table_list.append(sql_parser.get_single_table_query(alias = alias_name))

        return single_repr_list, single_table_list

    # @utils.timing_decorator
    def get_missing_subquery(self, sub_repr_list, sql_parser: SQLParser):
        """
        获得未获得基数估计的子查询
        
        Args:
            sub_repr_list:
            sql_parser:
        Returns:
            sub_repr_left:
            subquery_list:
        """
        sub_repr_left = []
        subquery_list = []
        # print("sub_repr_list = {}.".format(sub_repr_list))
        # print("estimation_card_dict = {}.".format(self.estimation_card_dict))

        hit_cnt, miss_cnt = 0, 0
        for sub_repr in sub_repr_list:
            #
            if sub_repr not in self.estimation_card_dict:
                # 没有在历史字典出现，添加结果
                sub_repr_left.append(sub_repr)
                subquery_list.append(sql_parser.\
                    construct_PK_FK_sub_query(alias_list=sub_repr, workload_info=None))
                miss_cnt += 1
            else:
                hit_cnt += 1
        
        # print(f"get_missing_subquery: hit_cnt = {hit_cnt}. miss_cnt = {miss_cnt}.")
        # print("sub_repr_list = {}".format(sub_repr_list))
        return sub_repr_left, subquery_list


# %% 围绕QueryInstance相关的功能函数

def get_query_instance(workload, query_meta, ce_handler, external_dict = {}):
    """
    创建查询的实例，
    
    Args:
        workload:
        query_meta:
        external_dict: 外部的信息
    Returns:
        res_instance:
        res2:
    """
    # 相关组件的创建
    data_manager = external_dict.get("data_manager", data_management.DataManager(wkld_name=workload))
    mv_manager = external_dict.get("mv_manager", mv_management.MaterializedViewManager(workload=workload))
    # ce_handler = external_dict.get("ce_handler", ce_injection.PGInternalHandler(workload=workload))
    query_ctrl = external_dict.get("query_ctrl", query_exploration.QueryController(workload=workload))
    multi_builder = external_dict.get("multi_builder", grid_preprocess.MultiTableBuilder(workload = workload, \
        data_manager_ref = data_manager, mv_manager_ref = mv_manager))
    
    bins_builder = external_dict.get("bins_builder", grid_preprocess.BinsBuilder(workload=workload, 
        data_manager_ref=data_manager, mv_manager_ref=mv_manager))

    if isinstance(ce_handler, str):
        ce_handler = ce_injection.get_ce_handler_by_name(workload, ce_handler)
        
    res_instance = QueryInstance(workload=workload, query_meta=query_meta, 
        data_manager_ref=data_manager, ce_handler=ce_handler, query_controller=query_ctrl, 
        grid_preprocessor=multi_builder, bins_builder=bins_builder)
    
    return res_instance

def get_query_join_order(workload, query_meta, external_dict = {}):
    """
    获得查询的连接顺序
    
    Args:
        workload:
        query_meta:
        external_dict:
    Returns:
        join_order_str: 表示为连接顺序的字符串 
        res2:
    """
    ce_handler = external_dict['ce_handler']
    query_instance = get_query_instance(workload=workload, \
        query_meta=query_meta, ce_handler=ce_handler, external_dict=external_dict)
    local_plan = query_instance.physical_info
    join_order = local_plan.leading[len("Leading"):]    # 删除prefix
    return join_order


def get_query_join_order_batch(workload, meta_list):
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    # 装填外部信息
    data_manager = data_management.DataManager(wkld_name=workload)
    mv_manager = mv_management.MaterializedViewManager(workload=workload)
    external_dict = {
        "data_manager": data_manager,
        "mv_manager": mv_manager,
        "ce_handler": ce_injection.PGInternalHandler(workload=workload),
        "query_ctrl": query_exploration.QueryController(workload=workload),
        "multi_builder": grid_preprocess.MultiTableBuilder(workload = workload, \
            data_manager_ref = data_manager, mv_manager_ref = mv_manager)
    }

    # ts = time.time()
    join_order_list = [get_query_join_order(workload=workload, query_meta=meta_info,
                            external_dict=external_dict) for meta_info in meta_list]
    return join_order_list

# %%

def construct_instance_element(query_instance: QueryInstance):
    """
    构建查询实例的元素，抽取重要信息

    Args:
        query_instance:
        arg2:
    Returns:
        query_text:
        query_meta:
        card_dict:
    """
    query_text = query_instance.query_text
    query_meta = query_instance.query_meta
    card_dict = {
        "true": {
            "subquery": query_instance.true_card_dict,
            "single_table": query_instance.true_single_table
        },
        "estimation": {
            "subquery": query_instance.estimation_card_dict,
            "single_table": query_instance.estimation_single_table
        }
    }

    # 显示card_dict的内容
    # utils.display_card_dict(card_dict)
    
    return query_text, query_meta, card_dict

# %%
