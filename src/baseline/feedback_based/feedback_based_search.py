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

# %%

from utility import generator
from data_interaction import data_management, mv_management
from query import ce_injection, query_exploration
from grid_manipulation import grid_preprocess
from plan import node_query, node_extension, plan_init

from baseline.utility import util_funcs, base
from query import query_construction

# %%

class FBBasedPlanSearcher(base.BasePlanSearcher):
    """
    基于FeedBack的搜索器，在这里考虑两个feedback，分别是template和predicates

    template: 代表模版结构的调优，这个可以优先去实现，感觉比较简单
    predicates: 代表谓词大小的调优，考虑之后实现

    Members:
        field1:
        field2:
    """

    def __init__(self, schema_total, workload, mode: list, time_limit = 60000, ce_type:str = "internal"):
        """
        {Description}

        Args:
            mode: 调优模式，包含schema, column和predicates的
            arg2:
        """
        super(FBBasedPlanSearcher, self).__init__(workload=workload, ce_type=ce_type, time_limit=time_limit)

        # 模式列表
        self.mode = mode
        self.schema_total = schema_total

        self.search_initializer = plan_init.get_initializer_by_workload(schema_list=schema_total, workload=workload)
        self.alias_mapping = query_construction.abbr_option[workload]
        self.alias_inverse = {}
        for k, v in self.alias_mapping.items():
            self.alias_inverse[v] = k

        self.bins_builder = grid_preprocess.BinsBuilder(workload = workload, 
            data_manager_ref=self.data_manager, mv_manager_ref=self.mv_manager)

        self.schema_prob_dict, self.column_prob_dict, self.predicate_prob_dict = {}, {}, {}
        self.workload = workload


        # 概率字典的初始化
        self.prob_dict_init()

    def prob_dict_init(self,):
        """
        将所有概率字典初始化
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        # 全局workload的generator
        global_generator = generator.QueryGenerator(schema_list=self.schema_total, \
            dm_ref=self.data_manager, bins_builder=self.bins_builder, workload=self.workload)

        # self.schema_prob_dict = {}
        table_num = len(self.schema_total)
        for s in self.schema_total:
            self.schema_prob_dict[s] = 1 / table_num

        self.column_prob_dict = global_generator.table_prob_dict
        self.predicate_prob_dict = global_generator.column_prob_dict
        
        return self.schema_prob_dict, self.column_prob_dict, self.predicate_prob_dict
    
    def prob_dict_subset(self, schema_subset):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        table_prob_local, column_prob_local = {}, {}
        for k, v in self.column_prob_dict.items():
            if k in schema_subset:
                table_prob_local[k] = v

        for k, v in self.predicate_prob_dict.items():
            if k[0] in schema_subset:
                column_prob_local[k] = v

        return table_prob_local, column_prob_local
        

    def generate_schema_subset(self, schema_num):
        """
        {Description}
    
        Args:
            schema_num:
            arg2:
        Returns:
            schema_subset:
            return2:
        """
        print("FBBasedPlanSearcher.generate_schema_subset: self.schema_prob_dict = {}.".\
                format(self.schema_prob_dict))
        
        schema_list, prob_list = util_funcs.dict2list(self.schema_prob_dict)
        max_try_times = 5000
        all_comb_list = self.search_initializer.schema_combination_dict[schema_num]

        # 检验item的有序性，之后没问题可以注释掉
        for item in all_comb_list:
            if item != tuple(sorted(item)):
                raise ValueError("generate_schema_subset: item ({}) is unsorted. correct item is {}.".\
                                 format(item, sorted(item)))
        #
        while True:
            schema_subset = np.random.choice(a = schema_list, \
                                size=schema_num, replace=False, p = prob_list)
            max_try_times -= 1
            if max_try_times <= 0:
                print("FBBasedPlanSearcher.generate_schema_subset: schema_subset = {}.".format(schema_subset))
                print("FBBasedPlanSearcher.generate_schema_subset: all_comb_list = {}.".format(all_comb_list))
                # raise ValueError("generate_schema_subset: max_try_time exceeded!")
                # 退化到随机选一个
                idx = int(np.random.choice(range(len(all_comb_list))))
                schema_subset = all_comb_list[idx]
                break
            if tuple(sorted(schema_subset)) not in all_comb_list:
                # 不合法的情况
                continue
            else:
                # 合法就直接退出
                break

        return schema_subset

    def workload_generation(self, query_num):
        """
        生成workload

        Args:
            query_num: 查询数目
            arg2:
        Returns:
            query_list:
            meta_list:
        """
        query_list, meta_list = [], []
        for _ in range(query_num):
            # 生成schema_list，暂时指定为5
            schema_local = self.generate_schema_subset(schema_num=5)

            # 生成概率表
            # table_prob_local, column_prob_local = {}, {}
            table_prob_local, column_prob_local = self.prob_dict_subset(schema_subset=schema_local)
            # 生成查询
            local_gen = generator.QueryGenerator(schema_list=schema_local, dm_ref=self.data_manager, bins_builder = self.bins_builder, \
                            workload=self.workload, table_prob_config=table_prob_local, column_prob_config=column_prob_local)
            query_text, query_meta = local_gen.generate_query(with_meta=True)
            query_list.append(query_text)
            meta_list.append(query_meta)

        return query_list, meta_list


    def schema_prob_table_adjustment(self, meta_list, result_list):
        """
        schema概率表的调整

        Args:
            meta_list: 
            result_list: 
        Returns:
            return1: 
            return2: 
        """
        def update_single_item(schema, val):
            self.schema_prob_dict[schema] *= val

        for query_meta, p_error in zip(meta_list, result_list):
            schema_list, filter_list = query_meta
            for schema in schema_list:
                if p_error >= 1.2:
                    val = 1.1
                elif p_error <= 1.1:
                    val = 1 / 1.1
                else:
                    val = 1.0
                update_single_item(schema=schema, val = val)

        # 归一化调整
        self.schema_prob_dict = util_funcs.prob_dict_resize(self.schema_prob_dict)

        return self.schema_prob_dict

    def column_prob_table_adjustment(self, meta_list, result_list):
        """
        column概率表的调整，暂时根据p_error进行调优，期望

        Args:
            meta_list:
            result_list:
        Returns:
            return1:
            return2:
        """
        def update_single_item(schema_name, column_name, val):
            # 用乘法完成更新
            self.column_prob_dict[schema_name][column_name] *= val

        for query_meta, p_error in zip(meta_list, result_list):
            schema_list, filter_list = query_meta
            for item in filter_list:
                alias_name, column_name = item[0], item[1]
                schema_name = self.alias_inverse[alias_name]
                if p_error >= 1.2:
                    val = 2.0
                elif p_error <= 1.1:
                    val = 0.5
                else:
                    val = 1.0
                    
                update_single_item(schema_name=schema_name, \
                                   column_name=column_name, val = val)
                
        # 结果归一化
        # self.column_prob_dict = util_funcs.prob_dict_resize(self.column_prob_dict)
        res_dict = {}
        for schema, local_dict in self.column_prob_dict.items():
            res_dict[schema] = util_funcs.prob_dict_resize(local_dict)
        # return self.column_prob_dict
        self.column_prob_dict = res_dict
        return res_dict

    def predicate_prob_table_adjustment(self, meta_list, result_list):
        """
        谓词概率表的调整

        Args:
            meta_list:
            result_list:
        Returns:
            return1:
            return2:
        """
        # TODO: 这个之后慢慢实现，有点复杂
        pass


    def batch_plans_evaluation(self, query_list, meta_list):
        """
        评测一批查询计划的好坏
    
        Args:
            query_list:
            meta_list:
        Returns:
            result_list:
            card_dict_list:
        """
        result_list = []
        card_dict_list = []
        for query_text, query_meta in zip(query_list, meta_list):
            print("batch_plans_evaluation: query_text = {}. query_meta = {}.".format(query_text, query_meta))
            result, card_dict = self.explore_query(query_text=query_text, query_meta=query_meta)
            result_list.append(result)
            card_dict_list.append(card_dict)

        return result_list, card_dict_list


    def update_prob_tables(self, meta_list, p_error_list):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        # 根据要求完成概率表的更新
        if "schema" in self.mode:
            self.schema_prob_table_adjustment(meta_list=meta_list, result_list=p_error_list)
        if "column" in self.mode:
            self.column_prob_table_adjustment(meta_list=meta_list, result_list=p_error_list)

        if "predicate" in self.mode:
            raise NotImplementedError("predicate_prob_table_adjustment还没有实现好!")
        
        return True
    
    def run_episode(self, query_num):
        """
        运行一个周期
    
        Args:
            arg1:
            arg2:
        Returns:
            query_local:
            result_local:
        """
        # query_local, meta_local = [], []
        # result_local, card_dict_local = [], []

        # 首先生成workload
        query_local, meta_local = self.workload_generation(query_num=query_num)

        # 获得评测的结果
        result_local, card_dict_local = self.batch_plans_evaluation(query_list=query_local, meta_list=meta_local)

        # 提取p_error的值
        p_error_local = [item[0] for item in result_local]

        # # 根据要求完成概率表的更新
        # if "schema" in self.mode:
        #     self.schema_prob_table_adjustment(meta_list=meta_local, result_list=p_error_local)
        # if "column" in self.mode:
        #     self.column_prob_table_adjustment(meta_list=meta_local, result_list=p_error_local)

        # if "predicate" in self.mode:
        #     raise NotImplementedError("predicate_prob_table_adjustment还没有实现好!")
        
        self.update_prob_tables(meta_list=meta_local, p_error_list=p_error_local)
        
        return query_local, meta_local, result_local, card_dict_local

    def launch_search_process(self, total_time, with_start_time=False):
        """
        启动搜索的过程

        Args:
            budget: 搜索的预算
            constraint: 搜索的限制
        Returns:
            query_list:
            meta_list:
            result_list:
            card_dict_list:
        """
        print("call FBBasedPlanSearcher.launch_search_process.")

        start_time = time.time()
        self.query_list, self.result_list = [], []
        while True:
            query_local, meta_local, result_local, card_dict_local = self.run_episode(query_num=5)

            self.query_list.extend(query_local)
            self.meta_list.extend(meta_local)
            self.result_list.extend(result_local)
            self.card_dict_list.extend(card_dict_local)

            end_time = time.time()
            if end_time - start_time > total_time:
                break

        return self.query_list, self.meta_list, self.result_list, self.card_dict_list


    def get_search_result(self,):
        """
        获得搜索的结果

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        pass