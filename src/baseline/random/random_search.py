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
from query import query_exploration, ce_injection
from plan import plan_init, node_query, node_extension
from utility import generator
from baseline.utility import base
from data_interaction import data_management, mv_management
from grid_manipulation import grid_preprocess

# %%

class RandomPlanSearcher(base.BasePlanSearcher):
    """
    {Descript

    Members:
        field1:
        field2:
    """

    def __init__(self, schema_total, workload, table_num_dist, time_limit = 60000, ce_type:str = "internal"):
        """
        {Description}

        Args:
            schema_total: 探索所有的数据表范围
            workload: 工作负载
            table_num_dist: 总的table数分布情况
            time_limit: 单个查询计划探索的时间限制
            ce_type: 基数估计器的类型
        """
        print("RandomPlanSearcher.__init__: workload = {}.".format(workload))
        self.workload = workload
        self.search_initializer = plan_init.get_initializer_by_workload(schema_list=schema_total, workload=workload)
        self.table_num_dist = table_num_dist
        
        # 调用父类构造函数
        super(RandomPlanSearcher, self).__init__(workload=workload, ce_type=ce_type, time_limit=time_limit)

    def single_query_generation(self, ):
        """
        生成所有的schema
    
        Args:
            arg1:
            arg2:
        Returns:
            query_text: 
            query_meta:
        """
        def dict2list(in_dict):
            key_list, value_list = [], []
            for k, v in in_dict.items():
                key_list.append(k)
                value_list.append(v)
            return key_list, value_list
        
        value_list, prob_list = dict2list(self.table_num_dist)
        table_num = int(np.random.choice(a=value_list, p=prob_list))
        schema_comb_list = self.search_initializer.schema_combination_dict[table_num]
        selected_idx = np.random.randint(len(schema_comb_list))
        schema_comb = schema_comb_list[selected_idx]
        query_text, query_meta = self.search_initializer.single_query_generation(schema_subset=schema_comb)
        return query_text, query_meta


    def launch_search_process(self, total_time, with_start_time=False):
        """
        {Description}

        Args:
            total_time: 总的时间
            with_start_time: 先把参数加上，暂不执行
        Returns:
            query_list: 
            meta_list: 
            result_list:
            card_dict_list: 
        """
        print("call RandomPlanSearcher.launch_search_process.")

        start_time = time.time()
        while True:
            # query_text, result = self.explore_query()
            query_text, query_meta = self.single_query_generation()
            result, card_dict = self.explore_query(query_text=query_text, query_meta=query_meta)

            self.add_exploration_instance(query_text=query_text, \
                query_meta=query_meta, result=result, card_dict=card_dict)
            
            end_time = time.time()
            delta_time = end_time - start_time
            if delta_time > total_time:
                break

        return self.query_list, self.meta_list, self.result_list, self.card_dict_list

    def return_results(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            pair_list: (query, P_Error)组成的列表
        """
        return list(zip(self.query_list, self.result_list))
