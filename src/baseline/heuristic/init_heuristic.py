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
from baseline.utility import base
from query import ce_injection
from plan import plan_init, node_extension, node_query
from query import query_construction
from data_interaction import mv_management
from utility import generator

# %%

class InitGreedyPlanSearcher(base.BasePlanSearcher):
    """
    探索子查询中Q_error大的结果，然后在上面添加新的Condition进行实验

    Members:
        field1:
        field2:
    """

    def __init__(self, schema_total, workload, time_limit = 60000, ce_type:str = "internal"):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        self.schema_total = schema_total
        self.workload = workload
        self.search_initializer = plan_init.get_initializer_by_workload(schema_list=schema_total, workload=workload)

        super(InitGreedyPlanSearcher, self).__init__(workload=workload, ce_type=ce_type, time_limit=time_limit)


    def workload_evaluation(self, query_list, meta_list):
        """
        {Description}
        
        Args:
            query_list:
            meta_list:
        Returns:
            estimation_list:
        """
        # 需要考虑超时的问题
        estimation_list = self.ce_handler.get_cardinalities(query_list=query_list)
        return estimation_list
    

    def find_candidate_queries(self, query_list, meta_list, label_list, estimation_list, num):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            selected_query:
            selected_meta:
        """
        q_error_list = [max(true_card / (2.0 + estimation_card), estimation_card / (true_card + 2.0)) for \
                        true_card, estimation_card in zip(label_list, estimation_list)]
        selected_query, selected_meta = [], []
        sorted_index = np.argsort(q_error_list)[::-1]

        for idx in sorted_index[:num]:
            selected_query.append(query_list[idx])
            selected_meta.append(meta_list[idx])

        return selected_query, selected_meta


    def complement_left_conditions(self, query_text, query_meta, extend_num, out_num):
        """
        补全剩余的信息

        Args:
            query_text: 
            query_meta: 
            extend_num: 
            query_num:
        Returns:
            query_list:
            meta_list:
        """
        query_list, meta_list = [], []
        schema_init = query_meta[0]
        candidate_schema_comb = []

        # print("InitGreedyPlanSearcher.complement_left_conditions: schema_init = {}.".format(schema_init))
        all_comb_set = self.search_initializer.schema_combination_dict[len(query_meta[0]) + extend_num]
        for schema_comb in all_comb_set:
            if set(schema_init).issubset(set(schema_comb)):
                # print("InitGreedyPlanSearcher.complement_left_conditions: schema_comb = {}.".format(schema_comb))
                candidate_schema_comb.append(schema_comb)

        for _ in range(out_num):
            selected_idx = int(np.random.randint(len(candidate_schema_comb)))
            selected_schema_comb = candidate_schema_comb[selected_idx]
            left_schema_comb = list(set(selected_schema_comb).difference(set(schema_init)))

            left_query, left_meta = self.search_initializer.\
                single_query_generation(schema_subset=left_schema_comb)
            final_meta = mv_management.meta_merge(left_meta=left_meta, right_meta=query_meta)
            final_query = query_construction.construct_origin_query(\
                query_meta=final_meta, workload=self.workload)
            
            query_list.append(final_query)
            meta_list.append(final_meta)

        return query_list, meta_list

    def set_properties(self, budget, constraint):
        """
        {Description}
    
        Args:
            budget: 搜索的预算
            constraint: 搜索的限制
        Returns:
            return1:
            return2:
        """
        pass


    def generate_candidates(self, workload_gen_num, candidate_num, \
            schema_init_num, schema_final_num, out_num = 3):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        left_num = schema_final_num - schema_init_num

        query_list, meta_list, label_list = self.search_initializer.workload_generation(\
            table_num_dist={schema_init_num:1.0} , total_num=workload_gen_num, timeout=20000)
        estimation_list = self.workload_evaluation(query_list=query_list, meta_list=meta_list)

        init_selected_query, init_selected_meta = self.find_candidate_queries(\
            query_list, meta_list, label_list, estimation_list, num=candidate_num)
        
        final_selected_query, final_selected_meta = [], []
        for init_query, init_meta in zip(init_selected_query, init_selected_meta):
            query_local, meta_local = self.complement_left_conditions(query_text=init_query, \
                query_meta=init_meta, extend_num=left_num, out_num=out_num)
            final_selected_query.extend(query_local)
            final_selected_meta.extend(meta_local)

        return final_selected_query, final_selected_meta
    
    def launch_search_process(self, total_time, with_start_time=False):
        """
        启动搜索的过程，完善返回结果，一方面是为了展示，

        Args:
            total_time:
        Returns:
            query_list:
            meta_list:
            result_list:
            card_dict_list:
        """
        print("call InitGreedyPlanSearcher.launch_search_process.")

        workload_gen_num = 25
        candidate_num = 5
        schema_init_num, schema_final_num = 3, 5
        out_num = 3

        is_terminate = False
        start_time = time.time()

        while True:
            final_selected_query, final_selected_meta = self.generate_candidates(\
                workload_gen_num, candidate_num, schema_init_num, schema_final_num, out_num)

            for test_query, test_meta in zip(final_selected_query, final_selected_meta):
                # 探索
                result, card_dict = self.explore_query(test_query, test_meta)
                # 添加结果
                self.add_exploration_instance(query_text = test_query, query_meta=test_meta,
                                              result=result, card_dict=card_dict)
                end_time = time.time()
                if end_time - start_time > total_time:
                    is_terminate = True
                    break

            if is_terminate == True:
                break

        return self.query_list, self.meta_list, self.result_list, self.card_dict_list
