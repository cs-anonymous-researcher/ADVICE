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
from plan import plan_init, node_query, node_extension

# %%

class FinalGreedyPlanSearcher(base.BasePlanSearcher):
    """
    探索最终查询中Q_error大的结果，然后在上面添加新的Condition进行实验

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

        super(FinalGreedyPlanSearcher, self).__init__(workload=workload, ce_type=ce_type, time_limit=time_limit)

    
    def workload_evaluation(self, query_list, meta_list):
        """
        {Description}
        
        Args:
            query_list:
            meta_list:
        Returns:
            label_list:
            estimation_list:
        """
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
        q_error_list = [max(true_card / (1 + estimation_card), estimation_card / (true_card + 1)) \
                        for true_card, estimation_card in zip(label_list, estimation_list)]
        
        selected_query, selected_meta = [], []
        sorted_index = np.argsort(q_error_list)[::-1]

        for idx in sorted_index[:num]:
            selected_query.append(query_list[idx])
            selected_meta.append(meta_list[idx])

        return selected_query, selected_meta


    def launch_search_process(self, total_time, with_start_time=False):
        """
        启动搜索的过程

        Args:
            total_time: 总的搜索时间，以秒为单位
        Returns:
            return1:
            return2:
        """
        print("call FinalGreedyPlanSearcher.launch_search_process.")

        workload_gen_num = 25
        candidate_num = 5
        schema_num = 5
        is_terminate = False
        start_time = time.time()

        while True:
            query_list, meta_list, label_list = self.search_initializer.workload_generation(\
                table_num_dist={schema_num:1.0} , total_num=workload_gen_num, timeout=20000)
            estimation_list = self.workload_evaluation(query_list=query_list, meta_list=meta_list)

            selected_query, selected_meta = self.find_candidate_queries(\
                query_list, meta_list, label_list, estimation_list, num=candidate_num)

            for test_query, test_meta in zip(selected_query, selected_meta):
                result, card_dict = self.explore_query(test_query, test_meta)
                self.add_exploration_instance(query_text=test_query, query_meta=test_meta,
                                              result=result, card_dict=card_dict)

                end_time = time.time()
                if end_time - start_time > total_time:
                    is_terminate = True
                    break

            if is_terminate == True:
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