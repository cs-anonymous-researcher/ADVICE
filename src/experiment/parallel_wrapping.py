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
from experiment import parallel_exploration, template_exploration
from baseline.utility import base


# %%

class FeedbackAwareParallelSearcher(base.BasePlanSearcher):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, schema_total, workload, tmpl_meta_path, time_limit = 60000, max_step = 10, ce_type:str = "internal", \
            expl_estimator = "graph_corr_based", resource_config = parallel_exploration.default_resource_config,):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        self.schema_total = schema_total
        self.workload = workload

        super(FeedbackAwareParallelSearcher, self).__init__(workload, ce_type, time_limit)

        # 默认的实验配置

        #
        selected_tables = schema_total
        
        expt_config = {
            "selected_tables": selected_tables,
            "ce_handler": ce_type
        }
    
        self.join_order_evaluator = template_exploration.JoinOrderEvaluator(workload, selected_tables)
        self.ref_query_evaluator = template_exploration.ReferenceQueryEvaluator(workload, selected_tables)

        # 模版动态生成的相关配置
        dynamic_config = {
            "time": 25,                 # 时间至少大于此阈值
            "cardinality": 5000000     # 结果基数至少要大于阈值
        }

        # path_split_list = os.path.split(tmpl_meta_path)
        def split_path(path):
            parts = []
            while True:
                path, tail = os.path.split(path)
                if tail:
                    parts.insert(0, tail)
                else:
                    if path:
                        parts.insert(0, path)
                    break
            return parts
        
        path_split_list = split_path(tmpl_meta_path)
        # 模版存储的命名空间
        namespace = path_split_list[-2]

        if namespace == ce_type:
            namespace = None
        
        # print("")
        self.template_explorer = template_exploration.TemplateExplorationExperiment(workload, \
            ce_type, table_num=5, dynamic_config=dynamic_config, namespace=namespace)

        self.forest_explorer = parallel_exploration.ParallelForestExploration(workload, \
            expt_config=expt_config, expl_estimator=expl_estimator, resource_config=resource_config, \
            max_expl_step=max_step, tmpl_meta_path = tmpl_meta_path)


    def reset_state(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        self.join_order_evaluator.result_dict = {}
        self.ref_query_evaluator.current_result = {}
        # self.template_explorer.
        self.forest_explorer.reset_search_state()
    
    def launch_search_process(self, total_time: int):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        jo_evaluator = self.join_order_evaluator

        preprocess_start = time.time()
        if self.generate_join_order == True and self.generate_template == True:
            # 生成新结果
            min_jo_query = 500
            comb_set_num = 10
            num_per_comb = int(np.ceil(min_jo_query / comb_set_num))
            jo_evaluator.join_leading_priori(schema_total=5, \
                comb_set_num=comb_set_num, num_per_comb=num_per_comb)
        else:
            # 加载历史结果
            jo_evaluator.load_join_order_result()

        leading_comb_set = jo_evaluator.result_aggregation(priori_num = 3, out_num = 5)

        ref_evaluator = self.ref_query_evaluator
        if self.generate_ref_query == True and self.generate_template == True:
            # 生成新结果
            min_ref_query = 250
            gen_config = {
                'mode': "priori",
                "schema_comb_set": leading_comb_set,
                "num": min_ref_query,
            }
            ref_evaluator.custom_workload_generation(config=gen_config)
            query_list, meta_list, card_list = ref_evaluator.get_current_result()
        else:
            # 加载历史结果
            query_list, meta_list, card_list = ref_evaluator.result_aggregation({})
        
        query_aggr, meta_aggr, card_aggr = ref_evaluator.\
            filter_valid_records(query_list, meta_list, card_list)
        
        tmpl_explorer = self.template_explorer
        tmpl_explorer.load_historical_workload(query_aggr, meta_aggr, card_aggr)
        over_template_num, under_template_num = 4, 6
        if self.generate_template == True:
            # 
            tmpl_explorer.construct_hybrid_template_dict(over_template_num=over_template_num, \
                under_template_num=under_template_num, existing_range="", bins_builder=\
                ref_evaluator.search_initilizer.bins_builder)
            meta_path = tmpl_explorer.get_meta_path()
            self.forest_explorer.update_template_info(new_meta_path=meta_path)
        else:
            #
            pass

        preprocess_end = time.time()
        print(f"launch_search_process: preprocess delta_time = {preprocess_end - preprocess_start: .2f}")

        explore_mode = self.explore_mode
        valid_list = ["polling_based_parallel", "epsilon_greedy_parallel", "correlated_MAB_parallel"]
        assert explore_mode in valid_list, f"launch_search_process: explore_mode = {explore_mode}. valid_list = {valid_list}"
        
        config_dict = {
            "template_id_list": "all", 
            # "step": 10, 
            "root_config": {    # 创建根节点的配置
                "mode": "bayesian",
                "min_card": 1000, 
                "max_card": 10000000000,
                "num": 40
            }, 
            "tree_config": {    # 树探索的配置
                "max_depth": 6,
            }, 
            "search_config": {  # 搜索过程的配置
                "max_step": 10,
                "return": "full"
            },
            "total_time": total_time - (preprocess_end - preprocess_start)
        }

        if explore_mode == "polling_based_parallel":
            result = self.forest_explorer.polling_based_workload_generation(**config_dict)
        elif explore_mode == "epsilon_greedy_parallel":
            result = self.forest_explorer.Epsilon_Greedy_workload_generation(**config_dict)
        elif explore_mode == "correlated_MAB_parallel":
            result = self.forest_explorer.Correlated_MAB_workload_generation(**config_dict)
            
        query_list, meta_list, result_list, card_dict_list = result
        return query_list, meta_list, result_list, card_dict_list


    def set_config(self, explore_mode, generate_join_order = False, generate_ref_query = False, generate_template = False):
        """
        设置新的实验配置

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        self.explore_mode, self.generate_join_order, self.generate_ref_query, self.generate_template = \
            explore_mode, generate_join_order, generate_ref_query, generate_template
        self.show_config()

    def show_config(self,):
        """
        显示当前的实验配置
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        print("FeedbackAwareParallelSearcher: explore_mode = {}. generate_join_order = {}. generate_ref_query = {}. generate_template = {}.".\
              format(self.explore_mode, self.generate_join_order, self.generate_ref_query, self.generate_template))

    def construct_templates(self,):
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

    
    def construct_join_order_queries(self,):
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

    def construct_reference_queries(self,):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        pass

# %%
