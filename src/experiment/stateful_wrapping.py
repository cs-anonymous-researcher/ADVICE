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

from experiment import parallel_exploration, template_exploration
from baseline.utility import base
from experiment import stateful_exploration
from utility import common_config

# %%

default_sample_config = {
    "jo_query_num": 500,
    "comb_set_num": 10,
    "order_aggr_num": 20,
    "ref_query_min": 200,
    "ref_query_max": 2000,
    "ref_total_time": 60000,
    "ref_timeout": 10000
}

default_template_config = {
    "over_estimation_num": 4,
    "under_estimation_num": 6,
    "priori_num": 3,
    "cardinality_constraint": 100000000,
    "time_constraint": 30
}

default_warm_up_config = {
    "max_expl_step": 20,
    "warm_up_num": 3
}

class StatefulParallelSearcher(base.BasePlanSearcher):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    # def __init__(self, schema_total, workload, tmpl_meta_path, time_limit = 60000, max_step = 10, ce_type:str = "internal", 
    def __init__(self, schema_total, workload, tmpl_meta_path, time_limit = 60000, max_step = 100, ce_type:str = "internal", 
            expl_estimator = "dummy", resource_config = stateful_exploration.default_resource_config, 
            card_est_input = "graph_corr_based", action_selection_mode = "local", root_selection_mode = "normal", 
            noise_parameters = None, sample_config = default_sample_config, template_config = default_template_config, 
            split_budget = 100):
        """
        {Description}

        Args:
            schema_total:
            workload:
            tmpl_meta_path:
            time_limit: 
            max_step: 
            ce_type: 
            expl_estimator: 
            resource_config: 
            card_est_input: 
            action_selection_mode: 
            root_selection_mode:
            noise_parameters: 
            sample_config: 
            template_config:
            split_budget:
        """
        self.schema_total = schema_total
        self.workload = workload
        super(StatefulParallelSearcher, self).__init__(workload, ce_type, time_limit)

        print(f"stateful_wrapping.StatefulParallelSearcher: split_budget = {split_budget}. "\
              f"sample_config = {sample_config}. template_config = {template_config}.")

        # 用做测试
        # exit(-1)
        selected_tables = schema_total
        
        expt_config = {
            "selected_tables": selected_tables,
            "ce_handler": ce_type
        }
        self.expt_config = expt_config
        
        self.sample_config, self.template_config = sample_config, template_config
        self.join_order_evaluator = template_exploration.JoinOrderEvaluator(\
            workload, selected_tables, split_budget = split_budget)
        self.ref_query_evaluator = template_exploration.ReferenceQueryEvaluator(\
            workload, selected_tables, split_budget = split_budget)

        # 模版动态生成的相关配置
        # dynamic_config = {
        #     "time": 25,                 # 时间至少大于此阈值
        #     "cardinality": 5000000      # 结果基数至少要大于阈值
        # }
        # 2024-03-12: 设置template包含的表数目
        priori_num = self.template_config["priori_num"]
        common_config.template_table_num = priori_num
        
        dynamic_config = {
            "time": self.template_config["time_constraint"], 
            "cardinality": self.template_config["cardinality_constraint"]
        }

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

        self.template_explorer = template_exploration.TemplateExplorationExperiment(
            workload, ce_type, table_num=5, dynamic_config=dynamic_config, 
            namespace=namespace, dump_strategy = "remain", split_budget=split_budget)
        
        self.forest_explorer = stateful_exploration.StatefulExploration(workload, 
            expt_config=expt_config, expl_estimator=expl_estimator, resource_config=resource_config, 
            max_expl_step=max_step, tmpl_meta_path=tmpl_meta_path, card_est_input=card_est_input, 
            action_selection_mode=action_selection_mode, root_selection_mode=root_selection_mode, 
            noise_parameters=noise_parameters, warm_up_num=common_config.warm_up_num, 
            tree_config=common_config.tree_config_dict[workload], init_strategy=common_config.init_strategy)

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
    
    def search_initialization(self, min_jo_query = 500, comb_set_num = 10, min_ref_query = 200, max_ref_query = 2000):
        """
        搜索过程的初始化

        Args:
            arg1:
            arg2:
        Returns:
            delta_time:
            return2:
        """
        # print(f"search_initialization: generate_join_order = {self.generate_join_order}. "\
        #       f"generate_template = {self.generate_template}. generate_ref_query = {self.generate_ref_query}.")

        jo_evaluator = self.join_order_evaluator
        preprocess_start = time.time()
        if self.generate_join_order == True and self.generate_template == True:
            # 生成新结果
            num_per_comb = int(np.ceil(min_jo_query / comb_set_num))
            jo_evaluator.join_leading_priori(schema_total=5, \
                comb_set_num=comb_set_num, num_per_comb=num_per_comb)
        else:
            # 加载历史结果
            jo_evaluator.load_join_order_result()

        # leading_comb_set = jo_evaluator.result_aggregation(\
        #     priori_num = 3, out_num = self.sample_config["order_aggr_num"])
        leading_comb_set = jo_evaluator.result_aggregation(priori_num = 
            self.template_config["priori_num"], out_num = self.sample_config["order_aggr_num"])

        ref_evaluator = self.ref_query_evaluator

        print(f"search_initialization: leading_comb_set = {leading_comb_set}.")
        # return ref_evaluator, query_list, meta_list, label_list
        if self.generate_ref_query == True and self.generate_template == True:
            # phase 0，检测生成测试查询的质量
            schema_comb_list = list(leading_comb_set)
            error_threshold = 5.0
            batch_comb_num, batch_query_num = 5, 100
            time_limit, retain_limit = self.sample_config["ref_total_time"] // 1000, 3
            
            ce_str = self.ce_type
            query_list, meta_list, label_list = ref_evaluator.iterative_workload_generation(\
                schema_comb_list, error_threshold, batch_comb_num, batch_query_num, \
                time_limit, retain_limit, min_ref_query, max_ref_query, ce_str, self.sample_config["ref_timeout"])
            query_list, meta_list, card_list = ref_evaluator.get_current_result()
        else:
            # 加载历史结果
            query_list, meta_list, card_list = ref_evaluator.result_aggregation({})
        
        query_aggr, meta_aggr, card_aggr = ref_evaluator.\
            filter_valid_records(query_list, meta_list, card_list)

        # phase 1
        # metrics_dict = tmpl_explorer.get_workload_error_state()
        # return metrics_dict
    
        # phase 2
        # result_list, filtered_list = tmpl_explorer.select_potential_templates(\
        #     num = under_template_num, strategy="max-greedy", mode="under")
        # return result_list, filtered_list

        # phase 3
        if self.generate_template == True:
            tmpl_explorer = self.template_explorer
            tmpl_explorer.load_historical_workload(query_aggr, meta_aggr, card_aggr)
            over_template_num, under_template_num = \
                self.template_config['over_estimation_num'], self.template_config['under_estimation_num']

            # 2024-03-30: 根据error智能选择template数目
            total_template_num = over_template_num + under_template_num
            over_template_num, under_template_num = tmpl_explorer.workload_manager.select_template_number(\
                batch_id="latest", total_num=total_template_num, min_num=2)

            # 生成新的Template用于测试
            tmpl_explorer.construct_hybrid_template_dict(over_template_num=over_template_num, \
                under_template_num=under_template_num, existing_range="", bins_builder=\
                ref_evaluator.search_initilizer.bins_builder, schema_duplicate = self.template_config['schema_duplicate'])
            meta_path = tmpl_explorer.get_meta_path()
            # 
            self.forest_explorer.update_template_info(new_meta_path=meta_path)
        else:
            #
            pass
        preprocess_end = time.time()
        return preprocess_end - preprocess_start

    def launch_search_process(self, total_time: int, with_start_time = False, template_only = False):
        """
        {Description}

        Args:
            total_time:
            with_start_time:
            template_only:
        Returns:
            return1:
            return2:
        """
        print(f"StatefulParallelSearcher.launch_search_process: total_time = {total_time}. with_start_time = {with_start_time}. template_only = {template_only}.")
        if with_start_time:
            self.start_time = time.time()

        delta_time = self.search_initialization(self.sample_config["jo_query_num"], \
            self.sample_config["comb_set_num"], self.sample_config["ref_query_min"], self.sample_config["ref_query_max"])
        print(f"launch_search_process: preprocess delta_time = {delta_time: .2f}")

        # explore_mode = self.explore_mode
        # valid_list = ["polling_based_parallel", "epsilon_greedy_parallel", "correlated_MAB_parallel"]
        # assert explore_mode in valid_list, f"launch_search_process: explore_mode = {explore_mode}. valid_list = {valid_list}"
        
        config_dict = {
            "template_id_list": "all", 
            # "step": 10, 
            "root_config": {    # 创建根节点的配置
                # "mode": "bayesian",
                "mode": "sample-based",
                "min_card": 1000, 
                "max_card": 100000000000,
                # "num": 40
                "num": 100,
            }, 
            "tree_config": {    # 树探索的配置
                "max_depth": 8,
            }, 
            "search_config": {  # 搜索过程的配置
                "max_step": 20,
                "return": "full"
            },
            "total_time": total_time - delta_time   # 考虑预处理时间，用做最终的实验对比
            # "total_time": total_time        # 不考虑预处理时间，仅用作测试
        }

        # 20240224: 直接退出，用做测试
        # exit(-1)
        if template_only == True:
            # 如果只考虑生成template，直接退出
            if with_start_time == False:
                return [], [], [], []
            else:
                return [], [], [], [], ()

        result = self.forest_explorer.stateful_workload_generation(**config_dict)
        if with_start_time:
            self.end_time = time.time()

        query_list, meta_list, result_list, card_dict_list = result

        if with_start_time == False:
            return query_list, meta_list, result_list, card_dict_list
        else:
            time_info = self.forest_explorer.time_list, self.start_time, self.end_time
            return query_list, meta_list, result_list, card_dict_list, time_info

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
        
        print(f"StatefulParallelSearcher.set_config: generate_template = {self.generate_template}")
        if self.generate_template == True:
            # 20231123: 调用overwrite，会删除之前的结果
            self.template_explorer.template_manager.clean_directory()
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
        print("StateAwareParallelSearcher: explore_mode = {}. generate_join_order = {}. generate_ref_query = {}. generate_template = {}.".\
              format(self.explore_mode, self.generate_join_order, self.generate_ref_query, self.generate_template))


# %%
