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
from experiment import forest_exploration
from baseline.utility import base

# %%

class StateAwareSearcher(base.BasePlanSearcher):
    """
    两阶段探索的Searcher封装

    Members:
        field1:
        field2:
    """

    def __init__(self, schema_total, workload, expt_config, time_limit = 60000, ce_type:str = "internal"):
        """
        expt_config = {
            "table_num": "生成查询表的数目",
            "namespace": "结果保存的命名空间",
            "sample_queries": "是否生成sample_queries",
            "generate_template": "是否生成template",
            "explore_mode": "树的探索模式"
        }

        Args:
            arg1:
            arg2:
        """
        self.expt_config = expt_config
        self.schema_total = schema_total
        super(StateAwareSearcher, self).__init__(workload=workload, ce_type=ce_type, time_limit=time_limit)

        if expt_config["namespace"] == "history":
            # 不设置namespace
            namespace = None
        else:
            namespace = expt_config["namespace"]

        self.template_explore_expt = forest_exploration.TemplateExplorationExperiment(workload=workload, \
            selected_tables=schema_total, table_num=expt_config["table_num"], namespace=namespace)
        
        forest_config = {
            "selected_tables": schema_total
        } 

        self.forest_explore_expt = forest_exploration.ForestExplorationExperiment(workload=workload, \
            expt_config=forest_config)


    def launch_search_process(self, total_time):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        start_time = time.time()

        # 首先生成sample queries
        if self.expt_config["sample_queries"] == True:
            # 使用自己生成的workload
            config_dummy = {
                "mode": "priori"
            }
            self.template_explore_expt.custom_workload_generation(config=config_dummy)
        else:
            # 
            pass

        # 然后生成templates
        if self.expt_config["generate_template"] == True:
            self.template_explore_expt.construct_spec_template_dict(template_num=10, template_params_list=[])
        else:
            pass

        end_time = time.time()
        print("delta_time = {}".format(end_time - start_time))

        left_time = total_time - (end_time - start_time)

        # 在templates的基础上做探索
        explore_mode = self.expt_config["explore_mode"]

        # 根节点的配置
        root_config = {
            "target": "under", 
            "min_card": 5000, 
            "max_card": 1000000
        }

        # 树的配置
        tree_config = {
            "max_depth":5, 
            "timeout": 60000
        }

        # 搜索过程的配置
        search_config = {
            "max_step": 5,
            "return": "full" # 返回树搜索的所有结果
        }
        # 这里step设置一个很大的值，由left_time控制实验结束
        template_id_list, step = "all", 1000000 
        if explore_mode == "polling_based":
            result = self.forest_explore_expt.polling_based_workload_generation(
                template_id_list=template_id_list, step=step, tree_config=tree_config,
                search_config=search_config, root_config=root_config, total_time=left_time
            )
        elif explore_mode == "epsilon_greedy":
            result = self.forest_explore_expt.Epsilon_Greedy_workload_generation(
                template_id_list=template_id_list, step=step, tree_config=tree_config,
                search_config=search_config, root_config=root_config, total_time=left_time
            )
        elif explore_mode == "correlated_MAB":
            result = self.forest_explore_expt.Correleated_MAB_workload_generation(
                template_id_list=template_id_list, step=step, tree_config=tree_config,
                search_config=search_config, root_config=root_config, total_time=left_time
            )

        self.query_list, self.meta_list, self.result_list, self.card_dict_list = result
        return self.query_list, self.meta_list, self.result_list, self.card_dict_list
        
    def generate_potential_templates(self,):
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

    def explore_on_forests(self,):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """

