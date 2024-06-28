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
from utility import utils
from plan import plan_init, node_query, node_extension
from estimation import plan_estimation
from data_interaction import mv_management, data_management
from grid_manipulation import grid_preprocess
from query import query_exploration, ce_injection, query_construction

# %%

class TemplateEstimator(object):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, workload, ce_type = "internal", template_dir = \
        "/home/lianyuan/Research/CE_Evaluator/intermediate/", card_est_input = "graph_corr_based"):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        # 
        self.workload, self.ce_type = workload, ce_type
        self.card_est_input = card_est_input
        self.meta_path = p_join(template_dir, workload, "template_obj/meta_info.json")
        self.load_template_meta(self.meta_path)

        self.ce_handler = ce_injection.get_ce_handler_by_name(workload=workload, ce_type=ce_type)
        self.ce_handler.initialization()

        # 模版和查询的实例
        self.template_instance = None
        self.query_text, self.query_meta = None, None
        self.query_instance, self.extension_instance = None, None

        # 创建node_query需要的基本素材
        self.data_manager = data_management.DataManager(wkld_name=workload)
        self.mv_manager = mv_management.MaterializedViewManager(workload=workload)

        self.multi_builder = grid_preprocess.MultiTableBuilder(workload = workload, \
            data_manager_ref = self.data_manager, mv_manager_ref = self.mv_manager)
        self.query_ctrl = query_exploration.QueryController(workload=workload)


    def load_template_meta(self, meta_path):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        self.meta_dict = utils.load_json(meta_path)
        return self.meta_dict


    def generate_query_instance(self, table_spec = None, column_spec = None, set_query = True):
        """
        生成查询实例(目前还只是随机生成)
    
        Args:
            arg1:
            arg2:
        Returns:
            query_text: 
            query_meta: 
            subquery_dict: 
            single_table_dict:
        """
        query_text, query_meta = self.template_instance.generate_random_query()
        subquery_dict, single_table_dict = \
            self.template_instance.get_plan_cardinalities(in_meta=query_meta, query_text=query_text)
        
        if set_query == True:
            self.query_text, self.query_meta = query_text, query_meta
            query_instance = node_query.get_query_instance(workload=self.workload, \
                query_meta=self.query_meta, external_dict=self.get_external_dict())
            self.query_instance = query_instance
            self.query_instance.add_true_card(subquery_dict, mode="subquery")
            self.query_instance.add_true_card(single_table_dict, mode="single_table")
            # 
            
        return query_text, query_meta, subquery_dict, single_table_dict


    def get_external_dict(self,):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        return {
            "data_manager": self.data_manager,
            "mv_manager": self.mv_manager,
            "ce_handler": self.ce_handler,
            "query_ctrl": self.query_ctrl,
            "multi_builder": self.multi_builder
        }
    
    def explore_single_query(self, query_text, query_meta, subquery_dict, \
                             single_table_dict, config = None, with_verify = True, timeout = 10000):
        """
        探索查询实例
        
        Args:
            query_text: 
            query_meta: 
            subquery_dict: 
            single_table_dict:
        Returns:
            reward: 
            eval_result_list:
        """

        external_query = self.get_external_dict()
        query_instance: node_query.QueryInstance = node_query.get_query_instance(
            workload=self.workload, query_meta=query_meta, external_dict=external_query)

        # print("explore_single_query: subquery_dict = {}. single_table_dict = {}.".\
        #       format(subquery_dict, single_table_dict))
        # 添加真实基数
        query_instance.add_true_card(card_dict=subquery_dict, mode="subquery")
        query_instance.add_true_card(card_dict=single_table_dict, mode="single_table")

        # node_extension.ExtensionInstance()
        # 创建extension
        extension_instance = node_extension.get_extension_from_query_instance(query_instance=query_instance)

        # 创建收益估计器
        estimator_instance = plan_estimation.PlanBenefitEstimator(\
            query_extension=extension_instance, card_est_input = self.card_est_input)
        reward, eval_result_list = estimator_instance.benefit_integration(config = config)

        if with_verify == False:
            return reward, eval_result_list
        else:
            actual_benefit, estimated_benefit, comparison_res_list = \
                estimator_instance.benefit_verification(result_list=eval_result_list, \
                    estimated_benefit=reward, timeout=timeout)
            return actual_benefit, estimated_benefit, comparison_res_list

    def load_template_instance(self, template_id):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        instance_path = self.meta_dict[str(template_id)]["info"]["path"]
        self.template_instance: plan_init.TemplatePlan = utils.load_pickle(instance_path)
        return self.template_instance
    

    def extend_table(self, table_name, column_num = 1):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        new_meta, new_query = None, None
        instance: node_query.QueryInstance = self.query_instance

        column_list = instance.predicate_selection(\
            table_name=table_name, column_num=column_num)
         
        bins_dict, reverse_dict = instance.construct_bins_dict(\
            column_list=column_list, split_budget=100)
        
        column_values = instance.random_on_columns_batch(\
            bins_dict=bins_dict, num=1)[0]
        
        new_meta = instance.add_new_table_meta(self.query_meta, \
                    new_table=table_name, column_range_dict=column_values)
        new_query = query_construction.construct_origin_query(query_meta=new_meta, workload=self.workload)

        return new_query, new_meta
    
    def extend_random_table(self, config = "equal_diff", with_verify = False):
        """
        {Description}
        
        Args:
            config:
            arg2:
        Returns:
            selected_table:
            query_text:
            query_meta:
            result:
        """
        # candidate_tables = self.query_instance.fetch_candidate_tables(\
        #     table_subset = set(self.query_meta[0]))
        candidate_tables = self.query_instance.fetch_candidate_tables()
        
        selected_table = np.random.choice(candidate_tables, 1)[0]
        print("extend_random_table: candidate_tables = {}. selected_table = {}.".\
              format(candidate_tables, selected_table))

        query_text, query_meta = self.extend_table(table_name=selected_table)

        # result = self.explore_single_query(query_text=query_text, query_meta=query_meta, \
        #     subquery_dict=self.query_instance.subquery_true, single_table_dict=self.query_instance.single_table_true)
        result = self.explore_single_query(query_text=query_text, query_meta=query_meta, \
            config=config, with_verify=with_verify, subquery_dict=self.query_instance.true_card_dict, \
            single_table_dict=self.query_instance.true_single_table)

        return selected_table, query_text, query_meta, result

    def evaluate_all_tables(self, ):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        candidate_tables = self.query_instance.fetch_candidate_tables(table_subset = set(self.query_meta[0]))
        result_list = []

        for table in candidate_tables:
            query_meta, query_text = self.extend_table(table_name=table)    # 拓展一张的表

        return result_list
    

    def benefit_comparison(self,):
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
