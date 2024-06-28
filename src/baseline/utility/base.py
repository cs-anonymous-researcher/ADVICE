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
from data_interaction import mv_management, data_management
from grid_manipulation import grid_preprocess
from query import query_exploration, ce_injection
from plan import node_query, node_extension

# %%

class BasePlanSearcher(object):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, workload = "stats", ce_type = "internal", time_limit = 60000):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        # 输入变量赋值
        self.ce_type = ce_type
        self.time_limit = time_limit
        self.workload = workload

        # 构建QueryInstance和ExtensionInstance所需要的素材
        self.data_manager = data_management.DataManager(wkld_name=workload)
        self.mv_manager = mv_management.MaterializedViewManager(workload=workload)

        self.multi_builder = grid_preprocess.MultiTableBuilder(workload = workload, \
        data_manager_ref = self.data_manager, mv_manager_ref = self.mv_manager)
        self.query_ctrl = query_exploration.QueryController(workload=workload)

        if ce_type.lower() == "internal":
            self.ce_handler = ce_injection.PGInternalHandler(workload=workload)
        elif ce_type.lower() == "deepdb":
            # 默认采用Server模式来实现
            self.ce_handler = ce_injection.DeepDBHandler(workload=workload, mode="server")
            # self.ce_handler = ce_injection.
        elif ce_type.lower() == "neurocard":
            self.ce_handler = ce_injection.NeuroCardHandler(workload = workload)
        elif ce_type.lower() == "oracle":
            self.ce_handler = ce_injection.OracleHandler(workload = workload)
        elif ce_type.lower() == "sqlserver":
            self.ce_handler = ce_injection.SQLServerHandler(workload = workload)
        elif ce_type.lower() == "mscn":
            self.ce_handler = ce_injection.MSCNHandler(workload = workload)
        elif ce_type.lower() == "fcn":
            self.ce_handler = ce_injection.FCNHandler(workload = workload)
        elif ce_type.lower() == "fcnpool":
            self.ce_handler = ce_injection.FCNPoolHandler(workload = workload)
        elif ce_type.lower() == "factorjoin":
            self.ce_handler = ce_injection.FactorJoinHandler(workload = workload)
        elif ce_type.lower() == "deepdb_rdc":
            self.ce_handler = ce_injection.DeepDBRDCHandler(workload = workload)
        elif ce_type.lower() == "deepdb_jct":
            self.ce_handler = ce_injection.DeepDBJCTHandler(workload = workload)
        elif ce_type.lower() == "oceanbase":
            self.ce_handler = ce_injection.OceanbaseHandler(workload = workload)
        else:
            raise ValueError("Unsupported ce_type = {}.".format(ce_type))
        
        # 需要对ce_handler进行初始化，确保
        self.ce_handler.initialization()
        self.query_ctrl = query_exploration.QueryController(workload=workload)

        # 结果保存的容器
        self.query_list, self.meta_list = [], []
        self.result_list, self.card_dict_list = [], []

    def set_properties(self, budget, constriant):
        """
        {Description}
        
        Args:
            budget: 搜索的预算
            constraint: 搜索的限制
        Returns:
            res1:
            res2:
        """
        raise NotImplementedError("launch_search_process has not been implemented!")
    
    def launch_search_process(self, time_total):
        """
        启动搜索的过程

        Args:
            time_total:
        Returns:
            return1:
            return2:
        """
        # raise NotImplementedError("launch_search_process has not been implemented!")
        query_list, meta_list, result_list, card_dict_list = [], [], [], []
        # 默认返回空的结果
        return query_list, meta_list, result_list, card_dict_list
    

    def do_experiment(self, time_total):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        # 启动结果
        query_list, meta_list, result_list, card_dict_list = \
            self.launch_search_process(time_total)
        
        # 保存结果
        return query_list, meta_list, result_list, card_dict_list

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
        raise NotImplementedError("get_search_result has not been implemented!")


    def explore_query_asynchronously(self, query_text, query_meta):
        """
        异步非阻塞的实现查询探索
    
        Args:
            query_text:
            query_meta:
        Returns:
            return1:
            return2:
        """
        external_query = {
            "data_manager": self.data_manager,
            "mv_manager": self.mv_manager,
            "ce_handler": self.ce_handler,
            "query_ctrl": self.query_ctrl,
            "multi_builder": self.multi_builder
        }
        query_instance: node_query.QueryInstance = node_query.get_query_instance(
            workload=self.workload, query_meta=query_meta, external_dict=external_query)

        subquery_dict, single_table_dict = query_instance.estimation_card_dict, query_instance.estimation_single_table

        extension_instance: node_extension.ExtensionInstance = node_extension.get_extension_instance(\
            workload=self.workload, query_text=query_text, query_meta=query_meta, subquery_dict=subquery_dict, 
            single_table_dict=single_table_dict, query_ctrl=self.query_ctrl)

        return extension_instance
    

    def fetch_asynchronous_result(self, query_config):
        """
        获取异步的探索结果
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        pass


    def create_extension_instance(self, query_text, query_meta) -> node_extension.ExtensionInstance:
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        external_query = {
            "data_manager": self.data_manager,
            "mv_manager": self.mv_manager,
            "ce_handler": self.ce_handler,
            "query_ctrl": self.query_ctrl,
            "multi_builder": self.multi_builder
        }

        ce_handler = external_query['ce_handler']
        query_instance: node_query.QueryInstance = node_query.get_query_instance(
            workload=self.workload, query_meta=query_meta, ce_handler=ce_handler, external_dict=external_query)

        subquery_dict, single_table_dict = query_instance.estimation_card_dict, query_instance.estimation_single_table

        extension_instance: node_extension.ExtensionInstance = node_extension.get_extension_instance(\
            workload=self.workload, query_text=query_text, query_meta=query_meta, subquery_dict=subquery_dict, 
            single_table_dict=single_table_dict, query_ctrl=self.query_ctrl)
        
        return extension_instance
    
    def explore_query(self, query_text, query_meta):
        """
        {Description}
        
        Args:
            query_text:
            query_meta:
        Returns:
            result: 探索的结果
            card_dict: 基数字典
        """
        # external_query = {
        #     "data_manager": self.data_manager,
        #     "mv_manager": self.mv_manager,
        #     "ce_handler": self.ce_handler,
        #     "query_ctrl": self.query_ctrl,
        #     "multi_builder": self.multi_builder
        # }
        # query_instance: node_query.QueryInstance = node_query.get_query_instance(
        #     workload=self.workload, query_meta=query_meta, external_dict=external_query)

        # subquery_dict, single_table_dict = query_instance.estimation_card_dict, query_instance.estimation_single_table

        # extension_instance: node_extension.ExtensionInstance = node_extension.get_extension_instance(\
        #     workload=self.workload, query_text=query_text, query_meta=query_meta, subquery_dict=subquery_dict, 
        #     single_table_dict=single_table_dict, query_ctrl=self.query_ctrl)
        extension_instance: node_extension.ExtensionInstance = self.create_extension_instance(query_text, query_meta)
        
        flag, cost1, cost2 = extension_instance.true_card_plan_verification(time_limit=self.time_limit)
        print("flag = {}. cost1 = {}. cost2 = {}.".format(flag, cost1, cost2))

        if flag == False:
            # 表示查询计划不想等
            p_error = cost2 / cost1
        else:
            p_error = 1.0
        
        result = p_error, cost2, cost1  

        card_dict = {
            "true": {
                "subquery": extension_instance.subquery_true,
                "single_table": extension_instance.single_table_true
            },
            "estimation": {
                "subquery": extension_instance.subquery_estimation,
                "single_table": extension_instance.single_table_estimation
            }
        }
        return result, card_dict
    
    def add_exploration_instance(self, query_text, query_meta, result, card_dict):
        """
        添加探索的实例
        
        Args:
            query_text: 
            query_meta: 
            result: 
            card_dict:
        Returns:
            res1:
            res2:
        """

        self.query_list.append(query_text)
        self.meta_list.append(query_meta)
        self.result_list.append(result)
        self.card_dict_list.append(card_dict)

# %%
