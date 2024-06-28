#!/usr/bin/env python
# -*- coding: utf-8 -*-


# %%
from evaluation import baseline
from evaluation.baseline import equal_diff_strategy, equal_ratio_strategy, graph_strategy_dict
from utilities import utils, framework_test

import numpy as np
import pandas as pd
from os.path import join as p_join
from matplotlib import pyplot as plt
import json, os, time, re, math
import pickle
from copy import deepcopy
from interaction import state_management

# %%

def remove_dict_null(in_dict: dict):
    # 消除dict中value为null的项
    out_dict = {k: v for k, v in in_dict.items() if v is not None}
    return out_dict

# %%

class ExperimentController(object):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, workload, data_dir = "/home/lianyuan/Research/GNN_Estimator/data/simulation"):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        self.workload = workload
        self.data_dir = data_dir
        self.meta_list, self.card_dict_list = [], []
        self.meta_train, self.meta_test = [], []
        self.card_dict_train, self.card_dict_test = [], []
        self.instance_tuple_test = []       # 用于测试的实例
        self.current_ce_method = "undefined"

        self.result_dict = {}               # 不同方法的结果保存
        self.method_list = ["equal_diff", "equal_ratio", "kde", "max", "gat", "gcn", "dropout_bayes"]

    def dump_expt_state(self, signature, out_dir = "./output"):
        """
        导出当前的实验状态
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        out_name = f"{self.workload}_expt_state_{signature}.pkl"
        out_path = p_join(out_dir, out_name)

        # 中间状态全部合并保存在一个元组里
        utils.dump_pickle((self.meta_list, self.card_dict_list, self.meta_train, 
            self.meta_test, self.card_dict_train, self.card_dict_test, self.instance_tuple_test, 
            self.result_dict, self.current_ce_method), out_path)


    def load_expt_state(self, signature, out_dir = "./output"):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        out_name = f"{self.workload}_expt_state_{signature}.pkl"
        out_path = p_join(out_dir, out_name)

        self.meta_list, self.card_dict_list, self.meta_train, self.meta_test, self.card_dict_train, \
            self.card_dict_test, self.instance_tuple_test, self.result_dict, self.current_ce_method = utils.load_pickle(out_path)

    def load_data(self, ce_method):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        self.current_ce_method = ce_method
        result_name = f"{self.workload}_{ce_method}_out.pkl"
        result_path = p_join(self.data_dir, self.workload, result_name)
        self.meta_list, self.card_dict_list = utils.load_pickle(result_path)

    def construct_test_card_mask(self, mask_num_info = 0.3):
        """
        制作测试cardinality的mask
    
        Args:
            mask_num_info:
            arg2:
        Returns:
            instance_tuple_test:
            return2:
        """
        # 清空数据
        self.instance_tuple_test = [] 

        for query_meta, card_dict in zip(self.meta_test, self.card_dict_test):
            out_card_dict, subquery_missing, single_table_missing, target_table = \
                framework_test.card_dict_mask_along_table(self.workload, card_dict, mask_num_info)
            self.instance_tuple_test.append((out_card_dict, 
                subquery_missing, single_table_missing, target_table))

        assert len(self.instance_tuple_test) == len(self.meta_test) == len(self.card_dict_test)
        return self.instance_tuple_test

    def split_train_test_data(self, split_ratio = 0.5):
        """
        {Description}
    
        Args:
            split_ratio:
            arg2:
        Returns:
            meta_train:
            meta_test:
            card_dict_train:
            card_dict_test:
        """
        assert len(self.meta_list) == len(self.card_dict_list)
        assert len(self.meta_list) > 0

        train_num = int(len(self.meta_list) * split_ratio)
        test_num = len(self.meta_list) - train_num
        meta_list, card_dict_list = self.meta_list, self.card_dict_list

        # 按照时间划分结果
        self.meta_train, self.meta_test = meta_list[:train_num], meta_list[train_num: train_num + test_num]
        self.card_dict_train, self.card_dict_test = \
            card_dict_list[:train_num], card_dict_list[train_num: train_num + test_num]

        assert len(self.meta_train) == len(self.card_dict_train)
        assert len(self.meta_test) == len(self.card_dict_test)

        return self.meta_train, self.meta_test, self.card_dict_train, self.card_dict_test

    def execute_estimation(self, method):
        """
        {Description}
        method_list = ["equal_diff", "equal_ratio", "kde", "max", "gat", "gcn", "dropout_bayes"]
        
        Args:
            method:
            arg2:
        Returns:
            result_list:
            return2:
        """
        assert method in self.method_list

        if method in ("equal_diff", "equal_ratio", "kde", "max"):
            # 启发式的方法
            return self.execute_heuristic_estimation(method)
        elif method in ("gat", "gcn", "dropout_bayes"):
            # 学习型的方法
            return self.execute_learned_estimation(method)
        else:
            raise ValueError(f"execute_estimation: invalid method = {method}.")
        

    def execute_heuristic_estimation(self, method):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        if method == "equal_diff":
            estimator = baseline.BuiltinEstimator(self.workload, equal_diff_strategy)
        elif method == "equal_ratio":
            estimator = baseline.BuiltinEstimator(self.workload, equal_ratio_strategy)
        elif method == "kde":
            estimator = baseline.GraphCorrBasedEstimator(self.workload, graph_strategy_dict["kde"])
        elif method == "max":
            estimator = baseline.GraphCorrBasedEstimator(self.workload, graph_strategy_dict["max"])

        dummy_text = "SELECT COUNT(*) FROM 1;"
        assert len(self.meta_test) ==  len(self.instance_tuple_test)
        result_list = []

        for query_meta, (out_card_dict, subquery_missing, single_table_missing, target_table) in \
                zip(self.meta_test, self.instance_tuple_test):
            estimator.set_instance(dummy_text, query_meta)
            # 添加结果到list
            subquery_true, single_table_true, subquery_est, single_table_est = \
                utils.extract_card_info(out_card_dict)

            # remove subquery_true和single_table_true中的None信息
            subquery_true, single_table_true = remove_dict_null(subquery_true), remove_dict_null(single_table_true)
            estimator.set_existing_card_dict(subquery_true, single_table_true, subquery_est, single_table_est)

            subquery_candidates, single_table_candidates = \
                estimator.make_value_sampling(subquery_missing, single_table_missing)

            subquery_new = utils.dict_merge(subquery_true, subquery_candidates)
            single_table_new = utils.dict_merge(single_table_true, single_table_candidates)
            res_card_dict = utils.pack_card_info(subquery_new, single_table_new, 
                subquery_est, single_table_est, dict_copy=True)
            result_list.append((query_meta, res_card_dict, subquery_missing, single_table_missing, target_table))
        
        self.result_dict[method] = result_list
        return result_list

    def execute_learned_estimation(self, method):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        signature = f"{self.workload}_{self.current_ce_method}_test"
        if method == "gcn":
            curr_manager = state_management.StateManager(self.workload, 
                self.current_ce_method, signature, model_type="gcn", auto_update=False)
        elif method == "gat":
            curr_manager = state_management.StateManager(self.workload, 
                self.current_ce_method, signature, model_type="gat", auto_update=False)
        elif method == "dropout_bayes":
            curr_manager = state_management.StateManager(self.workload, 
                self.current_ce_method, signature, model_type="dropout_bayes", auto_update=False)

        curr_manager.set_sample_num(new_num=5)
        curr_manager.clean_directory()
        # 添加训练数据
        curr_manager.add_train_data(self.meta_train, self.card_dict_train)
        # curr_manager.add_train_data(self.meta_list, self.card_dict_list)
        # 训练图神经网络模型
        curr_manager.model_retrain(lock=state_management.retrain_lock)
        # 重新加载模型
        curr_manager.model_reload()

        result_list = []

        print_flag = False
        for query_meta, (out_card_dict, subquery_missing, single_table_missing, target_table) in \
                zip(self.meta_test, self.instance_tuple_test):
            # subquery_candidates, single_table_candidates = \
            #     estimator.make_value_sampling(subquery_missing, single_table_missing)
            # # 添加结果到list
            subquery_true, single_table_true, subquery_est, single_table_est = \
                utils.extract_card_info(out_card_dict)
            # 在实例上进行推断
            flag, subquery_res, single_table_res = curr_manager.infer_on_instance(query_meta, out_card_dict, reload = False)
            if print_flag == False:
                print(f"execute_learned_estimation: subquery_res = {subquery_res}. single_table_res = {single_table_res}.")
                print_flag = True
            assert flag == True, "execute_learned_estimation: infer_on_instance. flag = False"
            subquery_new = utils.dict_merge(subquery_true, subquery_res)
            single_table_new = utils.dict_merge(single_table_true, single_table_res)
            res_card_dict = utils.pack_card_info(subquery_new, single_table_new, 
                subquery_est, single_table_est, dict_copy=True)
            result_list.append((query_meta, res_card_dict, subquery_missing, single_table_missing, target_table))

        self.result_dict[method] = result_list
        return result_list

    def execute_estimation_with_uncertainty(self, method):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        # signature = f"{self.workload}_{self.current_ce_method}_test"
        signature = f"{self.workload}_{self.current_ce_method}_uncertainty"

        curr_manager = state_management.StateManager(self.workload, 
            self.current_ce_method, signature, model_type="dropout_bayes", auto_update=False)

        curr_manager.set_sample_num(new_num=5)
        curr_manager.clean_directory()
        # 添加训练数据
        curr_manager.add_train_data(self.meta_train, self.card_dict_train)
        # 训练图神经网络模型
        curr_manager.model_retrain(lock=state_management.retrain_lock)
        # 重新加载模型
        curr_manager.model_reload()

        result_list = []

        print_flag = False
        for query_meta, (out_card_dict, subquery_missing, single_table_missing, target_table) in \
                zip(self.meta_test, self.instance_tuple_test):
            # 添加结果到list
            subquery_true, single_table_true, subquery_est, single_table_est = \
                utils.extract_card_info(out_card_dict)
            
            # 在实例上进行推断，并且考虑
            flag, subquery_res, single_table_res = curr_manager.infer_on_instance_with_uncertainty(query_meta, out_card_dict, reload = False)
            
            # if print_flag == False:
            #     print(f"execute_learned_estimation_with_uncertainty: subquery_res = {subquery_res}. single_table_res = {single_table_res}.")
            #     print_flag = True
            assert flag == True, "execute_learned_estimation: infer_on_instance. flag = False"
            subquery_new = utils.dict_merge(subquery_true, subquery_res)
            single_table_new = utils.dict_merge(single_table_true, single_table_res)
            res_card_dict = utils.pack_card_info(subquery_new, single_table_new, 
                subquery_est, single_table_est, dict_copy=True)
            result_list.append((query_meta, res_card_dict, subquery_missing, single_table_missing, target_table))

        self.result_dict[method] = result_list
        return result_list


    def dump_result(self, signature, out_dir = "./output"):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        out_name = f"{self.workload}_result_dict_{signature}.pkl"
        out_path = p_join(out_dir, out_name)
        utils.dump_pickle(self.result_dict, out_path)
    

# %%
