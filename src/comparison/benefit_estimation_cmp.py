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
from utility import utils
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, mean_absolute_percentage_error
from estimation import plan_estimation, estimation_interface
from result_analysis import case_analysis
from multiprocessing import Pool

# %%

def get_instance_p_error(input_params):
    workload, query_meta, card_dict = input_params
    analyzer = case_analysis.construct_case_instance(query_meta, card_dict, workload)
    return analyzer.p_error

# %%

class BenefitEstimationComparison(object):
    """
    不同收益估计方法的性能比较

    Members:
        field1:
        field2:
    """

    def __init__(self, workload, data_dir = "/home/lianyuan/Research/GNN_Estimator/test/experiment_test/output"):
        """
        {Description}

        Args:
            workload:
            data_dir:
        """
        self.workload = workload
        self.data_dir = data_dir
        self.result_dict = {}
        self.meta_test, self.card_dict_test = [], []

    def load_result(self, f_name: str = "job_result_dict_internal_baseline.pkl"):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        f_path = p_join(self.data_dir, f_name)
        self.result_dict = utils.load_pickle(f_path)
        return self.result_dict
    
    def append_result(self, f_name: str = "job_result_dict_internal_baseline.pkl"):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        f_path = p_join(self.data_dir, f_name)
        curr_dict: dict = utils.load_pickle(f_path)
        self.result_dict.update(curr_dict)
        return self.result_dict
    

    def load_expt_state(self, f_name: str = "job_expt_state_internal_0.pkl"):
        """
        {Description}
        
        Args:
            f_name:
            arg2:
        Returns:
            res1:
            res2:
        """
        f_path = p_join(self.data_dir, f_name)
        
        # self.meta_list, self.card_dict_list, self.meta_train, self.meta_test, self.card_dict_train, 
        #     self.card_dict_test, self.instance_tuple_test, self.result_dict, self.current_ce_method 

        _, _, _, self.meta_test, _, self.card_dict_test, _, _, _ = utils.load_pickle(f_path)


    def eval_all_cardinality(self, method_list = None):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        result_dict = {}
        for method_name, result_list in self.result_dict.items():
            # 跳过不必要的方法
            if method_list is not None and method_name not in method_list:
                continue

            true_card_list, est_card_list = self.eval_single_cardinality(result_list)
            true_card_arr, est_card_arr = np.array(true_card_list), np.array(est_card_list)
            true_card_arr = true_card_arr + 1.0
            est_card_arr = est_card_arr + 1.0

            true_card_arr, est_card_arr = np.log(true_card_arr), np.log(est_card_arr)
            r2_score_val = r2_score(true_card_arr, est_card_arr)
            MAPE_val = mean_absolute_percentage_error(true_card_arr, est_card_arr)
            print(f"eval_all_cardinality: method_name = {method_name}. r2_score = {r2_score_val:.3f}. MAPE = {MAPE_val:.3f}.")
            result_dict[method_name] = {
                "r2_score": r2_score_val,
                "MAPE_val": MAPE_val
            } 
        return result_dict

    def get_instance_p_error(self, query_meta, card_dict):
        """
        {Description}
        
        Args:
            query_meta:
            card_dict:
        Returns:
            error:
            res2:
        """
        # dummy_query_text = "SELECT 1"
        # dummy_result = (1.0, 1.0, 1.0)
        # analyzer = case_analysis.CaseAnalyzer(dummy_query_text, 
        #     query_meta, dummy_result, card_dict, self.workload)
        analyzer = case_analysis.construct_case_instance(query_meta, card_dict, self.workload)
        return analyzer.p_error

    def eval_all_cost(self, method_list = None):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        result_dict = {}
        for method_name, result_list in self.result_dict.items():
            if method_list is not None and method_name not in method_list:
                continue
            
            time_start = time.time()
            true_cost_list, est_cost_list = self.eval_single_cost(result_list)
            time_end = time.time()
            print(f"eval_all_cost: method_name = {method_name}. delta_time = {time_end - time_start:.2f}")

            true_cost_arr, est_cost_arr = np.array(true_cost_list), np.array(est_cost_list)
            true_cost_arr = true_cost_arr + 1.0
            est_cost_arr = est_cost_arr + 1.0

            # print(f"true_cost_list = {utils.list_round(true_cost_list)}")
            # print(f"est_cost_list = {utils.list_round(est_cost_list)}")

            true_cost_arr, est_cost_arr = np.log(true_cost_arr), np.log(est_cost_arr)
            r2_score_val = r2_score(true_cost_arr, est_cost_arr)
            MAPE_val = mean_absolute_percentage_error(true_cost_arr, est_cost_arr)
            # MAE_val = mean_absolute_error(true_cost_arr, est_cost_arr)

            print(f"eval_all_cost: method_name = {method_name}. r2_score = {r2_score_val:.3f}. MAPE = {MAPE_val:.3f}.")
            result_dict[method_name] = {
                "r2_score": r2_score_val,
                "MAPE": MAPE_val
            }
        return result_dict
    

    def eval_single_cardinality(self, result_list):
        """
        检验单个方法基数估计情况

        Args:
            result_list:
            arg2:
        Returns:
            return1:
            return2:
        """
        true_card_list, est_card_list = [], []

        assert len(result_list) == len(self.card_dict_test), \
            f"eval_single_cardinality: len(result_list) = {len(result_list)}. len(self.card_dict_test) = {len(self.card_dict_test)}."
        
        for ref_card_dict, (query_meta, res_card_dict, subquery_missing, single_table_missing, target_table) in zip(self.card_dict_test, result_list):
            subquery_res, single_table_res, _, _ = \
                utils.extract_card_info(res_card_dict, dict_copy=False)

            subquery_true, single_table_true, _, _ = utils.extract_card_info(ref_card_dict)

            for k in subquery_missing:
                true_card_list.append(subquery_true[k])
                if isinstance(subquery_res[k], (list, tuple)) == True:
                    est_card_list.append(np.average(subquery_res[k]))
                else:
                    est_card_list.append(float(subquery_res[k]))

        assert len(true_card_list) == len(est_card_list)
        return true_card_list, est_card_list

    
    def eval_single_cost(self, result_list):
        """
        单个方法比较对cost的影响
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        true_error_list, est_error_list = [], []
        plan_sample_num = 20

        assert len(self.card_dict_test) == len(result_list), \
            f"eval_single_cost: len(self.card_dict_test) = {len(self.card_dict_test)}. len(result_list) = {len(result_list)}."

        print_flag = True
        for ref_card_dict, (query_meta, res_card_dict, subquery_missing, \
                single_table_missing, target_table) in zip(self.card_dict_test, result_list):
            true_error = self.get_instance_p_error(query_meta, ref_card_dict)
            true_error_list.append(true_error)

            subquery_true, single_table_true, subquery_est, single_table_est = \
                utils.extract_card_info(res_card_dict)
            
            subquery_candidates = utils.dict_subset(subquery_true, 
                lambda a: a in subquery_missing, mode="key")
            single_table_candidates = utils.dict_subset(single_table_true, 
                lambda a: a in single_table_missing, mode="key")

            value_sample_combinations = plan_estimation.make_combinations(subquery_missing, 
                single_table_missing, subquery_candidates, single_table_candidates, plan_sample_num)
            
            #
            card_dict_list = []
            for subquery_cmp, single_table_cmp in value_sample_combinations:
                # 创建新的dict
                local_subquery_dict = utils.dict_merge(subquery_true, subquery_cmp)
                local_single_table_dict = utils.dict_merge(single_table_true, single_table_cmp)

                local_card_dict = utils.pack_card_info(local_subquery_dict, 
                    local_single_table_dict, subquery_est, single_table_est, dict_copy=True)
                
                card_dict_list.append(local_card_dict)

            # 
            if print_flag == True:
                print_flag = False
                print(f"len(card_dict_list) = {len(card_dict_list)}. len(value_sample_combinations) = {len(value_sample_combinations)}.")

            if len(card_dict_list) < 5:
                sample_error_list = [self.get_instance_p_error(query_meta, card_dict) for card_dict in card_dict_list]
            else:
                # 
                params_list = [(self.workload, query_meta, card_dict) for card_dict in card_dict_list]
                with Pool(10) as p:
                    sample_error_list = p.map(get_instance_p_error, params_list)
            
            est_error_list.append(np.average(sample_error_list))

        return true_error_list, est_error_list

# %%
