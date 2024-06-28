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

from plan import node_query, node_utils, node_extension
from collections import defaultdict
from itertools import product
from workload import physical_plan_info
from utility import utils
from estimation import builtin_card_estimation, external_card_estimation

from functools import reduce
from operator import mul
# %%
from utility.common_config import default_strategy, \
    equal_diff_strategy, equal_ratio_strategy, option_collections
# %%

def make_combinations(subquery_key_list, single_table_key_list, \
        subquery_candidates, single_table_candidates, plan_sample_num = 100):
    """
    {Description}

    Args:
        subquery_key_list:
        single_table_key_list:
        subquery_candidates:
        single_table_candidates:
        plan_sample_num:
    Returns:
        value_sample_combinations:
        return2:
    """
    # print("make_combinations: subquery_key_list = {}. single_table_key_list = {}.".\
    #       format(subquery_key_list, single_table_key_list))
    # print("make_combinations: subquery_candidates = {}. single_table_candidates = {}.".\
    #       format(subquery_candidates, single_table_candidates))

    value_sample_combinations = []
    all_values_list = []
    for key in subquery_key_list:
        if isinstance(subquery_candidates[key], int):
            all_values_list.append(tuple((subquery_candidates[key],)))
        else:
            all_values_list.append(tuple(subquery_candidates[key]))

    for key in single_table_key_list:
        try:
            if isinstance(subquery_candidates[key], int):
                all_values_list.append(tuple((single_table_candidates[key],)))
            else:
                all_values_list.append(tuple(single_table_candidates[key]))
        except KeyError as e:
            print(f"make_combinations: meet KeyError. key = {key}. single_table_candidates = {single_table_candidates.keys()}.")
            raise e

    candidate_size = reduce(mul, [len(i) for i in all_values_list], 1)
    enumeration_threshold = 1e4

    # print(f"make_combinations: candidate_size = {candidate_size}. enumeration_threshold = {enumeration_threshold}.")
    if candidate_size < enumeration_threshold:
        # 
        all_combinations_list = list(product(*all_values_list))     # 
        # print("make_combinations: len(all_combinations_list) = {}.".format(len(all_combinations_list)))
        # print("make_combinations: sample_combination = {}".format(all_combinations_list[0]))
        # 感觉数目还是有点多，考虑采样上的优化
        if len(all_combinations_list) >= plan_sample_num:
            selected_idx = np.random.choice(list(range(len(\
                all_combinations_list))), size=plan_sample_num, replace=False)
        else:
            selected_idx = range(0, len(all_combinations_list))

        selected_combinations = utils.list_index(all_combinations_list, selected_idx)
    else:
        # 
        selected_combinations = set()
        max_try_times, curr_cnt = 10000, 0
        while True:
            curr_cnt += 1
            if curr_cnt > max_try_times:
                raise ValueError(f"make_combinations: Exceed max_try_times! curr_cnt = {curr_cnt}")
            
            local_list = []
            for local_values in all_values_list:
                # print(f"make_combinations: local_values = {local_values}")
                local_list.append(np.random.choice(local_values))

            selected_combinations.add(tuple(local_list))
            if len(selected_combinations) >= plan_sample_num:
                break
        selected_combinations = list(selected_combinations)

    # for value_list in all_combinations_list:
    for value_list in selected_combinations:
        subquery_cmp, single_table_cmp = {}, {}
        for idx, value in enumerate(value_list):
            if idx < len(subquery_key_list):
                # print("make_combinations: subquery_cmp = {}. subquery_key_list = {}. idx = {}.".\
                #       format(subquery_cmp, subquery_key_list, idx))
                subquery_cmp[subquery_key_list[idx]] = value
            else:
                shift_idx = idx - len(subquery_key_list)
                single_table_cmp[single_table_key_list[shift_idx]] = value
        value_sample_combinations.append((subquery_cmp, single_table_cmp))

    # print("make_combinations: len(value_sample_combinations) = {len(value_sample_combinations)}."
    return value_sample_combinations

# %%

class PlanBenefitEstimator(object):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, query_extension: node_extension.ExtensionInstance, \
                 card_est_input: str | dict, mode = "under-estimation", target_table = None):
        """
        {Description}

        Args:
            query_extension: 查询拓展的实例
            card_estimator: 外部的估计器
        """
        self.workload = query_extension.workload
        self.query_extension = query_extension
        self.subquery_true = query_extension.subquery_true
        self.single_table_true = query_extension.single_table_true
        self.subquery_estimation = query_extension.subquery_estimation
        self.single_table_estimation = query_extension.single_table_estimation

        # self.external_estimator = external_estimator
        # self.strategy = default_strategy
        self.mode, self.target_table = mode, target_table
        self.card_estimator = self.construct_estimator_instance(card_est_input)
    
    def mask_target_table(self, table):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        alias = self.query_extension.query_parser.inverse_alias[table]
        subquery_keys = list(self.subquery_true.keys())

        for k in subquery_keys:
            if alias in k:
                try:
                    del self.subquery_true[k]
                except KeyError:
                    continue
        
        try:
            del self.single_table_true[alias]
        except KeyError:
            pass

        return self.subquery_true.keys(), self.single_table_true.keys()

    def construct_estimator_instance(self, card_est_input) -> builtin_card_estimation.BaseEstimator:
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        if isinstance(card_est_input, str):
            card_est_input = option_collections[card_est_input]
        elif not isinstance(card_est_input, dict):
            raise TypeError(f"construct_estimator_instance: unsupported card_est_input type({type(card_est_input)})")
        
        est_str = card_est_input['estimator']

        if est_str == "built-in":
            card_estimator = builtin_card_estimation.BuiltinEstimator(\
                workload=self.workload, strategy=card_est_input['strategy'])
        elif est_str == "graph_corr":
            card_estimator = builtin_card_estimation.GraphCorrBasedEstimator(\
                workload=self.workload, strategy=card_est_input['strategy'])
        elif est_str == "external":
            card_estimator = external_card_estimation.MixEstimator(workload=self.workload, \
                model_type=card_est_input["model_type"], internal_type = card_est_input['internal'])
        else:
            raise ValueError(f"construct_estimator_instance: est_str = {est_str}. "\
                    "valid_list = ['built-in', 'graph_corr_based', 'external']")
        return card_estimator

    def infer_missing_cardinalities(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        return 


    def subquery_relation_eval(self,):
        """
        分析子查询之间的关系，建立图来表示
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        pass
        

    def benefit_integration(self, plan_sample_num = 100, ):
        """
        通过积分的方式来求解预期的收益情况
        需要尽可能的优化效率

        Args:
            config: 收益积分的相关配置
            arg2:
        Returns:
            reward:
            eval_result_list:
        """
        eval_result_list, cost_pair_list = self.apply_card_sample(plan_sample_num)
        reward = self.calculate_reward(cost_pair_list=cost_pair_list)
        return reward, eval_result_list
    
    @utils.timing_decorator
    def cost_pair_integration(self, config = None, plan_sample_num = 100, restrict_order = True):
        """
        获得平均的true_cost和estimation_cost
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        eval_result_list, cost_pair_list = \
            self.apply_card_sample(plan_sample_num, restrict_order)
        avg_cost_pair = self.calculate_cost_pair(cost_pair_list)
        return avg_cost_pair, eval_result_list


    def generate_card_combinations(self, plan_sample_num):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        subquery_missing, single_table_missing = self.query_extension.get_external_missing_card()
        subquery_key_missing, single_table_key_missing = \
            list(subquery_missing.keys()), list(single_table_missing.keys())
        
        ext = self.query_extension
        self.card_estimator.set_instance(ext.query_text, ext.query_meta)
        self.card_estimator.set_existing_card_dict(ext.subquery_true, \
            ext.single_table_true, ext.subquery_estimation, ext.single_table_estimation)

        # 
        subquery_candidates, single_table_candidates = self.card_estimator.make_value_sampling(\
            subquery_missing=subquery_key_missing, single_table_missing=single_table_key_missing)

        value_sample_combinations = self.make_combinations(subquery_key_list = subquery_key_missing, \
                single_table_key_list = single_table_key_missing, subquery_candidates = subquery_candidates, \
                single_table_candidates = single_table_candidates, plan_sample_num = plan_sample_num)
        
        return value_sample_combinations
    

    def generate_card_sample(self, plan_sample_num = 100):
        """
        {Description}
    
        Args:
            plan_sample_num:
            arg2:
        Returns:
            card_dict_list:
            return2:
        """
        card_dict_list = []
        value_sample_combinations = self.generate_card_combinations(plan_sample_num)

        #
        for subquery_cmp, single_table_cmp in value_sample_combinations:
            origin_subquery_dict = self.subquery_true
            origin_single_table_dict = self.single_table_true

            # 创建新的dict
            local_subquery_dict = node_utils.dict_merge(old_dict=\
                origin_subquery_dict, new_dict=subquery_cmp)
            local_single_table_dict = node_utils.dict_merge(old_dict=\
                origin_single_table_dict, new_dict=single_table_cmp)

            local_card_dict = utils.pack_card_info(local_subquery_dict, local_single_table_dict, 
                self.subquery_estimation, self.single_table_estimation, dict_copy=True)
            
            card_dict_list.append(local_card_dict)
        # 
        return card_dict_list
    

    @utils.timing_decorator
    def apply_card_sample(self, plan_sample_num = 100, restrict_order = True):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        # start_time = time.time()
        # 
        value_sample_combinations = self.generate_card_combinations(plan_sample_num)

        # 2024-03-10: 对比实验确认新方法的有效性和正确性
        # ts_1 = time.time()
        # eval_result_list1 = self.card_dict_complement_batch(origin_subquery_dict=self.subquery_true, 
        #     origin_single_table_dict=self.single_table_true, value_sample_combinations=value_sample_combinations)
        # te_1 = time.time()

        ts_2 = time.time()
        eval_result_list2 = self.card_dict_complement_parallel(self.subquery_true, 
            self.single_table_true, value_sample_combinations, restrict_order)
        te_2 = time.time()

        # def cmp_local(item1, item2):
        #     # print(f"0: {item1[0] == item2[0]}. 1: {item1[1] == item2[1]}. 2: {item1[2] == item2[2]}. "\
        #     #       f"3: {item1[3] == item2[3]}. 4: {item1[4] == item2[4]}. 5: {item1[5] == item2[5]}.")
        #     if item1[0: 3] == item2[0: 3] and item1[5:] == item2[5:]:
        #         return True
        #     else:
        #         return False
            
        # for idx, (item1, item2) in enumerate(zip(eval_result_list1, eval_result_list2)):
        #     # if item1 != item2:
        #     if cmp_local(item1, item2) == False:
        #         print(f"apply_card_sample: result unmatch. idx = {idx}. \nitem1 = {item1}. \nitem2 = {item2}.")
        #         raise ValueError("apply_card_sample: card_dict_complement result unmatch.")

        # print(f"apply_card_sample: query_table_num = {len(self.query_extension.query_meta[0])}. "\
        #       f"plan_sample_num = {plan_sample_num}. delta_time1 = {1000 * (te_1 - ts_1): .2f}ms. "\
        #       f"delta_time2 = {1000 * (te_2 - ts_2): .2f}ms.")
        
        eval_result_list = eval_result_list2
        # print("benefit_integration: eval_result_list = {}.".format(eval_result_list))
        # end_time = time.time()

        cost_pair_list = [(item[1], item[2]) for item in eval_result_list]
        # print("apply_card_sample: delta time = {}. query_table_num = {}. plan_sample_num = {}.".\
        #       format(end_time - start_time, len(self.query_extension.query_meta[0]), plan_sample_num))
        return eval_result_list, cost_pair_list
    

    @utils.timing_decorator
    def calculate_reward(self, cost_pair_list, mode = "error-oriented"):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        # print("calculate_reward: cost_pair_list = {}.".format(cost_pair_list))
        reward = 0.0

        if mode == "error-oriented":
            error_list= [item[1]/item[0] for item in cost_pair_list]
            reward = np.average(error_list)
        elif mode == "cost-oriented":
            total_cost1 = np.sum([item[0] for item in cost_pair_list])
            total_cost2 = np.sum([item[1] for item in cost_pair_list])
            reward = total_cost2 / total_cost1

        return reward

    # @utils.timing_decorator
    def calculate_cost_pair(self, cost_pair_list):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        true_cost_list, estimation_cost_list = zip(*cost_pair_list)
        return np.average(true_cost_list), np.average(estimation_cost_list)
    
    @utils.timing_decorator
    def benefit_verification(self, result_list, estimated_benefit, timeout):
        """
        {Description}
        
        Args:
            result_list: 
            estimated_benefit: 
            timeout:
        Returns:
            actual_benefit: 
            estimated_benefit: 
            comparison_res_list:
        """
        flag, cost_true, cost_estimation, plan_true, plan_estimation = \
            self.query_extension.true_card_plan_comparison(time_limit=timeout)
        actual_benefit = cost_estimation / cost_true
        print("estimated_benefit = {}. actual_benefit = {}.".format(estimated_benefit, actual_benefit))
        
        comparison_res_list = []

        # 进行结果匹配
        for (flag, cost1, cost2, plan1, plan2, subquery_cmp, single_table_cmp) in result_list:
            # 比较基数dict的相似度
            card_cmp_res = self.card_dict_comparison(subquery_true=\
                self.query_extension.subquery_true, subquery_cmp=subquery_cmp)

            # 比较查询计划的一致性
            plan_cmp_res = physical_plan_info.physical_comparison(physical_plan1=plan1, physical_plan2=plan_true)

            # 代价的比较
            cost_cmp_res = cost1, cost2, cost_true, cost_estimation
            comparison_res_list.append((card_cmp_res, plan_cmp_res, cost_cmp_res))
        
        return actual_benefit, estimated_benefit, comparison_res_list

    @utils.timing_decorator
    def card_dict_comparison(self, subquery_true: dict, subquery_cmp: dict):
        """
        {Description}
        
        Args:
            subquery_true:
            subquery_cmp:
        Returns:
            res1:
            res2:
        """
        subquery_out = {}

        for key in subquery_cmp.keys():
            val1, val2 = subquery_true[key], subquery_cmp[key]
            curr_q_error = max(val1 / val2, val2 / val1)
            subquery_out[key] = (val1, val2, curr_q_error)

        return subquery_out


    # @utils.timing_decorator
    def make_combinations(self, subquery_key_list, single_table_key_list, \
            subquery_candidates, single_table_candidates, plan_sample_num = 100):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        # print("make_combinations: subquery_key_list = {}. single_table_key_list = {}.".\
        #       format(subquery_key_list, single_table_key_list))
        # print("make_combinations: subquery_candidates = {}. single_table_candidates = {}.".\
        #       format(subquery_candidates, single_table_candidates))

        value_sample_combinations = []
        all_values_list = []
        for key in subquery_key_list:
            all_values_list.append(tuple(subquery_candidates[key]))

        for key in single_table_key_list:
            try:
                all_values_list.append(tuple(single_table_candidates[key]))
            except KeyError as e:
                print(f"make_combinations: meet KeyError. key = {key}. single_table_candidates = {single_table_candidates.keys()}.")
                raise e

        candidate_size = reduce(mul, [len(i) for i in all_values_list], 1)
        enumeration_threshold = 1e4

        # print(f"make_combinations: candidate_size = {candidate_size}. enumeration_threshold = {enumeration_threshold}.")
        if candidate_size < enumeration_threshold:
            # 
            all_combinations_list = list(product(*all_values_list))     # 
            # print("make_combinations: len(all_combinations_list) = {}.".format(len(all_combinations_list)))
            # print("make_combinations: sample_combination = {}".format(all_combinations_list[0]))
            # 感觉数目还是有点多，考虑采样上的优化
            if len(all_combinations_list) >= plan_sample_num:
                selected_idx = np.random.choice(list(range(len(\
                    all_combinations_list))), size=plan_sample_num, replace=False)
            else:
                selected_idx = range(0, len(all_combinations_list))

            selected_combinations = utils.list_index(all_combinations_list, selected_idx)
        else:
            # 
            selected_combinations = set()
            max_try_times, curr_cnt = 10000, 0
            while True:
                curr_cnt += 1
                if curr_cnt > max_try_times:
                    raise ValueError(f"make_combinations: Exceed max_try_times! curr_cnt = {curr_cnt}")
                
                local_list = []
                for local_values in all_values_list:
                    # print(f"make_combinations: local_values = {local_values}")
                    local_list.append(np.random.choice(local_values))

                selected_combinations.add(tuple(local_list))
                if len(selected_combinations) >= plan_sample_num:
                    break


            selected_combinations = list(selected_combinations)
        # for value_list in all_combinations_list:
        for value_list in selected_combinations:
            subquery_cmp, single_table_cmp = {}, {}
            for idx, value in enumerate(value_list):
                if idx < len(subquery_key_list):
                    # print("make_combinations: subquery_cmp = {}. subquery_key_list = {}. idx = {}.".\
                    #       format(subquery_cmp, subquery_key_list, idx))
                    subquery_cmp[subquery_key_list[idx]] = value
                else:
                    shift_idx = idx - len(subquery_key_list)
                    single_table_cmp[single_table_key_list[shift_idx]] = value
            value_sample_combinations.append((subquery_cmp, single_table_cmp))
    
        # print("make_combinations: len(value_sample_combinations) = {}.".\
        #       format(len(value_sample_combinations)))
        # import sys
        # sys.exit(0)
        return value_sample_combinations


    def card_dict_complement_parallel(self, origin_subquery_dict, \
            origin_single_table_dict, value_sample_combinations, restrict_order = True):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        subquery_dict_list, single_table_dict_list = [], []
        for subquery_cmp, single_table_cmp in value_sample_combinations:
            local_subquery_dict = node_utils.dict_merge(old_dict=\
                origin_subquery_dict, new_dict=subquery_cmp)
            local_single_table_dict = node_utils.dict_merge(old_dict=\
                origin_single_table_dict, new_dict=single_table_cmp)
            
            subquery_dict_list.append(local_subquery_dict)
            single_table_dict_list.append(local_single_table_dict)

        mode = self.mode
        # if mode == "under-estimation":
        #     eval_result_list = []
        # elif mode == "over-estimation":
        #     eval_result_list = []

        # 2024-03-09: 将two_plan改成multi_plan
        # 每一项的内容为(plan_flag, table_flag, cost1, cost2, plan1, plan2)
        out_list = self.query_extension.multi_plan_verification_under_constraint(                        
            subquery_dict_list1=subquery_dict_list, single_table_dict_list1=single_table_dict_list, 
            subquery_dict2=self.subquery_estimation, single_table_dict2=self.single_table_estimation,
            keyword1="mixed", keyword2="estimation", target_table=self.target_table, return_plan=True)

        def proc_result(item1, item2):
            plan_flag, table_flag, cost1, cost2, plan1, plan2 = item1
            subquery_cmp, single_table_cmp = item2
            if table_flag == False and mode == "over-estimation" and restrict_order == True:
                cost1 = -cost2

            return plan_flag, cost1, cost2, plan1, plan2, subquery_cmp, single_table_cmp
        
        assert len(out_list) == len(value_sample_combinations), \
            f"card_dict_complement_parallel: out_list = {len(out_list)}. value_sample_combinations = {len(value_sample_combinations)}."
        eval_result_list = [proc_result(item1, item2) for item1, item2 in zip(out_list, value_sample_combinations)]
        return eval_result_list


    # @utils.timing_decorator
    def card_dict_complement_batch(self, origin_subquery_dict, \
        origin_single_table_dict, value_sample_combinations):
        """
        cardinality dict补全以后生成结果的列表，每一项包含的元素为:
        flag, cost1, cost2, plan1, plan2, subquery_cmp, single_table_cmp
    
        Args:
            arg1:
            arg2:
        Returns:
            eval_result_list:
            return2:
        """
        eval_result_list = []

        for subquery_cmp, single_table_cmp in value_sample_combinations:
            local_subquery_dict = node_utils.dict_merge(old_dict=\
                origin_subquery_dict, new_dict=subquery_cmp)
            local_single_table_dict = node_utils.dict_merge(old_dict=\
                origin_single_table_dict, new_dict=single_table_cmp)

            mode = self.mode
            if mode == "under-estimation":
                flag, cost1, cost2, plan1, plan2 = self.query_extension.two_plan_comparison(
                    subquery_dict1=local_subquery_dict, single_table_dict1=local_single_table_dict, 
                    subquery_dict2=self.subquery_estimation, single_table_dict2=self.single_table_estimation,
                    keyword1="mixed", keyword2="estimation"
                )
            elif mode == "over-estimation":
                plan_flag, table_flag, cost1, cost2, plan1, plan2 = \
                    self.query_extension.two_plan_verification_under_constraint(
                        subquery_dict1=local_subquery_dict, single_table_dict1=local_single_table_dict, 
                        subquery_dict2=self.subquery_estimation, single_table_dict2=self.single_table_estimation,
                        keyword1="mixed", keyword2="estimation", target_table=self.target_table, return_plan=True)
                
                # flag = (plan_flag and table_flag)
                flag = plan_flag

                # if table_flag == False:
                #     # 为了适应API，调整cost，直接设置成负的
                #     cost1 = -cost2

            # 把未知的基数以及查询计划，补充也加进去
            eval_result_list.append((flag, cost1, cost2, plan1, plan2, subquery_cmp, single_table_cmp))
        
        return eval_result_list
    
# %%
