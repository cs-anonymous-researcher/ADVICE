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
from itertools import combinations, permutations

# %%
from result_analysis import case_analysis
from estimation import case_based_estimation
import graphviz
from collections import Counter
from query import query_construction
from functools import reduce
from utility import utils, common_config, workload_spec
from data_interaction import mv_management
# from experiment import stateful_exploration
from experiment import root_evaluation
from pprint import pprint
from workload import physical_plan_info
from estimation import estimation_verification

# %%

class SingleState(object):
    """
    状态实例

    Members:
        field1:
        field2:
    """

    def __init__(self, schema_order, workload, mode, manager_ref):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        assert mode in ("over-estimation", "under-estimation"), f"SingleState.__init__: mode = {mode}."
        self.schema_order = schema_order
        self.last_table = schema_order[-1]
        self.mode, self.workload = mode, workload
        self.estimator = case_based_estimation.CaseBasedEstimator(workload)
        self.instance_list = []

        self.status_list = []
        self.global_status = "invalid_extend"
        self.manager_ref = manager_ref

        self.index_mapping = {}     # 全局index到selected index的映射
        self.signature_set = set()  # 表示已经保存的signature


    def update_reference_result(self,):
        """
        更新参照的结果
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        pass


    def get_state_summary(self,):
        """
        获得状态的概要信息
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        p_error_list = [item['p_error'] for item in self.instance_list if \
            item['p_error'] is not None and item['reference_only'] == False]

        cnt = Counter(self.status_list)

        if len(p_error_list) != 0:
            max_error, min_error, median_error = \
                np.max(p_error_list), np.min(p_error_list), np.median(p_error_list)
        else:
            max_error, min_error, median_error = 0.0, 0.0, 0.0

        alias_mapping = query_construction.abbr_option[self.workload]

        node_label = f"schema_order: {[alias_mapping[s] for s in self.schema_order]}\n"\
            f"max_err={max_error:.2f}. min_err={min_error:.2f}. median_err={median_error:.2f}\n"\
            f"invalid_num={cnt['invalid_extend']}. valid_num={cnt['valid_extend']}. effective_num={cnt['effective_extend']}."

        return node_label
    
    def add_new_instance(self, query_meta, card_dict, valid, true_plan = None, \
            est_plan = None, true_cost = None, est_cost = None, p_error = None, reference_only = False):
        """
        {Description}

        Args:
            query_meta:
            card_dict:
            valid: 该实例是否有效，即生成的误差来源符合预期
            true_plan:
            est_plan:
            true_cost:
            est_cost:
            p_error:
            reference_only: 是否只用来参考，不涉及状态本身评价
        Returns:
            return1:
            return2:
        """
        # 2024-03-20: 判断case是否已存在
        meta_signature = mv_management.meta_key_repr(query_meta, self.workload)

        if meta_signature in self.signature_set and reference_only == True:
            # print(f"add_new_instance: meta_signature = {meta_signature}. has been in self.signature_set.")
            return
        # else:
        #     print(f"add_new_instance: meta_signature = {meta_signature}. has not been in self.signature_set.")
        
        instance_dict = {
            "query_meta": query_meta,
            "card_dict": card_dict,
            "valid": valid,
            "true_plan": true_plan, "est_plan": est_plan,
            "true_cost": true_cost, "est_cost": est_cost,
            "p_error": p_error,
            "reference_only": reference_only
        }

        self.instance_list.append(instance_dict)
        # self.estimator.add_new_case(query_meta, card_dict)

        if reference_only == False:
            curr_status = self.determine_status(valid, p_error)
            self.status_list.append(curr_status)
            self.update_global_status(curr_status)

        self.signature_set.add(meta_signature)
    

    # def select_reference_cases(self, max_num = None, external_index_list = None):
    #     """
    #     {Description}
    
    #     Args:
    #         max_num:
    #         external_index_list: 
    #     Returns:
    #         return1:
    #         return2:
    #     """
    #     # selected_list = []
    #     # item = (query_meta, card_dict)
    #     valid_list = [] 

    #     assert self.global_status in ("invalid_extend", "valid_extend", "effective_extend")
    #     skip_cnt = 0
    #     for instance_dict in self.instance_list:
    #         if self.global_status == "invalid_extend":
    #             # 考虑选取所有的case
    #             valid_list.append((instance_dict['query_meta'], instance_dict['card_dict']))
    #         else:
    #             # 只选取card_dict完整的case
    #             if instance_dict['true_plan'] is None:
    #                 # print(f"select_reference_cases: global_status = {self.global_status}. true_plan is None. skip instance.")
    #                 skip_cnt += 1
    #                 continue
    #             valid_list.append((instance_dict['query_meta'], instance_dict['card_dict']))
        
    #     if valid_list == 0:
    #         # 确认所有case
    #         assert skip_cnt == len(self.instance_list)

    #     if max_num is None or max_num >= len(valid_list):
    #         # 先实现最简单的版本
    #         index_selected = np.arange(len(valid_list))
    #         # selected_list = valid_list
    #     else:
    #         index_all = np.arange(len(valid_list))
    #         index_selected = np.random.choice(index_all, max_num, replace=False)
    #         # selected_list = utils.list_index(valid_list, index_selected)

    #     selected_list = utils.list_index(valid_list, index_selected)
    #     self.index_mapping = {local_idx: global_idx for local_idx, global_idx in enumerate(index_selected)}
    #     self.estimator.instance_list = selected_list

    #     return selected_list

    def select_reference_cases(self, max_num = None, external_index_list = None):
        """
        {Description}
    
        Args:
            max_num: 最多的reference_cases数目
            external_index_list: 外部的index_list引用
        Returns:
            return1:
            return2:
        """
        # selected_list = []
        # item = (query_meta, card_dict)
        valid_index_list = [] 

        assert self.global_status in ("invalid_extend", "valid_extend", "effective_extend")
        skip_cnt = 0
        for idx, instance_dict in enumerate(self.instance_list):
            if self.global_status == "invalid_extend":
                # 考虑选取所有的case
                valid_index_list.append(idx)
            else:
                # 只选取card_dict完整的case
                if instance_dict['true_plan'] is None:
                    # print(f"select_reference_cases: global_status = {self.global_status}. true_plan is None. skip instance.")
                    skip_cnt += 1
                    continue
                valid_index_list.append(idx)


        if len(valid_index_list) == 0:
            # 确认所有case
            assert skip_cnt == len(self.instance_list)

        if external_index_list is not None:
            # 针对external_index_list的额外处理
            if max_num is not None:
                max_num -= len(external_index_list)
            valid_index_list = list(set(valid_index_list).difference(external_index_list))
            # print("")

        if max_num is None or max_num >= len(valid_index_list) or len(valid_index_list) == 0:
            # 先实现最简单的版本
            index_selected = list(valid_index_list)
            # selected_list = valid_list
        else:
            # index_all = np.arange(len(valid_list))
            try:
                if max_num > 0:
                    sample_res = np.random.choice(valid_index_list, max_num, replace=False)
                    index_selected = list(sample_res)
                else:
                    index_selected = []
            except Exception as e:
                print(f"select_reference_cases: meet Error. max_num = {max_num}. valid_index_list = {valid_index_list}. index_selected = {index_selected}.")
                raise e

            # selected_list = utils.list_index(valid_list, index_selected)

        if external_index_list is not None:
            index_selected = index_selected + external_index_list
            # print(f"select_reference_cases: valid_index_list = {valid_index_list}. "\
            #       f"index_selected = {index_selected}. external_index_list = {external_index_list}. ")

        selected_instance = utils.list_index(self.instance_list, index_selected)
        selected_list = [(item['query_meta'], item['card_dict']) for item in selected_instance]
        self.index_mapping = {local_idx: global_idx for local_idx, global_idx in enumerate(index_selected)}
        self.estimator.instance_list = selected_list

        return selected_list

    def is_reference_only(self, local_idx):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        global_idx = self.index_mapping[local_idx]
        return self.instance_list[global_idx]['reference_only']

    def get_max_p_error(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        p_error_list = [item["p_error"] for item in self.instance_list if \
            item["p_error"] is not None and item["reference_only"] == False]
        try:
            return max(p_error_list)
        except ValueError as e:
            return 0.0

    def determine_status(self, valid, p_error):
        """
        确定当前实例的状态，当前可选的状态
        ("invalid_extend", "valid_extend", "effective_extend")
    
        Args:
            valid:
            p_error:
        Returns:
            return1:
            return2:
        """
        if valid == False:
            return "invalid_extend"
        elif p_error is None or p_error < 1.0 + 1e-7:
            return "valid_extend"
        else:
            return "effective_extend"


    def update_global_status(self, new_status):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        state_level = {
            "invalid_extend": 0,
            "valid_extend": 1,
            "effective_extend": 2
        }
        assert new_status in state_level.keys(), f"update_global_status: new_status = {new_status}. available_list = {state_level.keys()}"

        global_level, curr_level = \
            state_level[self.global_status], state_level[new_status]
        
        if curr_level > global_level:
            self.global_status = new_status          

    def join_order_matching(self, jo_list1, jo_list2, skip_num = 3):
        """
        判断连接顺序是否匹配，这里的skip_num需要动态调整
        
        Args:
            jo_list1:
            jo_list2:
            skip_num:
        Returns:
            return1:
            return2:
        """
        flag1 = set(jo_list1[:skip_num]) == set(jo_list2[:skip_num])
        flag2 = jo_list1[skip_num:] == jo_list2[skip_num:]
        return flag1 and flag2
    
    def prefix_matching(self, jo_list1, jo_list2, prefix_num):
        """
        验证prefix对应的schema set是否对应
    
        Args:
            arg1:
            arg2:
        Returns:
            flag:
            return2:
        """
        flag = set(jo_list1[:prefix_num]) == set(jo_list2[:prefix_num])
        return flag
    
    def suffix_matching(self, jo_list1, jo_list2, prefix_num):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            flag:
            return2:
        """
        flag = jo_list1[prefix_num:] == jo_list2[prefix_num:]
        return flag
    
    def suffix_key(self, schema_key, jo_list, prefix_num):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        return tuple(list(schema_key[:prefix_num]) + list(jo_list[prefix_num:]))

    
    @utils.timing_decorator
    def analyze_results(self, result_list, prefix_num = 3):
        """
        分析估计结果，抽取当前新条件带来的关键增益
        额外加入对FuzzyCaseAnalyzer的处理
    
        Args:
            result_list: case_based估计结果，item = (query_meta, card_dict)
            prefix_num:
        Returns:
            candidate_list: 候选的case匹配列表，tuple = node_type, (query_meta, card_dict), p_error
            return2:
        """
        assert self.mode in ("over-estimation", "under-estimation")

        max_p_error = self.get_max_p_error()
        candidate_list = []         # 候选数据实例列表
        valid_index_list, valid_index_local = [], []       # 候选数据参照的对应索引
        alias_inverse = workload_spec.get_alias_inverse(self.workload)

        def card_dict_eval(card_dict):
            # 2024-03-11: 判断card_dict中是否存在tuple/list的项，True代表包含，False代表不包含
            subquery_true, single_table_true, subquery_est, \
                single_table_est = utils.extract_card_info(card_dict)

            num_func = lambda in_dict: sum([isinstance(a, (tuple, list)) for a in in_dict.values()])
            if num_func(subquery_true) + num_func(subquery_est) + \
                num_func(single_table_true) + num_func(single_table_est) > 0:
                # utils.display_card_dict(card_dict, "analyze_results.card_dict_eval")
                return True
            else:
                return False
        
        def proc_analyzer(in_analyzer: case_analysis.CaseAnalyzer, index: int):
            nonlocal candidate_list
            query_meta, card_dict = in_analyzer.meta, in_analyzer.card_dict
            curr_error = in_analyzer.p_error

            # print(f"analyze_results.proc_analyzer: global_status = {self.global_status}. "\
            #       f"max_error = {max_p_error: .2f}. curr_error = {curr_error: .2f}.")

            if self.mode == "under-estimation":
                flag, jo_list = in_analyzer.get_plan_join_order(mode="estimation")
                # jo_match_res = self.join_order_matching(jo_list, self.schema_order, skip_num=common_config.template_table_num)
            elif self.mode == "over-estimation":
                flag, jo_list = in_analyzer.get_plan_join_order(mode="true")

            jo_list = tuple([alias_inverse[alias] for alias in jo_list])    # 由alias转成schema
            jo_match_res = self.join_order_matching(jo_list, self.schema_order, skip_num=common_config.template_table_num)
            if flag == False:
                # print(f"analyze_results.proc_analyzer: jo_list = {jo_list}. return directly")
                return 
            
            if self.global_status == "invalid_extend":
                # 分析是否可能出现valid_extend
                # flag, jo_list = in_analyzer.get_plan_join_order(mode="estimation")
                # if flag == False:
                #     return
                # jo_match_res = self.join_order_matching(jo_list, self.schema_order)
                # if (self.mode == "under-estimation" and jo_match_res == True) or \
                #    (self.mode == "over-estimation" and jo_match_res == False):
                #     # 合法的结果
                #     candidate_list.append(("invalid_extend", (query_meta, card_dict), 0.0))                
                if jo_match_res == True:
                    candidate_list.append(("invalid_extend", (query_meta, card_dict), 0.0, self.schema_order))
                    valid_index_local.append(index)
            elif self.global_status == "valid_extend":
                # 分析是否可能出现effective_extend
                if curr_error > 1.0 + 1e-6:
                    # 2024-03-17: 考虑添加结果
                    # 
                    prefix_flag = self.prefix_matching(jo_list, self.schema_order, prefix_num)
                    suffix_flag = self.suffix_matching(jo_list, self.schema_order, prefix_num)

                    # 暂时生成warning，再进行处理
                    if prefix_flag == False:
                        # print(f"proc_analyzer: global_status = {self.global_status}. error = {curr_error:.2f}. prefix_flag is False. \n"\
                        #       f"expected_prefix = {self.schema_order[:prefix_num]}. actual_prefix = {jo_list[:prefix_num]}.")
                        # 2024-03-17: 暂不处理
                        pass
                    elif suffix_flag == False:
                        # print(f"proc_analyzer: global_status = {self.global_status}. error = {curr_error:.2f}. suffix_flag is False. \n"\
                        #       f"expected_prefix = {self.schema_order[prefix_num:]}. actual_prefix = {jo_list[prefix_num:]}.")
                        # 2024-03-17: 暂不处理
                        pass
                    elif prefix_flag == True and suffix_flag == True:
                        # print(f"proc_analyzer: global_status = {self.global_status}. error = {curr_error:.2f}. prefix_flag and suffix_flag is True")
                        candidate_list.append(("valid_extend", (query_meta, card_dict), curr_error, self.schema_order))
                        valid_index_local.append(index)
                    else:
                        raise ValueError(f"proc_analyzer: global_status = {self.global_status}. prefix_flag = {prefix_flag}. suffix_flag = {suffix_flag}.")
            elif self.global_status == "effective_extend":
                # 分析p_error
                if curr_error > max(max_p_error * 0.5, 2.0):
                    # if estimation_verification.save_estimation == True:
                    #     print(f"analyze_results: global_verifier.add_new_instance. error = {in_analyzer.p_error: .2f}.")
                    #     estimation_verification.global_verifier.add_new_instance(in_analyzer.query, \
                    #         in_analyzer.meta, in_analyzer.card_dict, prefix_num, in_analyzer.p_error)

                    # 2024-03-17: 考虑添加结果
                    prefix_flag = self.prefix_matching(jo_list, self.schema_order, prefix_num)
                    suffix_flag = self.suffix_matching(jo_list, self.schema_order, prefix_num)

                    # 暂时生成warning，再进行处理
                    if prefix_flag == False:
                        # 前缀不合法
                        # print(f"proc_analyzer: global_status = {self.global_status}. error = {curr_error:.2f}. prefix_flag is False. \n"\
                        #       f"expected_prefix = {self.schema_order[:prefix_num]}. actual_prefix = {jo_list[:prefix_num]}.")
                        # 加到待定dict中作为创建新的template的依据
                        # schema_order, query_meta, card_dict, valid, true_plan = None, \
                        #     est_plan = None, true_cost = None, est_cost = None, p_error = None, is_real = False
                        # 2024-03-20: 更严格的参考条件
                        if curr_error > max(max_p_error * 0.75, 2.0):
                            template_id, state_key = self.manager_ref.add_new_instance_external(jo_list, query_meta, card_dict, False, in_analyzer.plan_true, 
                                in_analyzer.plan_estimation, in_analyzer.true_cost, in_analyzer.estimation_cost, curr_error, False)
                            # 2024-03-20:考虑将match的case迁移到其他模版中
                            global_idx = self.index_mapping[index]
                            if template_id is not None: 
                                if self.is_reference_only(index) == False:
                                    # 将实例转移
                                    # transfer_case(self, src_tmpl_id, src_key, src_index, dst_tmpl_id, dst_key):
                                    self.manager_ref.explorer_ref.transfer_case(self.manager_ref.template_id, self.schema_order, global_idx, template_id, state_key)
                            else:
                                # 2024-03-21: 同样考虑添加到未来的state_manager中
                                self.manager_ref.explorer_ref.transfer_case(self.manager_ref.template_id, self.schema_order, global_idx, None, state_key)
                        else:
                            pass
                            # print(f"add_new_instance_external: curr_error = {curr_error:.2f}. max_p_error = {max_p_error:.2f}.")
                    elif suffix_flag == False:
                        # 后缀不合法
                        # print(f"proc_analyzer: global_status = {self.global_status}. error = {curr_error:.2f}. suffix_flag is False. \n"\
                        #       f"expected_prefix = {self.schema_order[prefix_num:]}. actual_prefix = {jo_list[prefix_num:]}.")   
                        target_key = self.suffix_key(self.schema_order, jo_list, prefix_num)
                        candidate_list.append(("effective_extend", (query_meta, card_dict), curr_error, target_key))
                        # valid_index_list.append(index)
                        # 2024-03-20: 考虑将match的case迁移到其他模版中
                        if self.is_reference_only(index) == False:
                            # 将实例转移
                            global_idx = self.index_mapping[index]
                            self.manager_ref.explorer_ref.transfer_case(self.manager_ref.template_id, 
                                self.schema_order, global_idx, self.manager_ref.template_id, target_key)

                    elif prefix_flag == True and suffix_flag == True:
                        # 前/后缀均合法
                        # print(f"proc_analyzer: global_status = {self.global_status}. error = {curr_error:.2f}. prefix_flag and suffix_flag is True")
                        # if estimation_verification.save_estimation == True:
                        #     print(f"analyze_results: global_verifier.add_new_instance. error = {in_analyzer.p_error: .2f}.")
                        #     estimation_verification.global_verifier.add_new_instance(in_analyzer.query, \
                        #         in_analyzer.meta, in_analyzer.card_dict, prefix_num, in_analyzer.p_error)
                        candidate_list.append(("effective_extend", (query_meta, card_dict), curr_error, self.schema_order))
                        valid_index_local.append(index)
                    else:
                        raise ValueError(f"proc_analyzer: global_status = {self.global_status}. prefix_flag = {prefix_flag}. suffix_flag = {suffix_flag}.")
                    
            else:
                raise ValueError(f"analyze_results.proc_analyzer: global_status = {self.global_status}")

        sample_num, scale = common_config.global_sample_num, common_config.global_scale
        for (query_meta, card_dict), idx in result_list:
            if card_dict_eval(card_dict) == True:
                # print(f"SingleState.analyze_results: card_dict_eval = True. sample_num = {sample_num}. scale = {scale}.")
                fuzzy_analyzer = case_analysis.FuzzyCaseAnalyzer("", query_meta, (), card_dict, self.workload)
                analyzer_list = fuzzy_analyzer.sample_on_card_dict(out_case_num=common_config.out_case_num) 
                for a in analyzer_list:
                    assert card_dict_eval(a.card_dict) == False
                    proc_analyzer(a, idx)
            else:
                # print(f"SingleState.analyze_results: card_dict_eval = False. sample_num = {sample_num}. scale = {scale}.")
                local_analyzer = case_analysis.CaseAnalyzer(\
                    "", query_meta, (), card_dict, self.workload)
                proc_analyzer(local_analyzer, idx)

        valid_index_list = [self.index_mapping[idx] for idx in valid_index_local]
        return candidate_list, valid_index_list

    
    @utils.timing_decorator
    def aggregate_results(self, result_list, prefix_num = 3, add_external_threshold = None, effective_extend_threshold = None):
        """
        分析估计结果，用于，抽取当前新条件带来的关键增益
        额外加入对FuzzyCaseAnalyzer的处理
    
        Args:
            result_list: case_based估计结果，item = (query_meta, card_dict)
            prefix_num:
        Returns:
            candidate_list: 候选的case匹配列表，tuple = node_type, (query_meta, card_dict), p_error
            return2:
        """
        assert self.mode in ("over-estimation", "under-estimation")

        max_p_error = self.get_max_p_error()
        candidate_list = []         # 候选数据实例列表
        valid_index_list, valid_index_local = [], []       # 候选数据参照的对应索引
        alias_inverse = workload_spec.get_alias_inverse(self.workload)

        if add_external_threshold is None:
            add_external_threshold = max(max_p_error * 0.75, 2.0)

        if effective_extend_threshold is None:
            effective_extend_threshold = max(max_p_error * 0.5, 2.0)

        def card_dict_eval(card_dict):
            # 2024-03-11: 判断card_dict中是否存在tuple/list的项，True代表包含，False代表不包含
            subquery_true, single_table_true, subquery_est, \
                single_table_est = utils.extract_card_info(card_dict)

            num_func = lambda in_dict: sum([isinstance(a, (tuple, list)) for a in in_dict.values()])
            if num_func(subquery_true) + num_func(subquery_est) + \
                num_func(single_table_true) + num_func(single_table_est) > 0:
                return True
            else:
                return False
        
        def proc_analyzer(in_analyzer: case_analysis.CaseAnalyzer, index: int):
            nonlocal candidate_list
            query_meta, card_dict = in_analyzer.meta, in_analyzer.card_dict
            curr_error = in_analyzer.p_error

            # print(f"aggregate_results.proc_analyzer: global_status = {self.global_status}. "\
            #       f"max_error = {max_p_error: .2f}. curr_error = {curr_error: .2f}.")

            if self.mode == "under-estimation":
                flag, jo_list = in_analyzer.get_plan_join_order(mode="estimation")
            elif self.mode == "over-estimation":
                flag, jo_list = in_analyzer.get_plan_join_order(mode="true")

            jo_list = tuple([alias_inverse[alias] for alias in jo_list])    # 由alias转成schema
            jo_match_res = self.join_order_matching(jo_list, self.schema_order, skip_num=common_config.template_table_num)
            if flag == False:
                print(f"aggregate_results.proc_analyzer: jo_list = {jo_list}. return directly")
                return 
            
            if self.global_status == "invalid_extend":
                # 分析是否可能出现valid_extend           
                if jo_match_res == True:
                    candidate_list.append(("invalid_extend", (query_meta, card_dict), 0.0, self.schema_order))
                    valid_index_local.append(index)
            elif self.global_status == "valid_extend":
                # 分析是否可能出现effective_extend
                if curr_error > 1.0 + 1e-6:
                    # 2024-03-17: 考虑添加结果
                    # 
                    prefix_flag = self.prefix_matching(jo_list, self.schema_order, prefix_num)
                    suffix_flag = self.suffix_matching(jo_list, self.schema_order, prefix_num)

                    # 暂时生成warning，再进行处理
                    if prefix_flag == False:
                        # print(f"aggregate_results.proc_analyzer: global_status = {self.global_status}. error = {curr_error:.2f}. prefix_flag is False. \n"\
                        #       f"expected_prefix = {self.schema_order[:prefix_num]}. actual_prefix = {jo_list[:prefix_num]}.")
                        # 2024-03-17: 暂不处理
                        pass    
                    elif suffix_flag == False:
                        # print(f"aggregate_results.proc_analyzer: global_status = {self.global_status}. error = {curr_error:.2f}. suffix_flag is False. \n"\
                        #       f"expected_prefix = {self.schema_order[prefix_num:]}. actual_prefix = {jo_list[prefix_num:]}.")
                        # 2024-03-17: 暂不处理
                        pass
                    elif prefix_flag == True and suffix_flag == True:
                        # print(f"aggregate_results.proc_analyzer: global_status = {self.global_status}. error = {curr_error:.2f}. prefix_flag and suffix_flag is True")
                        candidate_list.append(("valid_extend", (query_meta, card_dict), curr_error, self.schema_order))
                        valid_index_local.append(index)
                    else:
                        raise ValueError(f"proc_analyzer: global_status = {self.global_status}. prefix_flag = {prefix_flag}. suffix_flag = {suffix_flag}.")
            elif self.global_status == "effective_extend":
                # 分析p_error
                # if curr_error > max_p_error * 0.5:
                if curr_error > effective_extend_threshold:
                    # 2024-03-17: 考虑添加结果
                    # 
                    prefix_flag = self.prefix_matching(jo_list, self.schema_order, prefix_num)
                    suffix_flag = self.suffix_matching(jo_list, self.schema_order, prefix_num)

                    if prefix_flag == False or suffix_flag == False:
                        # 结果不及预期时，将ref_jo_list拿出来
                        global_idx = self.index_mapping[index]
                        ref_case_dict = self.instance_list[global_idx]
                        if self.mode == "over-estimation":
                            target_plan: physical_plan_info.PhysicalPlan = ref_case_dict['true_plan']
                        elif self.mode == "under-estimation":
                            target_plan: physical_plan_info.PhysicalPlan = ref_case_dict['est_plan']
                        is_bushy, ref_jo_list = target_plan.get_join_order_info()
                        ref_jo_list = workload_spec.list_alias_to_schema(ref_jo_list, self.workload)
                        if is_bushy:
                            print("aggregate_results.proc_analyzer: warning! reference case is bushy.")
                    else:
                        ref_jo_list = []


                    # 暂时生成warning，再进行处理
                    if prefix_flag == False:
                        # 前缀不合法
                        # print(f"aggregate_results.proc_analyzer: global_status = {self.global_status}. error = {curr_error:.2f}. prefix_flag is False. \n"\
                        #       f"expected_prefix = {self.schema_order[:prefix_num]}. actual_prefix = {jo_list[:prefix_num]}. ref_case_prefix = {ref_jo_list[:prefix_num]}.")
                        
                        # 加到待定dict中作为创建新的template的依据
                        # schema_order, query_meta, card_dict, valid, true_plan = None, \
                        #     est_plan = None, true_cost = None, est_cost = None, p_error = None, is_real = False
                        # 2024-03-20: 更严格的参考条件
                        # if curr_error > max(max_p_error * 0.75, 2.0):
                        if curr_error > add_external_threshold:
                            template_id, state_key = self.manager_ref.add_new_instance_external(jo_list, query_meta, card_dict, False, in_analyzer.plan_true, 
                                in_analyzer.plan_estimation, in_analyzer.true_cost, in_analyzer.estimation_cost, curr_error, False)
                            # print(f"aggregate_results.proc_analyzer: add_new_instance_external. template_id = {template_id}. state_key = {state_key}.")
                            # 2024-03-20:考虑将match的case迁移到其他模版中
                            global_idx = self.index_mapping[index]
                            if template_id is not None: 
                                if self.is_reference_only(index) == False:
                                    # 将实例转移
                                    # transfer_case(self, src_tmpl_id, src_key, src_index, dst_tmpl_id, dst_key):
                                    self.manager_ref.explorer_ref.transfer_case(self.manager_ref.template_id, self.schema_order, global_idx, template_id, state_key)
                            else:
                                # 2024-03-21: 同样考虑添加到未来的state_manager中
                                self.manager_ref.explorer_ref.transfer_case(self.manager_ref.template_id, self.schema_order, global_idx, None, state_key)
                        else:
                            pass
                            # print(f"add_new_instance_external: curr_error = {curr_error:.2f}. max_p_error = {max_p_error:.2f}.")
                    elif suffix_flag == False:
                        # 后缀不合法
                        # print(f"aggregate_results.proc_analyzer: global_status = {self.global_status}. error = {curr_error:.2f}. suffix_flag is False. \n"\
                        #       f"expected_suffix = {self.schema_order[prefix_num:]}. actual_suffix = {jo_list[prefix_num:]}. ref_case_suffix = {ref_jo_list[prefix_num:]}.")
                         
                        target_key = self.suffix_key(self.schema_order, jo_list, prefix_num)
                        candidate_list.append(("effective_extend", (query_meta, card_dict), curr_error, target_key))
                        # valid_index_list.append(index)
                        # 2024-03-20: 考虑将match的case迁移到其他模版中
                        if self.is_reference_only(index) == False:
                            # 将实例转移
                            global_idx = self.index_mapping[index]
                            self.manager_ref.explorer_ref.transfer_case(self.manager_ref.template_id, 
                                self.schema_order, global_idx, self.manager_ref.template_id, target_key)

                    elif prefix_flag == True and suffix_flag == True:
                        # 前/后缀均合法
                        if estimation_verification.save_estimation == True:
                            print(f"aggregate_results: global_verifier.add_new_instance. error = {in_analyzer.p_error: .2f}.")
                            estimation_verification.global_verifier.add_new_instance(in_analyzer.query, \
                                in_analyzer.meta, in_analyzer.card_dict, prefix_num, in_analyzer.p_error)
                        # print(f"proc_analyzer: global_status = {self.global_status}. error = {curr_error:.2f}. prefix_flag and suffix_flag is True")
                        candidate_list.append(("effective_extend", (query_meta, card_dict), curr_error, self.schema_order))
                        valid_index_local.append(index)
                    else:
                        raise ValueError(f"proc_analyzer: global_status = {self.global_status}. prefix_flag = {prefix_flag}. suffix_flag = {suffix_flag}.")
                else:
                    # print(f"aggregate_result.proc_analyzer: effective_extend fails. max_p_error = {max_p_error:.2f}. curr_error = {curr_error:.2f}")
                    pass
            else:
                raise ValueError(f"analyze_results.proc_analyzer: global_status = {self.global_status}")

        sample_num, scale = common_config.global_sample_num, common_config.global_scale
        for (query_meta, card_dict), idx in result_list:
            if card_dict_eval(card_dict) == True:
                # print(f"SingleState.analyze_results: card_dict_eval = True. sample_num = {sample_num}. scale = {scale}.")
                fuzzy_analyzer = case_analysis.FuzzyCaseAnalyzer("", query_meta, (), card_dict, self.workload)
                analyzer_list = fuzzy_analyzer.sample_on_card_dict(out_case_num=common_config.out_case_num) 
                for a in analyzer_list:
                    assert card_dict_eval(a.card_dict) == False
                    proc_analyzer(a, idx)
            else:
                # print(f"SingleState.analyze_results: card_dict_eval = False. sample_num = {sample_num}. scale = {scale}.")
                local_analyzer = case_analysis.CaseAnalyzer(\
                    "", query_meta, (), card_dict, self.workload)
                proc_analyzer(local_analyzer, idx)

        valid_index_list = [self.index_mapping[idx] for idx in valid_index_local]
        return candidate_list, valid_index_list


    def infer_new_condition(self, new_meta, true_card = None, est_card = None):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        self.select_reference_cases(common_config.reference_num)
        result_list = self.estimator.eval_new_condition(new_meta, true_card, est_card)
        # return self.analyze_results(result_list)
        candidate_list, valid_index_list = self.analyze_results(result_list)
        return candidate_list
    

    def infer_new_init_query(self, init_meta, card_dict, external_index_list = None, ref_benefit = None):
        """
        估计new_init的收益
        
        Args:
            init_meta:
            card_dict:
            external_index_list:
            ref_benefit:
        Returns:
            candidate_list: 
            valid_index_list:
        """
        # if external_index_list is None:
        self.select_reference_cases(common_config.reference_num, external_index_list)
            
        # 2024-03-16: 根据global_status选择不同的mode
        if self.global_status == "invalid_extend":
            result_list = self.estimator.eval_new_init(init_meta, card_dict, mode="estimation")
        else:
            result_list = self.estimator.eval_new_init(init_meta, card_dict, mode="both")
        
        prefix_num = len(init_meta[0])
        # print(f"infer_new_init_query: result_list = {len(result_list)}.")

        # 2024-03-31: 根据external_index_list调整aggregation的策略
        if external_index_list is None:
            # print("infer_new_init_query: external_index_list is None. call analyze_results.")
            candidate_list, valid_index_list = self.analyze_results(result_list, prefix_num)
        else:
            # print("infer_new_init_query: external_index_list is not None. call aggregate_results.")
            candidate_list, valid_index_list = self.aggregate_results(
                result_list, prefix_num, effective_extend_threshold = ref_benefit)
        # print(f"infer_new_init_query: max_error = {self.get_max_p_error(): .2f}. \ncandidate_list = {candidate_list}.")

        return candidate_list, valid_index_list

# %%

class StateManager(object):
    """
    状态实例的管理者

    Members:
        field1:
        field2:
    """

    # def __init__(self, workload, mode, template_id = -1, explorer_ref: stateful_exploration.StatefulExploration = None):
    def __init__(self, workload, mode, template_id = -1, explorer_ref = None):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        self.workload = workload
        self.mode = mode
        self.template_id = template_id
        self.state_dict = {}
        self.edge_list = []

        self.explorer_ref = explorer_ref
        self.case_matcher = root_evaluation.ExternalCaseMatcher(self.explorer_ref.get_template_by_id(template_id))

    def add_ref_case(self, query_meta, card_dict, p_error):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        self.case_matcher.add_new_case(query_meta, card_dict, p_error)

    def add_new_instance_external(self, schema_order, query_meta, card_dict, valid, true_plan = None, \
            est_plan = None, true_cost = None, est_cost = None, p_error = None, is_real = False):
        """
        将new_instance加到external的StateManager中去
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        template_id, state_key = self.explorer_ref.assign_new_instance(schema_order, query_meta, \
            card_dict, valid, true_plan, est_plan, true_cost, est_cost, p_error, is_real, self.mode)

        return template_id, state_key

    def get_exploration_history(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        state_num = len(self.state_dict)
        instance_num_list = [len(state.instance_list) for state in self.state_dict.values()]
        p_error_list = [state.get_max_p_error() for state in self.state_dict.values()]

        try:
            max_p_error = max(p_error_list)
        except ValueError as e:
            # print(f"get_exploration_history: meet ValueError. tmpl_id = {self.template_id}. instance_num_list = {instance_num_list}. p_error_list = {utils.list_round(p_error_list, 2)}")
            max_p_error = 0.5

        res_dict =  {
            "max_p_error": max_p_error,
            "instance_num": sum(instance_num_list),
            "state_num": state_num
        }
        return res_dict

    def add_manager_ref(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        for schema_order, v in self.state_dict.items():
            print(f"add_manager_ref: schema_order = {schema_order}.")
            state_dict: SingleState = v
            state_dict.manager_ref = self

        return self.state_dict

    def create_new_single_state(self, schema_order):
        """
        {Description}

        Args:
            single_state:
            arg2:
        Returns:
            return1:
            return2:
        """
        new_single_state = SingleState(schema_order, self.workload, self.mode, manager_ref = self)
        self.state_dict[tuple(schema_order)] = new_single_state

        return self.state_dict
    
    def show_available_states(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        schema_order_list = sorted(self.state_dict.keys())
        print("show_available_states:")
        for schema_order in schema_order_list:
            # max_error = self.state_dict[schema_order].get_max_p_error()
            single_state: SingleState = self.state_dict[schema_order]
            max_error = single_state.get_max_p_error()
            status = single_state.global_status
            print(f"{schema_order} max_error = {max_error:.2f}. global_status = {status}.")


    def construct_state_tree(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        self.complement_root()

        def valid_edge(schema_order1, schema_order2):
            if schema_order1 == schema_order2[:-1]:
                return True
            else:
                return False

        self.edge_list = []     # 每次清空edge_list，然后重建
        # for k1, k2 in combinations(self.state_dict.keys(), 2):
        for k1, k2 in permutations(self.state_dict.keys(), 2):
            if valid_edge(k1, k2) == True:
                self.edge_list.append((str(k1), str(k2)))

        return self.edge_list
    
    def complement_root(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        def find_common_prefix(list1, list2):
            common_prefix = []
            min_len = min(len(list1), len(list2))
            
            for i in range(min_len):
                if list1[i] == list2[i]:
                    common_prefix.append(list1[i])
                else:
                    break
            
            return common_prefix
        
        schema_order_list = list(self.state_dict.keys())
        root_order = reduce(find_common_prefix, schema_order_list[1:], schema_order_list[0])

        self.create_new_single_state(root_order)

        # # 添加由root延展的边
        # for 


    def plot_state_tree(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        self.construct_state_tree()

        state_graph = graphviz.Graph()
        for k, v in sorted(self.state_dict.items()):
            v: SingleState = v
            try:
                state_graph.node(name=str(k), label=v.get_state_summary())
            except TypeError as e:
                print(f"plot_state_tree: name = {k}")
                print(f"plot_state_tree: label = {v.get_state_summary()}")
                raise

        for s_node, e_node in self.edge_list:
            state_graph.edge(e_node, s_node)
        
        return state_graph

    def add_new_instance(self, schema_order, query_meta, card_dict, valid, true_plan = None, \
            est_plan = None, true_cost = None, est_cost = None, p_error = None, reference_only = False):
        """
        添加新的匹配实例
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        if valid == False and true_cost is not None:
            if schema_order is not None:
                # 2024-03-12: over-estimation并且invalid的情况，另外schema_order不能为空(这里表示bushy的情况)
                valid = True
                self.add_new_instance_external(schema_order, query_meta, card_dict, valid, 
                    true_plan, est_plan, true_cost, est_cost, p_error, reference_only)
            else:
                # 2024-03-13: 需要被忽略的状态
                pass
        else:
            if schema_order not in self.state_dict:
                self.create_new_single_state(schema_order)

            local_single_state: SingleState = self.state_dict[tuple(schema_order)]
            local_single_state.add_new_instance(query_meta, card_dict, valid, \
                true_plan, est_plan, true_cost, est_cost, p_error, reference_only)

    def add_instance_list(self, instance_list):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        for item in instance_list:
            self.add_new_instance(**item)

    def infer_new_condition_benefit(self, new_meta, true_card = None, est_card = None):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            tree_result:
            res2:
        """
        tree_result = {}
        
        for key, single_state in self.state_dict.items():
            single_state: SingleState = single_state
            candidate_list = single_state.infer_new_condition(\
                new_meta, true_card, est_card)

            tree_result[key] = candidate_list
        return tree_result
    
    def infer_new_init_benefit(self, init_meta, card_dict):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            tree_result: key为schema_order，value为
            return2:
        """
        tree_result = {}
        
        # 2024-03-21: 处理RuntimeError: dictionary changed size during iteration
        # for key, single_state in self.state_dict.items():
        #     single_state: SingleState = single_state
        key_list = list(self.state_dict.keys())
        error_list = [self.state_dict[key].get_max_p_error() for key in key_list]
        error_dict = {key: f"{self.state_dict[key].get_max_p_error():.2f}" for key in key_list}
        # print(f"infer_new_init_benefit: error_dict = {error_dict}.")

        # key_selected = np.random.choice(key_list, size = 5, replace = False, p = error_list)
        # 考虑基于benefit进行概率路径剪枝
        for key in key_list:
            single_state: SingleState = self.state_dict[key]
            candidate_list, valid_index_list = single_state.infer_new_init_query(init_meta, card_dict)
            tree_result[key] = candidate_list, valid_index_list

        return tree_result
    
    def infer_spec_new_init(self, init_meta, card_dict, path_list, ref_benefit = None):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            tree_result:
            res2:
        """
        tree_result = {}
        for key in path_list:
            try:
                single_state: SingleState = self.state_dict[key]
            except Exception as e:
                print(f"infer_spec_new_init: meet KeyError. key = {key}. available_paths = {self.state_dict.keys()}.")
                raise e
            
            candidate_list, valid_index_list = single_state.infer_new_init_query(
                init_meta, card_dict, ref_benefit=ref_benefit)
            tree_result[key] = candidate_list, valid_index_list
            # print(f"infer_spec_new_init: key = {key}. candidate_list = {candidate_list}.")
        return tree_result

    def infer_new_init_under_index_ref(self, init_meta, card_dict, path_list, external_index_dict: dict, ref_benefit = None):
        """
        在index参照下估计性能
        
        Args:
            arg1:
            arg2:
        Returns:
            tree_result:
            res2:
        """
        tree_result = {}
        for key in path_list:
            if key in external_index_dict:
                external_index_list = external_index_dict[key]
            else:
                external_index_list = None
                
            # print(f"infer_new_init_under_index_ref: key = {key}. external_index_list = {external_index_list}.")
            try:
                single_state: SingleState = self.state_dict[key]
            except Exception as e:
                print(f"infer_spec_new_init: meet KeyError. key = {key}. available_paths = {self.state_dict.keys()}.")
                # raise e
                # 暂时先continue
                continue
            
            candidate_list, valid_index_list = single_state.infer_new_init_query(
                init_meta, card_dict, external_index_list, ref_benefit)
            tree_result[key] = candidate_list, valid_index_list
            # print(f"infer_spec_new_init: key = {key}. candidate_list = {candidate_list}.")
        return tree_result
    

# %%

def tree_result_filter(tree_result: dict, ref_p_error):
    """
    根据p_error过滤获取的结果
    
    Args:
        tree_result: key为schema_order，
            value为(instance_type, (query_meta, card_dict), p_error)组成的元组
        ref_p_error:
    Returns:
        result_list: 元组为(schema_order, (query_meta, card_dict))
        max_error: 当前比较下最大的p_error
        max_path: 
    """
    result_list = []
    max_error, max_path = 0.0, None
    for schema_order, (candidate_list, valid_index_list) in tree_result.items():
        # 2024-03-19: 打印item，更新path_list的生成方法
        # if len(candidate_list) > 0:
        #     print(f"tree_result_filter: item[0] = {candidate_list[0]}.")
        for item in candidate_list:
            assert len(item) == 4, f"tree_result_filter: length = {len(item)}. item = {item}."
            if item[0] == "invalid_extend":
                # 
                curr_p_error = 10.0
                # curr_p_error = 100.0
            else:
                curr_p_error = item[2]

            if curr_p_error > ref_p_error:
                # result_list.append((schema_order, item[1]))
                result_list.append((item[-1], item[1]))

            if curr_p_error > max_error:
                max_error = max(max_error, curr_p_error)
                # max_path = (schema_order, item[1])
                max_path = (item[-1], item[1])

    return result_list, max_error, max_path
    # return out_p_error

# %%

def tree_result_max(tree_result: dict):
    ref_p_error = 0.0
    error_list = []
    # 直接打印树的结果
    # pprint(tree_result)
    for schema_order, (candidate_list, valid_index_list) in tree_result.items():
        # print(f"tree_result_max: schema_order = {schema_order}. error_list = {utils.list_round([item[2] for item in candidate_list])}. valid_index_list = {valid_index_list}.")
        for item in candidate_list:
            try:
                if item[0] == "invalid_extend":
                    # 
                    curr_p_error = 10.0
                else:
                    curr_p_error = item[2]

                error_list.append(curr_p_error)
                if curr_p_error > ref_p_error:
                    ref_p_error = curr_p_error
            except IndexError as e:
                print(f"tree_result_max: meet IndexError. item = {item}. tree_result = {tree_result}.")
                raise e

    # print(f"tree_result_max: error_list = {utils.list_round(error_list, digit=3)}")
    return ref_p_error

# %%
