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

from query import query_exploration, query_construction
from collections import defaultdict
from data_interaction import data_management
from result_analysis import case_analysis
from utility import utils, common_config
from workload import physical_plan_info

from pprint import pprint

# %%

def add_log_noise(in_val, scale, out_num):
    """
    在某个值上添加噪声

    Args:
        in_val:
        scale:
        out_num: 
    Returns:
        out_list:
        return2:
    """
    if in_val <= 0:
        # print(f"add_log_noise: in_val = {in_val}.")
        in_val = 1

    # assert in_val > 0
    log_val = np.log(in_val)
    sample_list = np.random.normal(log_val, scale, out_num)
    out_list = [np.exp(a) for a in sample_list]

    return out_list

# %%

def apply_adjust_factor(in_factor, card_val, with_noise = True):
    """
    将adjust_factor应用到cardinality中
    
    Args:
        in_factor:
        card_val:
    Returns:
        out_val_list:
        res2:
    """
    sample_num, scale = common_config.global_sample_num, common_config.global_scale
    if sample_num > 0 and with_noise == True:
        # sample_num为0的时候，不采用add_noise
        factor_list = add_log_noise(in_factor, scale, sample_num)
        out_val_list = [card_val * f for f in factor_list]
        return out_val_list
        # 
    else:
        return card_val * in_factor

# %%
class CaseBasedEstimator(object):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, workload):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        self.workload = workload
        self.query_ctrl = query_exploration.QueryController(workload=self.workload)
        self.alias_mapping = query_construction.abbr_option[workload]
        self.instance_list = []

    def add_new_case(self, query_meta, card_dict):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        self.instance_list.append((query_meta, card_dict))


    def eval_new_init(self, new_init, card_dict, mode = "both", with_noise = True):
        """
        估计new_init带来的价值
    
        Args:
            new_init:
            card_dict:
            mode:
        Returns:
            result_list: item = (query_meta, card_dict)
            return2: 
        """
        result_list = []
        for idx, ref_case in enumerate(self.instance_list):
            ref_meta, ref_card_dict = ref_case

            try:
                local_res = self.inference_on_new_init(\
                    new_init, card_dict, ref_meta, ref_card_dict, mode, with_noise)
            except TypeError as e:
                print(f"eval_new_init: meet TypeError. ref_meta = {ref_meta}. ref_card_dict = {ref_card_dict}.")
                raise e

            # out_meta, out_card_dict = local_res
            # print(f"eval_new_init: out_meta = {out_meta}.")
            # utils.display_card_dict(out_card_dict)

            # result_list.append(local_res)
            # 2024-03-20: 添加instance index
            result_list.append((local_res, idx))

        # print(f"eval_new_init: result_list = {len(result_list)}. instance_list = {len(self.instance_list)}.")
        return result_list
    
    def eval_new_condition(self, new_meta, est_card = None, true_card = None, mode = "both"):
        """
        {Description}
        
        Args:
            new_meta:
            est_card:
            true_card:
            mode:
        Returns:
            result_list: item = (query_meta, card_dict)
            res2:
        """
        result_list = []
        for ref_case in self.instance_list:
            ref_meta, ref_card_dict = ref_case
            local_res = self.inference_on_new_condition(\
                new_meta, ref_meta, ref_card_dict, est_card, true_card, mode)
            result_list.append(local_res)

        return result_list

    def inference_on_new_condition(self, new_meta, ref_meta, ref_dict, est_card = None, true_card = None, mode = "both"):
        """
        在新的condition上做估计
        
        Args:
            new_meta: 
            ref_meta: 
            ref_dict: 
            est_card: 
            true_card:
        Returns:
            out_meta: 
            out_card_dict:
        """
        print(f"inference_on_new_condition: mode = {mode}.")
        out_meta = self.meta_regroup(new_meta, ref_meta)
        target_table = new_meta[0][0]
        target_alias = self.alias_mapping[target_table]

        out_card_dict = {
            "true": {},
            "estimation": {}
        }
        subquery_true_ref, single_table_true_ref, subquery_est_ref, \
            single_table_est_ref = utils.extract_card_info(ref_dict)

        if mode == "both" or mode == "true":
            # 填充真实基数
            subquery_true_out, single_table_true_out = {}, {}

            if true_card is not None:
                adjust_factor = true_card / single_table_true_ref[target_alias]
            else:
                adjust_factor = est_card / single_table_est_ref[target_alias]

            for k in subquery_true_ref.keys():
                if target_alias in k:
                    subquery_true_out[k] = subquery_true_ref[k] * adjust_factor
                else:
                    subquery_true_out[k] = subquery_true_ref[k]

            for k in single_table_true_ref.keys():
                if k == target_alias:
                    single_table_true_out[k] = single_table_true_ref[k] * adjust_factor
                else:
                    single_table_true_out[k] = single_table_true_ref[k]

            out_card_dict['true']['subquery'] = subquery_true_out
            out_card_dict['true']['single_table'] = single_table_true_out
        
        if mode == "both" or mode == "estimation":
            # 填充估计基数
            subquery_est_out, single_table_est_out = {}, {}

            if est_card is not None:
                adjust_factor = est_card / single_table_est_ref[target_alias]
            else:
                adjust_factor = true_card / single_table_true_ref[target_alias]

            for k in subquery_est_ref.keys():
                if target_alias in k:
                    subquery_est_out[k] = subquery_est_ref[k] * adjust_factor
                else:
                    subquery_est_out[k] = subquery_est_ref[k]

            for k in single_table_est_ref.keys():
                if k == target_alias:
                    single_table_est_out[k] = single_table_est_ref[k] * adjust_factor
                else:
                    single_table_est_out[k] = single_table_est_ref[k]

            out_card_dict['estimation']['subquery'] = subquery_est_out
            out_card_dict['estimation']['single_table'] = single_table_est_out

        return out_meta, out_card_dict

    
    def inference_on_new_init(self, init_meta, card_dict, ref_meta, ref_dict, mode = "both", with_noise = True):
        """
        在新的init_query上做估计
        
        Args:
            init_meta: 
            card_dict: 
            ref_meta: 
            ref_dict: 
            mode:
        Returns:
            out_meta: 
            out_card_dict:
        """
        assert mode in ("true", "estimation", "both")
        # print(f"inference_on_new_init: mode = {mode}.")
        out_meta = self.meta_regroup(init_meta, ref_meta)

        out_card_dict = {
            "true": {
                "subquery": {},
                "single_table": {}
            },
            "estimation": {
                "subquery": {},
                "single_table": {}
            }
        }

        in_alias_set = set(query_construction.get_alias_list(init_meta, self.workload))
        ref_alias_set = set(query_construction.get_alias_list(ref_meta, self.workload))
        comp_alias_set = ref_alias_set.difference(in_alias_set)

        subquery_true_ref, single_table_true_ref, subquery_est_ref, \
            single_table_est_ref = utils.extract_card_info(ref_dict)
        
        # utils.display_card_dict(ref_dict, "inference_on_new_init")
        subquery_true_in, single_table_true_in, subquery_est_in, \
            single_table_est_in = utils.extract_card_info(card_dict)
        
        def int_val(a):
            if isinstance(a, (list, tuple)):
                return [int(item) for item in a]
            else:
                return int(a)
        
        if mode == "both":
            # 2024-03-16: both模式下确认card_dict的keys匹配
            assert set(subquery_true_ref.keys()) == set(subquery_est_ref.keys()), \
                f"inference_on_new_init: subquery_true_ref = {subquery_true_ref.keys()}. subquery_est_ref = {subquery_est_ref.keys()}"
            assert set(single_table_true_ref.keys()) == set(single_table_est_ref.keys()), \
                f"inference_on_new_init: single_table_true_ref = {single_table_true_ref.keys()}. single_table_est_ref = {single_table_est_ref.keys()}."

        if mode == "both" or mode == "true":
            # 填充真实基数
            subquery_true_out, single_table_true_out = {}, {}

            for k in subquery_true_ref.keys():
                flag1, flag2 = set(k).issubset(in_alias_set), set(k).issubset(comp_alias_set)
                if flag1 == False and flag2 == False:
                    closet_key = self.find_closet_reference(in_alias_set, k)
                    if len(closet_key) > 1:
                        # 2024-03-13: +1，防止真实值为空
                        adjust_factor = subquery_true_in[closet_key] / (1e-2 + subquery_true_ref[closet_key])
                    else:
                        closet_key = closet_key[0]
                        # 2024-03-13: +1，防止真实值为空
                        adjust_factor = single_table_true_in[closet_key] / (1e-2 + single_table_true_ref[closet_key])
                    # subquery_true_out[k] = adjust_factor * subquery_true_ref[k]
                    subquery_true_out[k] = apply_adjust_factor(adjust_factor, subquery_true_ref[k], with_noise)
                elif flag1 == True:
                    subquery_true_out[k] = subquery_true_in[k]
                else:
                    subquery_true_out[k] = subquery_true_ref[k]

            for k in single_table_true_ref.keys():
                if k in in_alias_set:
                    single_table_true_out[k] = single_table_true_in[k]
                else:
                    single_table_true_out[k] = single_table_true_ref[k]

            out_card_dict['true']['subquery'] = utils.dict_apply(subquery_true_out, int_val)
            out_card_dict['true']['single_table'] = utils.dict_apply(single_table_true_out, int_val)
        
        if mode == "both" or mode == "estimation":
            # 填充估计基数
            subquery_est_out, single_table_est_out = {}, {}

            for k in subquery_est_ref.keys():
                flag1, flag2 = set(k).issubset(in_alias_set), set(k).issubset(comp_alias_set)

                if flag1 == False and flag2 == False:
                    # 两边都匹配不上的情况
                    try:
                        closet_key = self.find_closet_reference(in_alias_set, k)
                        if len(closet_key) > 1:
                            adjust_factor = subquery_est_in[closet_key] / subquery_est_ref[closet_key]
                        else:
                            closet_key = closet_key[0]
                            adjust_factor = single_table_est_in[closet_key] / single_table_est_ref[closet_key]
                    except KeyError as e:
                        print(f"inference_on_new_init: closet_key = {closet_key}. in_alias_set = {in_alias_set}. k = {k}")
                        raise e
                    # subquery_est_out[k] = adjust_factor * subquery_est_ref[k]
                    subquery_est_out[k] = apply_adjust_factor(adjust_factor, subquery_est_ref[k], with_noise)
                elif flag1 == True:
                    subquery_est_out[k] = subquery_est_in[k]
                else:
                    subquery_est_out[k] = subquery_est_ref[k]

            for k in single_table_est_ref.keys():
                if k in in_alias_set:
                    single_table_est_out[k] = single_table_est_in[k]
                else:
                    single_table_est_out[k] = single_table_est_ref[k]

            out_card_dict['estimation']['subquery'] = utils.dict_apply(subquery_est_out, int_val)
            out_card_dict['estimation']['single_table'] = utils.dict_apply(single_table_est_out, int_val)

        # utils.display_card_dict(out_card_dict, "inference_on_new_init")
        return out_meta, out_card_dict
    
    def meta_valid_check(self, in_meta, ref_meta):
        """
        {Description}
    
        Args:
            in_meta:
            ref_meta:
        Returns:
            flag:
            return2:
        """
        flag = True
        for schema in in_meta[0]:
            if schema in ref_meta[0]:
                flag = False
        return flag
    

    def find_closet_reference(self, in_alias_list, target_alias_tuple):
        """
        找到meta最近的基数参考
    
        Args:
            in_alias_list:
            target_alias_tuple:
        Returns:
            res_alias_tuple:
            return2:
        """
        res_alias_list = []

        for alias in in_alias_list:
            if alias in target_alias_tuple:
                res_alias_list.append(alias)

        return tuple(sorted(res_alias_list))


    def meta_elimination(self, in_meta, ref_meta):
        """
        相同元信息消除
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """

        in_filter_dict = self.build_filter_dict(in_meta[1])
        ref_filter_dict = self.build_filter_dict(ref_meta[1])

        def condition_compare(alias):
            cond_set1 = set(in_filter_dict[alias])
            cond_set2 = set(ref_filter_dict[alias])
            return cond_set1 == cond_set2

        out_meta = [], in_meta[1]
        for schema in in_meta[0]:
            alias = self.alias_mapping[schema]
            if condition_compare(alias) == True:
                continue
            else:
                out_meta[0].append(schema)

        return out_meta

    def build_filter_dict(self, filter_list):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        filter_dict = defaultdict(list)
        for item in filter_list:
            # 
            curr_key = item[0]
            filter_dict[curr_key].append(item)

        return filter_dict

    def meta_regroup(self, in_meta, ref_meta):
        """
        重新组合元信息
    
        Args:
            in_meta: 新输入的元信息
            ref_meta: 用于参考的元信息
        Returns:
            out_meta:
            subquery_prop:
            single_table_prop:
        """
        in_meta = self.meta_elimination(in_meta, ref_meta)

        out_meta = [], []
        for schema in ref_meta[0]:
            out_meta[0].append(schema)

        in_filter_dict = self.build_filter_dict(in_meta[1])
        ref_filter_dict = self.build_filter_dict(ref_meta[1])

        for k, v in ref_filter_dict.items():
            if k not in in_filter_dict.keys():
                for item in v:
                    out_meta[1].append(item)
            else:
                for item in in_filter_dict[k]:
                    out_meta[1].append(item)

        return out_meta

    def inference_on_case(self, query_meta, card_dict, ref_meta, ref_dict, mode):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        out_meta = self.meta_regroup(query_meta, ref_meta)
        out_card_dict = {
            "true": {},
            "estimation": {}
        }
        in_alias_set = set(query_construction.get_alias_list(query_meta, self.workload))
        ref_alias_set = set(query_construction.get_alias_list(ref_meta, self.workload))
        comp_alias_set = ref_alias_set.difference(in_alias_set)

        subquery_true_ref, single_table_true_ref, subquery_est_ref, \
            single_table_est_ref = utils.extract_card_info(ref_dict)
        
        subquery_true_in, single_table_true_in, subquery_est_in, \
            single_table_est_in = utils.extract_card_info(card_dict)

        def construct_adjust_factors(in_key, mode="true"):
            if mode == "true":
                preferred_in, preferred_ref = single_table_true_in, single_table_true_ref
                alternative_in, alternative_ref = single_table_est_in, single_table_est_ref
            elif mode == "estimation":
                preferred_in, preferred_ref = single_table_est_in, single_table_est_ref
                alternative_in, alternative_ref = single_table_true_in, single_table_true_ref

            # 构造基数调整系数
            res_list = []
            if isinstance(in_key, tuple):
                # 
                for k in in_key:
                    if k in in_alias_set:
                        # 添加单表的调整值
                        try:
                            res_list.append(preferred_in[k]/preferred_ref[k])
                        except KeyError:
                            res_list.append(alternative_in[k]/alternative_ref[k])
            elif isinstance(in_key, str):
                # 
                k = in_key
                try:
                    res_list.append(preferred_in[k]/preferred_ref[k])
                except KeyError:
                    res_list.append(alternative_in[k]/alternative_ref[k])

            return res_list
        
        if mode == "both" or mode == "true":
            # 构建真实基数
            subquery_true_out, single_table_true_out = {}, {}

            for k in subquery_true_ref.keys():
                flag1, flag2 = set(k).issubset(in_alias_set), set(k).issubset(comp_alias_set)
                if flag2 == True:
                    subquery_true_out[k] = subquery_true_ref[k]
                else:
                    if flag1 == True and k in subquery_true_in:
                        subquery_true_out[k] = subquery_true_in[k]
                    else:
                        adjust_factor_list = construct_adjust_factors(k, mode="true")
                        subquery_true_out[k] = self.estimate_spec_cardinality(\
                            adjust_factor_list, subquery_true_ref[k])
                        
            for k in single_table_true_ref.keys():
                if k in in_alias_set:
                    single_table_true_out[k] = single_table_true_in[k]
                else:
                    single_table_true_out[k] = single_table_true_ref[k]

            out_card_dict['true']['subquery'] = utils.dict_apply(subquery_true_out, int)
            out_card_dict['true']['single_table'] = utils.dict_apply(single_table_true_out, int)
            
        if mode == "both" or mode == "estimation":
            # 构建估计基数
            subquery_est_out, single_table_est_out = {}, {}

            for k in subquery_est_ref.keys():
                flag1, flag2 = set(k).issubset(in_alias_set), set(k).issubset(comp_alias_set)
                if flag2 == True:
                    subquery_est_out[k] = subquery_est_ref[k]
                else:
                    if flag1 == True and k in subquery_est_in:
                        subquery_est_out[k] = subquery_est_in[k]
                    else:
                        adjust_factor_list = construct_adjust_factors(k, mode="estimation")
                        subquery_est_out[k] = self.estimate_spec_cardinality(\
                            adjust_factor_list, subquery_est_ref[k])
                        
            for k in single_table_est_ref.keys():
                if k in in_alias_set:
                    single_table_est_out[k] = single_table_est_in[k]
                else:
                    single_table_est_out[k] = single_table_est_ref[k]

            out_card_dict['estimation']['subquery'] = utils.dict_apply(subquery_est_out, int)
            out_card_dict['estimation']['single_table'] = utils.dict_apply(single_table_est_out, int)

        return out_meta, out_card_dict

    def estimate_spec_cardinality(self, adjust_factor_list, ref_card):
        """
        {Description}
        
        Args:
            adjust_factor_list:
            ref_card:
        Returns:
            out_card:
            res2:
        """
        union_factor = np.prod(adjust_factor_list)
        return int(union_factor * ref_card)


    def estimate_p_error(self, query_meta, card_dict):
        """
        推断新查询的p_error
    
        Args:
            query_meta:
            card_dict:
        Returns:
            p_error:
            return2:
        """
        query_text = query_construction.construct_origin_query(query_meta, self.workload)
        local_analyzer = case_analysis.CaseAnalyzer(query_text, query_meta, (), card_dict)
        return local_analyzer.p_error
    

    def card_complete_check(self, query_meta, subquery_dict, single_table_dict):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        flag = True
        single_table_alias = [self.alias_mapping[s] for s in query_meta[0]]
        subquery_alias = data_management.get_all_subqueries(\
            self.workload, single_table_alias)
        
        for item in single_table_alias:
            if item not in single_table_dict:
                flag = False
        
        for item in subquery_alias:
            if item not in subquery_dict:
                flag = False

        return flag

    def estimate_query_plan(self, query_text, query_meta, card_dict, mode):
        """
        估计生成的查询计划
        
        Args:
            query_meta: 查询元信息
            card_dict: 基数字典
            mode: 可选的模式("true", "estimation", "both")
        Returns:
            res_dict:
            res2:
        """
        res_dict = {}
        assert mode in ("true", "estimation", "both")
        
        subquery_true, single_table_true, subquery_estimation, \
            single_table_estimation = utils.extract_card_info(card_dict)
        
        local_ctrl = query_exploration.QueryController(workload=self.workload)
        local_ctrl.set_query_instance(query_text, query_meta)

        if mode == "true" or mode == "both":
            if self.card_complete_check(query_meta, \
                subquery_true, single_table_true) == False:
                raise ValueError("true_card not complete")
            
            true_plan_dict = local_ctrl.get_plan_by_external_card(\
                subquery_true, single_table_true)
            true_plan_physical = physical_plan_info.PhysicalPlan(query_text, true_plan_dict)
            res_dict['true'] = true_plan_physical
        elif mode == "estimation" or mode == "both":
            if self.card_complete_check(query_meta, \
                subquery_estimation, single_table_estimation) == False:
                raise ValueError("est_card not complete")

            est_plan_dict = local_ctrl.get_plan_by_external_card(\
                subquery_estimation, single_table_estimation)
            est_plan_physical = physical_plan_info.PhysicalPlan(query_text, est_plan_dict)
            res_dict['estimation'] = est_plan_physical

        return res_dict
    
    
    def case_matching(self,):
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


# %%

def case_pair_estimation(workload, meta_ref, card_ref, meta_test, card_test, mode):
    """
    针对单例的结果估计

    Args:
        workload: 工作负载
        query_ref:
        card_ref:
        query_test:
        card_test:
        mode: 基数估计的模式
    Returns:
        out_meta:
        out_card_dict:
    """
    case_based_est = CaseBasedEstimator(workload)
    # case_based_est.add_new_case(meta_ref, card_ref)
    out_meta, out_card_dict = case_based_est.inference_on_case(\
        meta_test, card_test, meta_ref, card_ref, mode)
    
    return out_meta, out_card_dict


# %%
