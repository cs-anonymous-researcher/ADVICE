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
import torchquad
import scipy
from collections import defaultdict

# %%

from query import query_exploration, query_construction
from data_interaction import postgres_connector
from utility import utils
from plan import node_utils
from estimation import plan_estimation
from multiprocessing import Pool
from result_analysis import case_analysis
from scipy.stats import multivariate_normal
from workload import physical_plan_info
import vegas

# %%

def get_instance_p_error(input_params):
    workload, query_meta, card_dict = input_params
    analyzer = case_analysis.construct_case_instance(query_meta, card_dict, workload)
    return analyzer.p_error


class BaseSampling(object):
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
        self.index_mapping, self.index_inverse = {}, {}
        # 创建数据库连接
        self.db_conn = postgres_connector.connector_instance(workload)
        # self.query_ctrl: query_exploration.QueryController = query_exploration.get_query_controller(workload)
        self.query_ctrl: query_exploration.QueryController = \
            query_exploration.QueryController(self.db_conn, workload=workload)

    def get_physical_plan(self, subquery_dict, single_table_dict):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        # 获得物理计划
        assert self.query_ctrl.query_text != ""
        target_plan = node_utils.plan_evaluation_under_cardinality(self.workload, 
            self.query_ctrl, self.query_meta, subquery_dict, single_table_dict)
        return target_plan
    

    def get_p_error(self, subquery_true, subquery_est, single_table_true, single_table_est, with_plan = False):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            p_error: 
            true_plan: 
            est_plan:
        """
        # 获得p-error
        true_plan = self.get_physical_plan(subquery_true, single_table_true)
        est_plan = self.get_physical_plan(subquery_est, single_table_est)

        # 分别代入真实基数获得cost
        true_plan.set_database_connector(db_conn=self.db_conn)
        est_plan.set_database_connector(db_conn=self.db_conn)

        cost_true = true_plan.get_plan_cost(subquery_true, single_table_true)
        cost_est = est_plan.get_plan_cost(subquery_true, single_table_true)

        # 计算p-error
        p_error = cost_est / cost_true

        if with_plan == False:
            return p_error
        else:
            return p_error, true_plan, est_plan

    def build_card_dict_index(self, subquery_missing: list, single_table_missing: list):
        """
        将缺失的cardinality索引化，即每个subquery、single_table对应到一个index上，目前考虑
        将subquery和single_table放到一个index空间中
        
        Args:
            arg1:
            arg2:
        Returns:
            index_mapping:
            index_inverse:
        """
        # 清空之前的结果
        self.index_mapping, self.index_inverse = {}, {}
        curr_index = 0

        for k in subquery_missing:
            self.index_mapping[k] = curr_index
            self.index_inverse[curr_index] = k
            curr_index += 1

        for k in single_table_missing:
            self.index_mapping[k] = curr_index
            self.index_inverse[curr_index] = k
            curr_index += 1

        return self.index_mapping, self.index_inverse
    

    def restore_origin_card(self, key, val):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        if isinstance(key, str):
            est_card = self.single_table_true[key][0]
        elif isinstance(key, tuple):
            est_card = self.subquery_true[key][0]
        ratio = np.exp(val)
        return int(ratio * est_card)
    

    def parse_card_dict_index(self, value_arr):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        subquery_out, single_table_out = {}, {}
        for idx, val in enumerate(value_arr):
            k = self.index_inverse[idx]
            if isinstance(k, str):
                single_table_out[k] = self.restore_origin_card(k, val)
            elif isinstance(k, tuple):
                subquery_out[k] = self.restore_origin_card(k, val)
        
        # print(f"parse_card_dict_index: value_arr = {value_arr}. single_table_out = {single_table_out}. subquery_out = {subquery_out}.")
        return subquery_out, single_table_out

    def load_instance(self, query_meta, res_card_dict, subquery_missing, single_table_missing, target_table):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        self.query_meta, self.res_card_dict, self.subquery_missing, self.single_table_missing, \
            self.target_table = query_meta, res_card_dict, subquery_missing, single_table_missing, target_table
        
        # 拆分card_dict
        self.subquery_true, self.single_table_true, self.subquery_est, self.single_table_est = \
            utils.extract_card_info(self.res_card_dict)

        self.query_text = query_construction.construct_origin_query(self.query_meta, self.workload)
        self.query_ctrl.set_query_instance(self.query_text, query_meta)
        self.build_card_dict_index(subquery_missing, single_table_missing)

        # 构造分布的信息
        # distribution_info = mean_array, cov_array
        mean_list, std_list = [], []

        for idx in sorted(self.index_inverse.keys()):
            k = self.index_inverse[idx]
            key = self.index_inverse[idx]   
            if isinstance(key, tuple):
                curr_mean, curr_std = self.subquery_true[key][1], self.subquery_true[key][2]
            elif isinstance(key, str):
                curr_mean, curr_std = self.single_table_true[key][1], self.single_table_true[key][2]

            mean_list.append(curr_mean)
            std_list.append(curr_std * curr_std)

        self.distribution_info = np.array(mean_list), np.diag(std_list)

    def construct_target_func(self, with_prob = False):
        """
        构造目标值的函数，输入是一组sample的值，输出是带有概率的P-Error，主要用于torchquad库
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        def out_func(in_params):
            # input是多维的输入
            # print(f"construct_target_func.out_func: in_params = {in_params}.")
            subquery_out, single_table_out = self.parse_card_dict_index(in_params)
            
            subquery_true = utils.dict_merge(self.subquery_true, subquery_out)
            single_table_true = utils.dict_merge(self.single_table_true, single_table_out)
            subquery_est, single_table_est = \
                deepcopy(self.subquery_est), deepcopy(self.single_table_est)
            
            # 构造cardinality的分布
            # mean_list, std_list = [], []
            mean, cov = self.distribution_info
            # print(f"mean = {mean}. cov = {cov}.")
            # 计算P-Error结果
            p_error = self.get_p_error(subquery_true, subquery_est, 
                single_table_true, single_table_est)

            if with_prob == True:
                # 计算概率对应的值
                prob_val = multivariate_normal.pdf(x = in_params, mean=mean, cov=cov)
                result = p_error * prob_val
            else:
                result = p_error

            return result

        return out_func


    def construct_card_range(self, ):
        """
        构建基数的范围，根据分布选择合法的区域
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        ordered_idx = sorted(self.index_inverse.keys())
        range_list = []
        for idx in ordered_idx:
            # 获得对应键值
            key = self.index_inverse[idx]   
            if isinstance(key, tuple):
                dist_mean, dist_std = self.subquery_true[key][1], self.subquery_true[key][2]
            elif isinstance(key, str):
                dist_mean, dist_std = self.single_table_true[key][1], self.single_table_true[key][2]
            else:
                raise ValueError(f"construct_card_range: type(key) = {type(key)}.")
            
            start, end = dist_mean - 1.5 * dist_std, dist_mean + 1.5 * dist_std
            range_list.append((start, end))

        return np.array(range_list)

    def calculate_probability(self,):
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


    def sample_on_dist(self, total_sample_num):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        raise NotImplementedError("sample_on_dist has not been implemented!")


# %%

class ProbSampling(BaseSampling):
    """
    根据概率进行采样

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
        super().__init__(workload)

    def sample_on_dist(self, total_sample_num, card_sample_num = 5):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            p_error_estimation:
            return2:
        """
        subquery_est, single_table_est = deepcopy(self.subquery_est), deepcopy(self.single_table_est)
        subquery_true, single_table_true = self.sample_values(card_sample_num)
        subquery_missing, single_table_missing = self.subquery_missing, self.single_table_missing 

        subquery_candidates = utils.dict_subset(subquery_true, 
            lambda a: a in subquery_missing, mode="key")
        single_table_candidates = utils.dict_subset(single_table_true, 
            lambda a: a in single_table_missing, mode="key")

        value_sample_combinations = plan_estimation.make_combinations(subquery_missing, 
            single_table_missing, subquery_candidates, single_table_candidates, total_sample_num)
        
        #
        card_dict_list = []
        for subquery_cmp, single_table_cmp in value_sample_combinations:
            # 创建新的dict
            local_subquery_dict = utils.dict_merge(subquery_true, subquery_cmp)
            local_single_table_dict = utils.dict_merge(single_table_true, single_table_cmp)

            local_card_dict = utils.pack_card_info(local_subquery_dict, 
                local_single_table_dict, subquery_est, single_table_est, dict_copy=True)
            
            card_dict_list.append(local_card_dict)

        if len(card_dict_list) < 5:
            sample_error_list = [get_instance_p_error((self.workload, self.query_meta, card_dict)) for card_dict in card_dict_list]
        else:
            # 
            params_list = [(self.workload, self.query_meta, card_dict) for card_dict in card_dict_list]
            with Pool(10) as p:
                sample_error_list = p.map(get_instance_p_error, params_list)

        return np.average(sample_error_list)

    def sample_values(self, card_sample_num = 5):
        """
        {Description}

        Args:
            card_sample_num:
            arg2:
        Returns:
            return1:
            return2:
        """
        subquery_true, single_table_true = deepcopy(self.subquery_true), deepcopy(self.single_table_true)

        #
        for key in self.index_mapping.keys():
            
            if isinstance(key, tuple):
                est_card, dist_mean, dist_std = subquery_true[key]
                sample_list= np.random.normal(dist_mean, dist_std, card_sample_num)
                subquery_true[key] = est_card * np.exp(sample_list)
            else:
                est_card, dist_mean, dist_std = single_table_true[key]
                sample_list = np.random.normal(dist_mean, dist_std, card_sample_num)
                single_table_true[key] = est_card * np.exp(sample_list)

        return subquery_true, single_table_true
# %%

class ImportanceSampling(BaseSampling):
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
        super().__init__(workload)

    def sample_on_dist(self, total_sample_num):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        # val_func = self.construct_target_func(with_prob=True)
        val_func = self.construct_target_func(with_prob=True)

        # 测试函数
        # val_func([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

        func_domain = self.construct_card_range()
        func_dim = len(self.index_mapping)  # 函数的维度

        # 
        # mc = torchquad.VEGAS()
        # print(f"sample_on_dist: val_func = {val_func}. func_domain = {func_domain}. func_dim = {func_dim}. total_sample_num = {total_sample_num}.")
        # integral_value = mc.integrate(fn=val_func, dim=func_dim, N=total_sample_num, integration_domain = func_domain)

        #
        integ = vegas.Integrator(func_domain)
        integral_value = integ(val_func, nitn=10, neval=100)

        return integral_value

    def func_name2(self,):
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

class DeterministicSampling(BaseSampling):
    """
    确定性的采样方法

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
        super().__init__(workload)

    def sample_on_dist(self, total_sample_num):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        val_func = lambda a: a
        func_dim = len(self.index_mapping)  # 函数的维度
        mc = torchquad.Boole()
        integral_value = mc.integrate(fn=val_func, 
            dim=func_dim, N=total_sample_num)

        return integral_value

    def func_name2(self,):
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

class PlanGuidedSampling(ProbSampling):
    """
    根据Plan结果进行采样

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
        super().__init__(workload)


    def sample_on_dist(self, total_sample_num):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        subquery_true, single_table_true = self.sample_values(total_sample_num)
        subquery_est, single_table_est = deepcopy(self.subquery_est), deepcopy(self.single_table_est)

        # 当前的基数字典
        subquery_curr, single_table_curr = {}, {}
        # 当前基数迭代的位置
        self.subquery_pos, self.single_table_pos = {}, {}

        # 初始化操作
        for k in self.index_mapping.keys():
            if isinstance(k, tuple):
                subquery_curr[k] = subquery_true[k][0]
                self.subquery_pos[k] = [0, len(subquery_true[k]) - 1]
            elif isinstance(k, str):
                single_table_curr[k] = single_table_true[k][0]
                self.single_table_pos[k] = [0, len(single_table_true[k]) - 1]

        self.subquery_weight, self.single_table_weight = self.construct_weight()

        sample_cnt = 0
        p_error_list = []

        while True:
            sample_cnt += 1
            p_error, true_plan, est_plan = self.get_p_error(subquery_curr, 
                subquery_est, single_table_curr, single_table_est, with_plan=True)
            k, new_value, new_pos = self.adjust_cardinalities(true_plan, est_plan)

            if k == "":
                break

            if isinstance(k, tuple):
                subquery_curr[k] = new_value
                self.subquery_pos[k][0] = new_pos
            elif isinstance(k, str):
                single_table_curr[k] = new_value
                self.single_table_pos[k] = new_pos

            # print(f"sample_on_dist: sample_cnt = {sample_cnt}. p_error = {p_error:.2f}.")
            p_error_list.append(p_error)
            if sample_cnt >= total_sample_num:
                break

        return np.average(p_error_list)

    def sample_values(self, total_sample_num):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        subquery_curr, single_table_curr = {}, {}

        # 根据distribution的variance设置sample数目
        weight_list= []

        for idx, key in sorted(self.index_inverse.items()):
            if isinstance(key, tuple):
                std_val = self.subquery_true[key][2]
            else:
                std_val = self.single_table_true[key][2]
            weight_list.append(std_val)
        
        weight_sum = np.sum(weight_list)
        for idx, weight in enumerate(weight_list):
            sample_num = int(weight * total_sample_num / weight_sum)
            key = self.index_inverse[idx]

            if isinstance(key, tuple):
                est_card, dist_mean, dist_std = self.subquery_true[key]
                sample_list= np.random.normal(dist_mean, dist_std, sample_num)
                subquery_curr[key] = est_card * np.exp(sample_list)
                # subquery_curr[key] = None
            elif isinstance(key, str):
                est_card, dist_mean, dist_std = self.single_table_true[key]
                sample_list = np.random.normal(dist_mean, dist_std, sample_num)
                single_table_curr[key] = est_card * np.exp(sample_list)

        subquery_true = utils.dict_merge(self.subquery_true, subquery_curr)
        single_table_true = utils.dict_merge(\
            self.single_table_true, single_table_curr)

        # 设置subquery/single_table的candidate values
        self.subquery_candidates = subquery_true
        self.single_table_candidates = single_table_true

        return subquery_true, single_table_true


    def adjust_cardinalities(self, true_plan, est_plan):
        """
        {Description}

        Args:
            true_plan:
            est_plan:
        Returns:
            key:
            new_value:
            new_pos:
        """
        key, new_value, new_pos = "", 0, 0
        # 根据Plan状态，抽取候选调整的subquery
        # candidate_list, prob_list = [], []
        candidate_list, prob_list = self.analyze_query_plan(true_plan)

        if len(candidate_list) == 0:
            return key, new_value, new_pos
        
        prob_sum = sum(prob_list)
        prob_list = [prob / prob_sum for prob in prob_list]

        # 按概率随机选择subquery
        selected_idx = np.random.choice(a = np.arange(len(candidate_list)), p = prob_list)
        key = candidate_list[selected_idx]

        # 设置新的值
        if isinstance(key, tuple):
            curr_pos = self.subquery_pos[key][0]
            new_pos = curr_pos + 1
            new_value = self.subquery_candidates[key][new_pos]
        elif isinstance(key, str):
            curr_pos = self.single_table_pos[key][0]
            new_pos = curr_pos + 1
            new_value = self.single_table_candidates[key][new_pos]
        
        return key, new_value, new_pos
    
    def construct_weight(self,):
        """
        根据subquery_pos和single_table_pos构建权重，考虑两个点，一是sample_num，二是table_num
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        subquery_weight, single_table_weight = {}, {}
        table_num_cnt = defaultdict(int)
        
        # 构建table_num_cnt
        for k in self.subquery_pos.keys():
            table_num_cnt[len(k)] += 1

        for k in self.single_table_pos.keys():
            table_num_cnt[1] += 1

        # print(f"construct_weight: table_num_cnt = {table_num_cnt}.")

        for k in self.subquery_pos.keys():
            subquery_weight[k] = table_num_cnt[len(k)] * self.subquery_pos[k][1]

        for k in self.single_table_pos.keys():
            single_table_weight[k] = table_num_cnt[1] * self.single_table_pos[k][1]

        # print(f"construct_weight: subquery_weight = {subquery_weight}. single_table_weight = {single_table_weight}.")
        return subquery_weight, single_table_weight

    def iteration_condition(self, key):
        """
        {Description}
    
        Args:
            key:
            arg2:
        Returns:
            flag: 是否支持继续迭代
            return2:
        """
        if isinstance(key, tuple):
            if key in self.subquery_pos.keys() and self.subquery_pos[key][0] < self.subquery_pos[key][1]:
                return True
            else:
                return False
        elif isinstance(key, str):
            if key in self.single_table_pos.keys() and self.single_table_pos[key][0] < self.subquery_pos[key][1]:
                return True
            else:
                return False
        else:
            raise ValueError(f"iteration_condition: type(key) = {type(key)}. key = {key}")
        
        return False

    def analyze_query_plan(self, true_plan: physical_plan_info.PhysicalPlan):
        """
        根据TruePlan获取需要调整的subquery
    
        Args:
            true_plan:
            arg2:
        Returns:
            candidate_list:
            prob_list:
        """
        candidate_list, prob_list = [], []
        subquery_res, single_table_res = postgres_connector.parse_all_subquery_cardinality(true_plan.plan_dict)

        # print(f"analyze_query_plan: self.subquery_pos = {self.subquery_pos}. self.single_table_pos = {self.single_table_pos}.")
        # 处理subquery
        for k, v in subquery_res.items():
            # if k in self.subquery_pos.keys() and :
                # print(f"analyze_query_plan: k = {k}. v = {v}.")
            if self.iteration_condition(k) == True:
                candidate_list.append(k)
                prob_list.append(self.subquery_weight[k])

        # 处理single_table
        for k, v in single_table_res.items():
            # if k in self.single_table_pos.keys() and :
                # print(f"analyze_query_plan: k = {k}. v = {v}.")
            if self.iteration_condition(k) == True:
                candidate_list.append(k)
                prob_list.append(self.single_table_weight[k])

        # raise ValueError("analyze_query_plan: stop.")
        return candidate_list, prob_list

# %%
