#!/usr/bin/env python
# -*- coding: utf-8 -*-

# %%
import requests
from estimation import builtin_card_estimation
from utility import utils, common_config

import numpy as np
import json
from copy import deepcopy
from os.path import join as p_join
# %%

def normal_dist(mean, variance):
    """
    {Description}

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    return np.random.normal(mean, variance)


# %%

# def convert_to_serializable(obj):
#     if isinstance(obj, np.int64):
#         print(f"convert_to_serializable: obj = {obj}.")
#         return int(obj)
#     else:
#         raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

# %% 应用全局变量和函数管理external_estimator

signature = None
current_model = None

def start_estimation(url, workload, ce_method, model_type, sample_num = 1):
    """
    {Description}
    
    Args:
        url:
        workload:
        ce_method:
        model_type:
        sample_num:
    Returns:
        signature:
        res2:
    """
    global signature, current_model
    current_model = model_type
    local_url = p_join(url, "start_new_task")
    reply = requests.post(local_url, params= {
        "workload": workload,
        "method": ce_method,
        "model_type": model_type,
        "sample_num": sample_num
    }, timeout=10)
    signature = reply.json()['signature']
    return signature

def end_estimation(url):
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    global signature, current_model

    signature = None
    current_model = None
    local_url = p_join(url, "finish_task")
    reply = requests.post(local_url, timeout=10)
    print(reply.json())
    return True

# %%

class ExternalEstimator(builtin_card_estimation.BaseEstimator):
    """
    外部的估计器

    Members:
        field1:
        field2:
    """

    def __init__(self, workload, url = "http://101.6.96.160:30007", model_type = "gcn"):
        """
        {Description}

        Args:
            workload:
            arg2:
        """
        self.url, model_type = url, model_type
        super().__init__(workload=workload)


    def dict_key_stringify(self, in_card_dict):
        """
        {Description}
        
        Args:
            in_card_dict:
            arg2:
        Returns:
            out_card_dict:
            res2:
        """
        subquery_true, single_table_true, subquery_est, \
            single_table_est = utils.extract_card_info(in_card_dict)
        
        subquery_true, subquery_est = \
            deepcopy(subquery_true), deepcopy(subquery_est)

        subquery_true = utils.dict_apply(subquery_true, str, mode="key")
        subquery_est = utils.dict_apply(subquery_est, str, mode="key")

        out_card_dict = utils.pack_card_info(subquery_true, \
            single_table_true, subquery_est, single_table_est)
        
        return out_card_dict

    def dict_key_parse(self, subquery_res, single_table_res):
        """
        {Description}
        
        Args:
            subquery_res:
            single_table_res:
        Returns:
            res1:
            res2:
        """
        subquery_out = utils.dict_apply(subquery_res, eval, mode="key")
        single_table_out = utils.dict_apply(single_table_res, str, mode="key")
        return subquery_out, single_table_out

    def valid_eval(self,):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        assert signature is not None, "valid_eval: signature is None"
        return signature

    def construct_request_dict(self, subquery_missing, single_table_missing):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        card_dict = utils.pack_card_info(self.subquery_true, \
            self.single_table_true, self.subquery_estimation, \
            self.single_table_estimation)
        
        # 打印card_dict中所有元素的类型
        # for k, v in card_dict.items():
        #     print(f"k = {k}")
        #     for kk, vv in v.items():
        #         print(f"kk = {kk}")
        #         for kkk, vvv in vv.items():
        #             print(vvv, type(vvv))

        card_dict = self.dict_key_stringify(card_dict)
        params_dict = {
            "query_text": self.query_text,
            "query_meta": str(self.query_meta),
            "card_dict": card_dict,
            "subquery_missing": subquery_missing,
            "single_table_missing": single_table_missing
        }

        return params_dict

    def upload_plan_instance(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        self.valid_eval()

        params_dict = self.construct_request_dict([], [])
        # params_join_str = json.dumps(params_dict, default=convert_to_serializable)
        local_url = p_join(self.url, "save_instance")
        result = requests.post(url=local_url, json=params_dict, timeout=10)
        # print(result.json())

        return result.json()

    def make_value_sampling(self, subquery_missing, single_table_missing):
        """
        {Description}

        Args:
            subquery_missing:
            single_table_missing:
        Returns:
            subquery_res:
            single_table_res:
        """
        self.valid_eval()
        params_dict = self.construct_request_dict(\
            subquery_missing, single_table_missing)
        local_url = p_join(self.url, "cardinality_inference")

        result = requests.post(url=local_url, json=params_dict, timeout=10)
        reply_dict = result.json()
        flag = reply_dict['flag']

        if flag == False:
            # print("make_value_sampling: flag = False.")
            return {}, {}
        else:
            subquery_res, single_table_res = reply_dict['result']
            subquery_res, single_table_res = \
                self.dict_key_parse(subquery_res, single_table_res)
            # print("make_value_sampling: flag = True. subquery_res = "\
            #       f"{subquery_res}. single_table_res = {single_table_res}.")
            return subquery_res, single_table_res

    @utils.timing_decorator
    def distribution_based_generation(self, num, mean, varience, dist_func = normal_dist):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        return [dist_func(mean, varience) for _ in range(num)]

# %%

def get_strategy(option_name):
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    return common_config.option_collections[option_name]['strategy']

# %%


class MixEstimator(builtin_card_estimation.BaseEstimator):
    """
    混合估计器，在external调用失败的时候，使用built-in方法

    Members:
        field1:
        field2:
    """

    def __init__(self, workload, url = "http://101.6.96.160:30007", \
            model_type = "gcn", internal_type = "graph_corr_based"):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        super().__init__(workload)
        self.external_caller = ExternalEstimator(workload, url, model_type)

        if internal_type == "graph_corr_based":
            self.internal_caller = builtin_card_estimation.GraphCorrBasedEstimator(\
                workload, get_strategy('graph_corr_based'))
        elif internal_type == "equal_diff":
            self.internal_caller = builtin_card_estimation.BuiltinEstimator(\
                workload, get_strategy('equal_diff'))
        elif internal_type == "equal_ratio":
            self.internal_caller = builtin_card_estimation.BuiltinEstimator(\
                workload, get_strategy('equal_ratio'))

    @utils.timing_decorator
    def make_value_sampling(self, subquery_missing, single_table_missing):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        flag, subquery_res, single_table_res = \
            self.request_external(subquery_missing, single_table_missing)

        if flag == False:
            # 远程调用失效
            subquery_res, single_table_res = \
                self.request_internal(subquery_missing, single_table_missing)
            # print("make_value_sampling: apply internal result.")
        else:
            # print("make_value_sampling: apply external result.")
            pass
        self.result_validation(subquery_missing, single_table_missing,
                               subquery_res, single_table_res)
        return subquery_res, single_table_res
        
    def request_external(self, subquery_missing, single_table_missing):
        """
        向external caller发送请求，获得结果

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        subquery_res, single_table_res = self.external_caller.\
            make_value_sampling(subquery_missing, single_table_missing)
        
        if subquery_res == {} and single_table_res == {}:
            return False, subquery_res, single_table_res
        else:
            return True, subquery_res, single_table_res

    def request_internal(self, subquery_missing, single_table_missing):
        """
        向internal caller发送请求，获得结果
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        return self.internal_caller.make_value_sampling(\
            subquery_missing, single_table_missing)
    
    def set_instance(self, query_text, query_meta):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        self.internal_caller.set_instance(query_text, query_meta)
        self.external_caller.set_instance(query_text, query_meta)

    def set_existing_card_dict(self, subquery_true, single_table_true, \
            subquery_estimation, single_table_estimation):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        self.internal_caller.set_existing_card_dict(subquery_true, single_table_true, \
            subquery_estimation, single_table_estimation)
        self.external_caller.set_existing_card_dict(subquery_true, single_table_true, \
            subquery_estimation, single_table_estimation)

    def upload_plan_instance(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        self.external_caller.upload_plan_instance()

# %%