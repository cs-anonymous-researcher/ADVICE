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

from workload import physical_plan_info
from query import query_construction, query_exploration
from utility import workload_parser, common_config
from typing import List
from data_interaction import connection_parallel

# %% 重要的工具方法

def plan_evaluation_under_cardinality(workload, query_ctrl, query_meta, \
    subquery_dict, single_table_dict) -> physical_plan_info.PhysicalPlan:
    """ 
    基数注入获得Plan的函数

    Args:
        workload:
        query_ctrl:
        query_meta:
        subquery_dict:
        single_table_dict:
        
    Returns:
        curr_physical_info:
        return2:
    """
    query_text = query_construction.construct_origin_query(\
        query_meta, workload = workload)

    query_ctrl.set_query_instance(query_text = query_text, query_meta = query_meta)
    res_plan = query_ctrl.get_plan_by_external_card(subquery_dict, single_table_dict)  # 获得注入基数的查询

    curr_physical_info = physical_plan_info.PhysicalPlan(query_text = \
        query_text, plan_dict = res_plan)   # 物理信息
    return curr_physical_info


# %%
def plan_evaluation_under_cardinality_parallel(workload, query_list, meta_list, \
    subquery_dict_list, single_table_dict_list) -> List[physical_plan_info.PhysicalPlan]:
    """ 
    基数注入获得Plan的函数

    Args:
        workload:
        query_ctrl:
        query_meta:
        subquery_dict:
        single_table_dict:
        
    Returns:
        plan_list:
        return2:
    """
    # query_text = query_construction.construct_origin_query(\
    #     query_meta, workload = workload)

    # query_ctrl.set_query_instance(query_text = query_text, query_meta = query_meta)
    # res_plan = query_ctrl.get_plan_by_external_card(subquery_dict, single_table_dict)  # 获得注入基数的查询

    # curr_physical_info = physical_plan_info.PhysicalPlan(query_text = \
    #     query_text, plan_dict = res_plan)   # 物理信息

    conn_pool: connection_parallel.ConnectionPool = connection_parallel.get_conn_pool_by_workload(workload)
    plan_dict_list = conn_pool.get_plan_under_card_parallel(query_list, subquery_dict_list, single_table_dict_list)

    assert len(query_list) == len(meta_list) == len(plan_dict_list)
    physical_plan_list = [physical_plan_info.PhysicalPlan(query_text, res_plan) 
        for query_text, res_plan in zip(query_list, plan_dict_list)]
    
    # return curr_physical_info
    return physical_plan_list

# %%
def dict_complement(old_dict, new_dict):
    """
    {Description}
    
    Args:
        old_dict:
        new_dict:
    Returns:
        merged_dict:
    """
    for k, v in new_dict.items():
        old_dict[k] = v
    return old_dict


def dict_merge(old_dict, new_dict):
    """
    字典合并，返回一个新的对象
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    out_dict = {**old_dict, **new_dict}
    return out_dict

def dict_make(key_list, value_list):
    """
    {Description}

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    res_dict = {}
    for k, v in zip(key_list, value_list):
        res_dict[k] = v
    return res_dict



def dict_difference(old_dict, new_dict):
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    res_dict = {}
    for k, v in old_dict.items():
        if k not in new_dict.keys():
            res_dict[k] = v

    return res_dict


# %% 把node_extension的函数转移过来
def plan_comparison(plan1:physical_plan_info.PhysicalPlan, plan2:physical_plan_info.PhysicalPlan, \
                    subquery_dict: dict, single_table_dict: dict):
    """
    比较估计基数的计划和混合基数的计划，以及混合基数下两者的差值

    Args:
        plan1: 混合/真实基数下的查询计划
        plan2: 估计基数下的查询计划
        subquery_dict: 混合/真实子查询基数
        single_table_dict: 混合/真实单表基数
    Returns:
        flag: True代表查询计划等价，False代表查询计划不等价
        cost1: 第一个查询计划在真实基数下的代价
        cost2: 第二个查询计划在真实基数下的代价
    """
    # print(plan1.leading, plan2.leading)
    # print(plan1.join_ops, plan2.join_ops)
    # print(plan1.scan_ops, plan2.scan_ops)

    flag, error_dict = physical_plan_info.physical_comparison((plan1.leading, \
        plan1.join_ops, plan1.scan_ops), (plan2.leading, plan2.join_ops, plan2.scan_ops))  # 调用比较物理计划的函数
    if flag == True:
        print("两个查询计划等价，探索失败")
        cost1 = plan1.get_plan_cost(subquery_dict, single_table_dict)
        cost2 = plan2.get_plan_cost(subquery_dict, single_table_dict)
        return flag, cost1, cost2
    else:
        cost1 = plan1.get_plan_cost(subquery_dict, single_table_dict)
        cost2 = plan2.get_plan_cost(subquery_dict, single_table_dict)
        print("两个查询计划不等价，探索成功. cost1 = {}. cost2 = {}.".format(cost1, cost2))
        return flag, cost1, cost2
    
# %%
def get_diff_keys(subquery_estimation, single_table_estimation, \
                  subquery_true, single_table_true):
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    subquery_diff_keys, single_table_diff_keys = [], []

    for k in subquery_estimation.keys():
        if k not in subquery_true.keys():
            subquery_diff_keys.append(k)
    
    for k in single_table_estimation.keys():
        if k not in single_table_true.keys():
            single_table_diff_keys.append(k)

    return subquery_diff_keys, single_table_diff_keys


# %%
def get_diff_queries(query_parser, subquery_diff_keys, single_table_diff_keys):
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    subquery_diff_queries, single_table_diff_queries = [], []

    for alias_list in subquery_diff_keys:
        local_query = query_parser.construct_PK_FK_sub_query(alias_list=alias_list)
        subquery_diff_queries.append(local_query)

    for alias in single_table_diff_keys:
        local_query = query_parser.get_single_table_query(alias=alias)
        single_table_diff_queries.append(local_query)
        # print("alias = {}. local_query = {}.".format(alias, local_query))

    return subquery_diff_queries, single_table_diff_queries

def get_diff_metas(query_parser: workload_parser.SQLParser, subquery_diff_keys, single_table_diff_keys):
    """
    {Description}

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    subquery_diff_metas, single_table_diff_metas = [], []

    for alias_list in subquery_diff_keys:
        # local_meta = query_parser.construct_PK_FK_sub_query(alias_list=alias_list)
        local_meta = query_parser.generate_subquery_meta(alias_list=alias_list)
        subquery_diff_metas.append(local_meta)

    for alias in single_table_diff_keys:
        # local_meta = query_parser.get_single_table_query(alias=alias)
        local_meta = query_parser.generate_subquery_meta(alias_list=[alias,])
        single_table_diff_metas.append(local_meta)

    return subquery_diff_metas, single_table_diff_metas

# %%    

def get_diff_cardinalities_with_hint(get_cardinalities_func, subquery_diff_queries, \
        single_table_diff_queries, subquery_diff_hint, time_limit = None):
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    print("get_diff_cardinalities_with_hint: time_limit = {}.".format(time_limit))

    ts = time.time()
    if time_limit is not None:
        # 只认为多表情况下会有问题
        subquery_diff_cardinalities = get_cardinalities_func(\
            sql_list=subquery_diff_queries, hint_list = subquery_diff_hint, timeout=time_limit)
    else:
        subquery_diff_cardinalities = get_cardinalities_func(\
            sql_list=subquery_diff_queries, hint_list = subquery_diff_hint)

    # 暂不考虑单表的时间限制
    single_table_diff_cardinalities = get_cardinalities_func(sql_list=single_table_diff_queries)
    te = time.time()

    print("get_diff_cardinalities_with_hint: delta time = {}.".format(te - ts))
    return subquery_diff_cardinalities, single_table_diff_cardinalities



def get_diff_cardinalities(get_cardinalities_func, subquery_diff_queries, \
        single_table_diff_queries, time_limit = None):
    """
    {Description}
    
    Args:
        subquery_diff_queries:
        single_table_diff_queries:
        get_cardinalities_func: 获得真实基数的函数
        time_limit: 
    Returns:
        res1:
        res2:
    """
    # print("get_diff_cardinalities: time_limit = {}.".format(time_limit))

    ts = time.time()
    if time_limit is not None:
        # 只认为多表情况下会有问题
        subquery_diff_cardinalities = get_cardinalities_func(subquery_diff_queries, timeout=time_limit)
    else:
        subquery_diff_cardinalities = get_cardinalities_func(subquery_diff_queries)

    # 暂不考虑单表的时间限制
    single_table_diff_cardinalities = get_cardinalities_func(single_table_diff_queries)
    te = time.time()

    # print("get_diff_cardinalities: delta time = {:.3f}.".format(te - ts))
    return subquery_diff_cardinalities, single_table_diff_cardinalities

# %% 

def parse_missing_card(query_parser, subquery_ref, single_table_ref, subquery_missing, single_table_missing, out_mode="query"):
    """
    {Description}
    
    Args:
        query_parser: 
        subquery_ref: 
        single_table_ref: 
        subquery_missing: 
        single_table_missing: 
        out_mode:
    Returns:
        subquery_diff_dict: 
        singe_table_diff_dict:
    """
    # print("parse_missing_card: subquery_ref = {}. single_table_ref = {}".\
    #       format(subquery_ref, single_table_ref))
    # print("parse_missing_card: subquery_missing = {}. single_table_missing = {}".\
    #       format(subquery_missing, single_table_missing))

    # 获得真实基数缺省的keys
    subquery_diff_keys, singe_table_diff_keys = get_diff_keys(subquery_estimation=subquery_ref,
        single_table_estimation=single_table_ref, subquery_true=subquery_missing, single_table_true=single_table_missing)

    # print("parse_missing_card: len(subquery_diff_keys) = {}. len(singe_table_diff_keys) = {}. len(subquery_ref) = {}. len(single_table_ref) = {}. len(subquery_missing) = {}. len(single_table_missing) = {}..".\
    #       format(len(subquery_diff_keys), len(singe_table_diff_keys), len(subquery_ref), len(single_table_ref), len(subquery_missing), len(single_table_missing)))

    if out_mode == "query" or out_mode == "both":
        # 获得对应的子查询和单表的queries
        subquery_diff_queries, single_table_diff_queries = get_diff_queries(query_parser=query_parser,\
            subquery_diff_keys=subquery_diff_keys, single_table_diff_keys=singe_table_diff_keys)
    
    if out_mode == "meta" or out_mode == "both":
        subquery_diff_metas, single_table_diff_metas = get_diff_metas(query_parser=query_parser,\
            subquery_diff_keys=subquery_diff_keys, single_table_diff_keys=singe_table_diff_keys)
    
    if out_mode == "query":
        subquery_diff_dict = dict_make(subquery_diff_keys, subquery_diff_queries)
        singe_table_diff_dict = dict_make(singe_table_diff_keys, single_table_diff_queries)
    elif out_mode == "meta":
        subquery_diff_dict = dict_make(subquery_diff_keys, subquery_diff_metas)
        singe_table_diff_dict = dict_make(singe_table_diff_keys, single_table_diff_metas)    
    elif out_mode == "both":
        subquery_diff_dict = dict_make(subquery_diff_keys, zip(subquery_diff_queries, subquery_diff_metas))
        singe_table_diff_dict = dict_make(singe_table_diff_keys, zip(single_table_diff_queries, single_table_diff_metas))
    else:
        raise ValueError("parse_missing_card: Unsupported out_mode({})".format(out_mode))

    return subquery_diff_dict, singe_table_diff_dict
    
# %%

# 参数变量名需要根据实际要求替换
# def complement_missing_card(query_parser, subquery_true, single_table_true, \
#     subquery_estimation, single_table_estimation, get_cardinalities_func, time_limit):

def get_subset_true_card(query_key, subquery_true, single_table_true):
    """
    {Description}

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    if isinstance(query_key, str):
        # single_table的情况，直接跳过
        return {}
    elif isinstance(query_key, tuple):
        # subquery的情况
        result_dict = {
            "subquery": {},
            "single_table": {}
        }
        for k, v in subquery_true.items():
            if len(set(k).union(query_key)) == len(set(query_key)):
                # k完全包含在query_key中的情况
                result_dict['subquery'][k] = v

        for k, v in single_table_true.items():
            if k in query_key:
                result_dict['single_table'][k] = v

        return result_dict


def complement_missing_card_with_hint(query_parser, subquery_ref, single_table_ref, \
        subquery_missing, single_table_missing, get_cardinalities_func, time_limit):
    """
    {Description}
    
    Args:
        query_key:
        subquery_true:
        single_table_true:
    Returns:
        card_dict:
        res2:
    """
    # 复制保存已有的真实基数
    subquery_origin, single_table_origin = \
        deepcopy(subquery_missing), deepcopy(single_table_missing)

    # 获得真实基数缺省的keys
    subquery_diff_keys, singe_table_diff_keys = get_diff_keys(subquery_estimation=subquery_ref,
        single_table_estimation=single_table_ref, subquery_true=subquery_missing, single_table_true=single_table_missing)

    # 获得对应的子查询和单表的queries
    subquery_diff_queries, single_table_diff_queries = get_diff_queries(query_parser=query_parser,\
        subquery_diff_keys=subquery_diff_keys, single_table_diff_keys=singe_table_diff_keys)
    
    # 获得对应的基数
    subquery_diff_hint = [get_subset_true_card(key, subquery_missing, \
        single_table_missing) for key in subquery_diff_keys]
    
    subquery_diff_cardinalities, single_table_diff_cardinalities = get_diff_cardinalities_with_hint(
        get_cardinalities_func, subquery_diff_queries, single_table_diff_queries, subquery_diff_hint, time_limit)


    subquery_diff_dict = dict_make(subquery_diff_keys, subquery_diff_cardinalities)
    singe_table_diff_dict = dict_make(singe_table_diff_keys, single_table_diff_cardinalities)

    subquery_full = dict_complement(subquery_origin, subquery_diff_dict)
    single_table_full = dict_complement(single_table_origin, singe_table_diff_dict)

    return subquery_full, single_table_full


def complement_missing_card(query_parser, subquery_ref, single_table_ref, \
        subquery_missing, single_table_missing, get_cardinalities_func, time_limit):
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    # 复制保存已有的真实基数
    subquery_origin, single_table_origin = \
        deepcopy(subquery_missing), deepcopy(single_table_missing)

    # 获得真实基数缺省的keys
    subquery_diff_keys, singe_table_diff_keys = get_diff_keys(subquery_estimation=subquery_ref,
        single_table_estimation=single_table_ref, subquery_true=subquery_missing, single_table_true=single_table_missing)

    # print(f"complement_missing_card: subquery_diff_keys = {subquery_diff_keys}")
    # print(f"complement_missing_card: singe_table_diff_keys = {singe_table_diff_keys}")

    # 获得对应的子查询和单表的queries
    subquery_diff_queries, single_table_diff_queries = get_diff_queries(query_parser=query_parser,\
        subquery_diff_keys=subquery_diff_keys, single_table_diff_keys=singe_table_diff_keys)
    
    # 获得对应的基数
    subquery_diff_cardinalities, single_table_diff_cardinalities = get_diff_cardinalities(get_cardinalities_func = get_cardinalities_func,
        subquery_diff_queries=subquery_diff_queries, single_table_diff_queries=single_table_diff_queries, time_limit = time_limit)

    subquery_diff_dict = dict_make(subquery_diff_keys, subquery_diff_cardinalities)
    singe_table_diff_dict = dict_make(singe_table_diff_keys, single_table_diff_cardinalities)

    subquery_full = dict_complement(subquery_origin, subquery_diff_dict)
    single_table_full = dict_complement(single_table_origin, singe_table_diff_dict)

    return subquery_full, single_table_full


# %%

def complement_estimation_card(query_parser, subquery_estimation, single_table_estimation, \
    subquery_ref, single_table_ref, get_cardinalities_func):
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    # print("complement_estimation_card: no time_limit")
    return complement_missing_card(query_parser, subquery_ref, single_table_ref, \
        subquery_estimation, single_table_estimation, get_cardinalities_func, time_limit = None)

# %%

def complement_true_card_with_hint(query_parser, subquery_true, single_table_true, \
        subquery_estimation, single_table_estimation, get_cardinalities_func, time_limit):
    """
    {Description}

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    return complement_missing_card_with_hint(query_parser, subquery_estimation, \
        single_table_estimation, subquery_true, single_table_true, get_cardinalities_func, \
        time_limit)

def complement_true_card(query_parser, subquery_true, single_table_true, \
    subquery_estimation, single_table_estimation, get_cardinalities_func, time_limit):
    """
    完成真实基数的补全
    
    Args:
        time_limit: 
        arg2:
    Returns:
        subquery_true: 真实子查询基数的字典
        single_table_true: 真实单表基数的字典
    """
    return complement_missing_card(query_parser = query_parser, subquery_ref=subquery_estimation, \
        single_table_ref = single_table_estimation, subquery_missing = subquery_true, \
        single_table_missing = single_table_true, get_cardinalities_func=get_cardinalities_func, \
        time_limit = time_limit)

# %%
def invalid_evaluation(card_dict):
    """
    判断是否会出现None的结果(主要是由超时的Query衍生而来)

    Args:
        card_dict:
    Returns:
        flag:
    """
    flag = True
    for k, v in card_dict.items():
        if v is None:
            return False
    return flag

