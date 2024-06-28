#!/usr/bin/env python
# -*- coding: utf-8 -*-


# %%

import numpy as np
import pandas as pd
from os.path import join as p_join
from matplotlib import pyplot as plt
import json, os, time, re, math
import pickle
from utilities import global_config

# %%

def load_json(data_path):
    '''
    根据数据文件加载JSON
    '''
    try:
        with open(data_path, "r") as f_in:
            res_dict = json.load(f_in)
        return res_dict
    except FileNotFoundError as e:
        print("load_json: file not found\ndata_path = {}".format(data_path))
        return {}   # 找不到文件，返回一个空的字典    
    except json.decoder.JSONDecodeError as e:
        print("load_json: file decode error\ndata_path = {}".format(data_path))
        return {}

def dump_json(res_dict, data_path):
    '''
    将文件以JSON格式导出
    '''
    try:
        with open(data_path, "w+") as f_out:
            json.dump(res_dict, f_out, indent=4, ensure_ascii=False)
    except FileNotFoundError as e:
        print("save_json: file not found\ndata_path = {}".format(data_path))
        return None


def load_pickle(data_path):
    """
    加载pickle对象
    
    Args:
        data_path:
    Returns:
        res_obj:
    """
    try:
        with open(data_path, "rb") as f_in:
            res_obj = pickle.load(f_in)
        return res_obj
    except FileNotFoundError as e:
        print("load_pickle: file not found\ndata_path = {}".format(data_path))
        return None   # 找不到文件，返回一个空的字典

def dump_pickle(res_obj, data_path):
    """
    导出pickle对象
    
    Args:
        res_obj:
        data_path:
    Returns:
        None
    """
    try:
        with open(data_path, "wb+") as f_out:
            pickle.dump(res_obj, f_out)
    except FileNotFoundError as e:
        print("save_json: file not found\ndata_path = {}".format(data_path))
        return None


JOB_alias = {
    "cast_info": "ci", "movie_keyword": "mk",
    "movie_companies": "mc", "company_type": "ct",
    "title": "t", "company_name": "cn",
    "keyword": "k", "movie_info_idx": "mi_idx",
    "info_type": "it", "movie_info": "mi"
}

STATS_alias = {
    "posts": "p", "votes": "v", "badges": "b",
    "users": "u", "posthistory": "ph",
    "postlinks": "pl", "tags": "t", "comments": "c"
}

RELEASE_alias = {
    "release": "r", "medium": "m", "release_country": "rc", 
    "release_tag": "rt", "release_label": "rl", "release_group": "rg",
    "artist_credit": "ac", "release_meta": "rm"
}

DSB_alias = {
    "store_sales": "ss", "catalog_sales": "cs", "web_sales": "ws",
    "store_returns": "sr", "catalog_returns": "cr",
    "web_returns": "wr", "customer_demographics": "cd",
    "customer": "c", "customer_address": "ca"
}
    
workload_alias_option = {
    "job": JOB_alias,
    "stats": STATS_alias,
    "release": RELEASE_alias,
    "dsb": DSB_alias
}

# %%

def list_index(value_list, index_list):
    # print("list_index: value_list = {}. index_list = {}.".\
    #       format(value_list, index_list))
    return [value_list[i] for i in index_list]

# %%

def create_data_instance(query_meta, card_dict):
    """
    创建数据实例

    Args:
        query_meta:
        card_dict:
    Returns:
        data:
        return2:
    """
    data = None

    return data

# %% card_dict相关处理函数
from copy import deepcopy

def extract_card_info(card_dict, dict_copy = False):
    """
    {Description}

    Args:
        card_dict:
        arg2:
    Returns:
        subquery_true:
        single_table_true:
        subquery_estimation:
        single_table_estimation:
    """
    try:
        subquery_true, single_table_true = \
            card_dict['true']['subquery'], card_dict['true']['single_table']
        subquery_estimation, single_table_estimation = \
            card_dict['estimation']['subquery'], card_dict['estimation']['single_table']
    except Exception as e:
        print(f"extract_card_info: meet Error. card_dict = {card_dict}.")
        raise e

    if dict_copy == True:
        return deepcopy(subquery_true), deepcopy(single_table_true), \
            deepcopy(subquery_estimation), deepcopy(single_table_estimation)
    else:
        return subquery_true, single_table_true, \
            subquery_estimation, single_table_estimation

def pack_card_info(subquery_true, single_table_true, \
        subquery_estimation, single_table_estimation, dict_copy = False):
    """
    {Description}

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """

    if dict_copy == True:
        out_card_dict = {
            "true": {
                "subquery": deepcopy(subquery_true),
                "single_table": deepcopy(single_table_true)
            },
            "estimation": {
                "subquery": deepcopy(subquery_estimation),
                "single_table": deepcopy(single_table_estimation)
            }
        }
    else:
        out_card_dict = {
            "true": {
                "subquery": subquery_true,
                "single_table": single_table_true
            },
            "estimation": {
                "subquery": subquery_estimation,
                "single_table": single_table_estimation
            }
        }

    return out_card_dict

def dict_merge(dict1, dict2):
    res_dict = {}
    for k, v in dict1.items():
        res_dict[k] = v
    for k, v in dict2.items():
        res_dict[k] = v
    return res_dict

def dict_apply(in_dict: dict, operation_func, mode = "value"):
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    out_dict = {}
    assert mode in ["key", "value"], f"dict_apply: mode = {mode}"
    for k, v in in_dict.items():
        if mode == "value":
            out_dict[k] = operation_func(v)
        elif mode == "key":
            out_dict[operation_func(k)] = (v)
    return out_dict


def get_feature_number(workload, mode = "both"):
    """
    {Description}

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    feature_num_path = p_join(global_config.data_root, "feature_number.json")
    # feature_num_path = "/home/jinly/GNN_Estimator/data/feature_number.json"
    feature_num_dict = load_json(feature_num_path)
    feature_num = feature_num_dict[workload]

    print(f"get_feature_number: workload = {workload}. feature_num = {feature_num}.")
    assert mode in ("both", "query-only"), "get_feature_number: available modes = [both, query-only]"
    if mode == "both":
        return feature_num
    elif mode == "query-only":
        return feature_num - 3

# %%


def tuple_in(in_tuple, elem):
    """
    检测elem是否包含在in_tuple中
    
    Args:
        in_tuple:
        elem: 可以是list/tuple型的，也可以是单个元素
    Returns:
        flag:
    """
    if isinstance(elem, (tuple, list)):
        # 考虑elem为tuple/list的情况
        flag = True
        for e in elem:
            if e not in in_tuple:
                flag = False
                break
    else:
        # 考虑单个元素的情况
        flag = elem in in_tuple

    return flag

# %%
