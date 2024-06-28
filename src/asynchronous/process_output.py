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
from asynchronous import construct_input

# %%

# def get_async_result(out_path):
#     return utils.load_json(out_path)
# %%

# def get_async_cardinalities(out_path):
#     """
#     {Description}
    
#     Args:
#         arg1:
#         arg2:
#     Returns:
#         res1:
#         res2:
#     """
#     # out_path = construct_input.get_output_path(query_signature)
#     return utils.load_json(out_path)["cardinality"]


# %%
def get_async_proc_cpu_time(out_path):
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    while True:
        # 考虑有json文件写一半的情况
        proc_mapping = utils.load_json(out_path)
        if proc_mapping is not None:
            break
        time.sleep(0.1)

    try:
        res_dict = {}
        # return proc_mapping["proc_time"]
        for k, v in proc_mapping["proc_time"].items():
            res_dict[int(k)] = v
        return res_dict
    except KeyError:
        return {}
    
# %%

def get_async_proc_mapping(out_path):
    """
    获得异步进程的信息
    
    Args:
        arg1:
        arg2:
    Returns:
        subquery_mapping: 
        single_table_mapping:
    """
    load_try_times = 10
    while True:
        # 考虑有json文件写一半的情况
        # print("get_async_proc_mapping: out_path = {}.".format(out_path))
        try:
            proc_mapping = utils.load_json(out_path)
        except json.decoder.JSONDecodeError:
            proc_mapping = None

        if proc_mapping is not None:
            break
        else:
            time.sleep(0.01)
            load_try_times -= 1
            if load_try_times == 0:
                print("get_async_proc_mapping: load json error. out_path = {}".\
                      format(out_path))
                raise FileExistsError("json file cannot be parsed")

    subquery_mapping = {}
    single_table_mapping = {}

    for k, v in proc_mapping.items():
        k = str(k)
        # print("value_k = {}. type_k = {}.".format(k, type(k)))
        if k.startswith("("):
            # subquery case
            k = eval(k)
            subquery_mapping[k] = int(v)
        elif k == "proc_time":
            # 记录进程时间的情况
            continue
        else:
            # single_table case
            single_table_mapping[k] = int(v)

    return subquery_mapping, single_table_mapping

# %%

def get_async_duration(out_path):
    """
    获得每一个异步查询的执行时间
    
    Args:
        out_path:
    Returns:
        subquery_out:
        single_table_out:
    """
    subquery_out, single_table_out = {}, {}

    max_try_times = 3
    while True:
        try:
            with open(out_path, "r") as f_in:
                line_list = f_in.read().splitlines()
            break
        except FileNotFoundError as e:
            max_try_times -= 1
            if max_try_times <= 0:
                print(f"get_async_duration: FileNotFoundError. out_path = {out_path}.")
                # raise e
                return {}, {}   # 返回两个空的字典
            time.sleep(0.1)

    for line in line_list:
        # 解析文件的内容并填入dict
        try:
            key, _, _, duration = line.split("#")
        except ValueError as e:
            print("get_async_duration: line = {}.".format(line))
            raise(e)
        
        key = str(key)

        # 需要转换成浮点型的数据
        if key.startswith("("):
            # subquery case
            key = eval(key)
            subquery_out[key] = float(duration)
        else:
            # single_table case
            single_table_out[key] = float(duration)

    return subquery_out, single_table_out

# %%

def get_async_cardinalities(out_path):
    """
    out_path在这里对应一个NodeExtension
    
    Args:
        out_path:
    Returns:
        subquery_out:
        single_table_out:
    """
    subquery_out, single_table_out = {}, {}

    with open(out_path, "r") as f_in:
        line_list = f_in.read().splitlines()

    for line in line_list:
        # 解析文件的内容并填入dict
        try:
            key, query, card, duration = line.split("#")
        except ValueError as e:
            print(f"get_async_cardinalities: line = {line}.")
            raise e
        key = str(key)

        # 需要转换成整型的数据
        if key.startswith("("):
            # subquery case
            key = eval(key)
            subquery_out[key] = int(card)
        else:
            # single_table case
            single_table_out[key] = int(card)

    return subquery_out, single_table_out

# %%
