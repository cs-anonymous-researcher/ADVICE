#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from os.path import join as p_join
from matplotlib import pyplot as plt
import json, os, time, re, math
import hashlib
from utility.workload_spec import equivalent_stats, abbr_option
from itertools import takewhile
from typing import Dict
from copy import deepcopy

DEBUG = False
# DEBUG = True

def list_index(value_list, index_list):
    # print("list_index: value_list = {}. index_list = {}.".\
    #       format(value_list, index_list))
    return [value_list[i] for i in index_list]

def list_index_batch(list_batch, index_list):
    """
    {Description}

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    return [list_index(item_list, index_list) for item_list in list_batch]

def debug(*args, **kwargs):
    if DEBUG == True:
        print(*args, **kwargs)
    else:
        return

def deterministic_hash(string):
    return int(hashlib.sha1(str(string).encode("utf-8")).hexdigest(), 16)

def point2range(point_list):
    # 从一个点到一个范围
    range_list = []
    for p in point_list:
        range_list.append((p, p + 1))
    return range_list

def point2range_batch(point_list_batch):
    return [point2range(point_list) for point_list in point_list_batch]


def result_decouple(res):
    # 将结果进行解绑
    param_list, label_list = [], []
    for arr in res:
        param_list.append(arr[:-1])
        label_list.append(arr[-1])

    return param_list, label_list


def array2index(in_arr:np.ndarray):
    # 将array转成index
    assert len(in_arr) % 2 == 0
    idx_res = []
    for i in range(len(in_arr)//2):
        idx_res.append(slice(in_arr[2*i], in_arr[2*i+1]))
    return tuple(idx_res)


def restore_origin_label(value_cnt_arr, param_list, mode = "range"):
    if mode == "range":
        idx_list = [array2index(in_arr) for in_arr in param_list]
        return [int(np.sum(value_cnt_arr[tuple(idx)])) for idx in idx_list]    # 涉及到求和的操作
    elif mode == "point":
        idx_list = param_list
        return [int(value_cnt_arr[tuple(idx)]) for idx in idx_list]


# %%
def restore_label_by_pair_list(value_cnt_arr, index_pair_list):
    arr_idx = []
    for pair in index_pair_list:
        if pair is None:
            arr_idx.append(Ellipsis)
        else:
            arr_idx.append(slice(*pair))
    print("function: restore_label_by_pair_list")
    print("index_pair_list = {}.".format(index_pair_list))
    print("arr_idx = {}.".format(arr_idx))
    print("value_cnt_sub = {}.".format(value_cnt_arr[tuple(arr_idx)]))
    return int(np.sum(value_cnt_arr[tuple(arr_idx)]))

# %% stats相关信息

# equivalent_stats = {
#     "users": ["badges", "votes"], 
#     "posts": ["tags", "postlinks", "posthistory", "comments"]
# }

# 通用的对象缓存方法
import pickle, hashlib, shutil

def item_signature(obj):
    """
    {Description}
    
    Args:
        obj:
    Returns:
        hash:
    """
    if isinstance(obj, np.ndarray):
        data_view = np.ascontiguousarray(obj).view(np.uint8)
        array_hash = hashlib.sha1(data_view).hexdigest()
        return array_hash
    elif isinstance(obj, (list, tuple)):
        h = hashlib.blake2b()
        for item in obj:
            h.update(bytes(item_signature(item), encoding='utf8'))
        list_hash = h.hexdigest()
        return list_hash
    else:
        obj_hash = hashlib.sha1(bytes(obj)).hexdigest()
        return obj_hash

def objects_signature(objects):
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    seg_length = 8
    sig_list = []
    if isinstance(objects, (tuple, list)) == False:
        raise TypeError("Unsupported objects type: {}".format(type(objects)))
        
    for obj in objects:
        sig_list.append(item_signature(obj))

    return "".join(map(lambda a: a[:seg_length], sig_list))

def general_cache_load(cache_dir, signature):
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    data_path = p_join(cache_dir, "{}.pkl".format(signature))

    if os.path.isfile(data_path) == True:
        with open(data_path, "rb") as f_in:
            return True, pickle.load(f_in)
    else:
        return False, None


def general_cache_dump(cache_dir, signature, objects, overwrite = False):
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    data_path = p_join(cache_dir, "{}.pkl".format(signature))

    if overwrite == True or os.path.isfile(data_path) == False:
        with open(data_path, "wb") as f_out:
            pickle.dump(objects, f_out)
        return True
    else:
        return False

def meta_assemble(table_abbr, column_names, val_pair_list):
    """
    包装meta的相关信息
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    
    filter_list = []
    if isinstance(table_abbr, str):
        table_abbr = [table_abbr for _ in column_names]
    # else:
    for abbr, col, pair in zip(table_abbr, column_names, val_pair_list):
        filter_list.append((abbr, col, pair[0], pair[1]))
        # filter_list.append(())
    return filter_list


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


def share_prefix(string_list):
    return ''.join(c[0] for c in takewhile(lambda x: all(x[0] == y for y in x), zip(*string_list)))


def prefix_add(column_list, abbr):
    column_res = []
    existing_prefix = share_prefix(column_list)
    if existing_prefix.startswith("{}_".format(abbr)):
        print("Warning! 当前的前缀已经存在了")
        return column_list

    for col in column_list:
        column_res.append("{}_{}".format(abbr, col))

    return column_res

    
def prefix_detection(data_df, column_list, abbr = None):
    """
    {Description}
    
    Args:
        data_df:
        column_list:
        abbr
    Returns:
        column_res:
    """
    if abbr is None:
        # 自动提取前缀
        prefix = share_prefix(data_df.columns)
    else:
        prefix = "{}_".format(abbr)

    prefix_cnt = 0
    for col in column_list:
        if col.startswith(prefix):
            prefix_cnt += 1

    column_res = []
    if prefix_cnt == len(column_list):
        for col in column_list:
            column_res.append(col)
    else:
        if prefix_cnt != 0:
            pass
        for col in column_list:
            column_res.append("{}{}".format(prefix, col))
    
    return column_res   # 返回带前缀的列


def dict_merge(dict1, dict2):
    res_dict = {}
    for k, v in dict1.items():
        res_dict[k] = v
    for k, v in dict2.items():
        res_dict[k] = v
    return res_dict

def dict2list(in_dict):
    """
    {Description}
    
    Args:
        in_dict:
        arg2:
    Returns:
        list_key: 
        list_value:
    """
    list_key, list_value = [], []
    for k, v in in_dict.items():
        list_key.append(k)
        list_value.append(v)

    return list_key, list_value

def result_print(error_list):
    """
    打印错误结果
    
    Args:
        error_list:
    Returns:
        median:
        quantile_90:
        max:
    """
    val_list = list(zip(*error_list))[1]
    print(np.median(val_list))
    print(np.quantile(val_list, 0.9))
    print(np.max(val_list))

    return np.median(val_list), \
        np.quantile(val_list, 0.9), np.max(val_list)

# %% tuple相关操作的封装

def tuple_delete(in_tuple, elem):
    """
    {Description}
    
    Args:
        in_tuple:
        elem:
    Returns:
        out_tuple:
    """
    if isinstance(elem, str):
        out_set = set(in_tuple).difference(set([elem,]))
    else:
        out_set = set(in_tuple).difference(set(elem))

    return tuple(sorted(list(out_set)))

def tuple_add(in_tuple, elem):
    """
    {Description}
    
    Args:
        in_tuple:
        elem:
    Returns:
        out_tuple:
    """
    if isinstance(elem, str):
        out_set = in_tuple + (elem,)
    else:
        out_set = in_tuple + tuple(elem)

    return tuple(sorted(list(out_set)))

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

def tuple_single(elem):
    # 将一个元素转化成元组
    return (elem,)

def tuple2str(in_tuple):
    # 将元组转化成字符串
    return " ".join(in_tuple)


# %%
def predicate_transform(bins_list, start_idx, end_idx):
    """
    {Description}
    
    Args:
        bins_list:
        start_idx:
        end_idx:
    Returns:
        start_val:
        end_val:
    """
    start_val, end_val = bins_list[start_idx], bins_list[end_idx]
    start_val += 1  # 调整
    return start_val, end_val
# %%

def predicate_location(reverse_dict, start_val, end_val, schema_name = None, column_name = None):
    """
    确定具体的predicate，bins_local的是构建bins是设置的分界线


    Args:
        bins_local: bins数值列表
        start_val: 开始的值
        end_val: 结束的值
    Returns:
        start_idx:
        end_idx:
    """
    start_val -= 1  # 调整
    try:
        start_idx, end_idx = \
            reverse_dict[start_val], reverse_dict[end_val]
    except KeyError as e:
        if schema_name is not None and column_name is not None:
            print(f"predicate_location: schema_name = {schema_name}. column_name = {column_name}.")
        print(f"predicate_location: start_val = {start_val}. end_val = {end_val}. "\
              f"reverse_dict = {reverse_dict}. meet KeyError.")
        raise e

    return start_idx, end_idx


# %%

def get_marginal_range(marginal_list, start_idx, end_idx):
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    return np.sum(marginal_list[start_idx: end_idx])

# %%
def prob_list_resize(in_prob_list):
    """
    {Description}
    
    Args:
        in_prob_list:
        arg2:
    Returns:
        out_prob_list:
        res2:
    """
    expected_sum = 1.0
    actual_sum = np.sum(in_prob_list)
    resize_factor = expected_sum / actual_sum

    out_prob_list = [i * resize_factor for i in in_prob_list]
    return out_prob_list

# %%

def prob_dict_normalization(in_dict, op_func = None):
    """
    {Description}

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    key_list, val_list = dict2list(in_dict)

    if op_func is not None:
        val_list = [op_func(val) for val in val_list]

    val_sum = sum(val_list)
    val_list = [val / val_sum for val in val_list]
    return {k: v for k, v in zip(key_list, val_list)}

def prob_dict_choice(prob_dict):
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    val_list, prob_list = dict2list(prob_dict)
    val_out = np.random.choice(a=val_list, p=prob_list)
    return val_out

# %%

def prob_dict_infer(in_dict, out_num):
    """
    {Description}

    Args:
        val_dict:
        out_num:
    Returns:
        return1:
        return2:
    """
    val_list, prob_list = dict2list(in_dict)
    if out_num >= len(val_list):
        if out_num == 1:
            return val_list[0]
        else:
            return val_list
    
    # prob_list = [item ** (1.5) for item in prob_list]
    prob_list = prob_list_resize(prob_list)
    # print(f"prob_dict_infer: prob_list = {list_round(prob_list, 3)}.")
    res_list = np.random.choice(a=val_list, \
        size=out_num, p=prob_list, replace=False)
    
    if len(res_list) == 1:
        return res_list[0]
    else:
        return res_list

# %% 可视化结果的输出

verbose_outpath = "/home/lianyuan/Research/CE_Evaluator/log/verbose.txt"    # 可视化结果输出路径

def set_verbose_path(new_path: str):
    """
    {Description}
    
    Args:
        new_path: 新的输出路径
        arg2:
    Returns:
        old_path: 旧的输出路径
        res2:
    """
    global verbose_outpath
    old_path = verbose_outpath
    verbose_outpath = new_path
    return old_path

def verbose(*args, **kwargs):
    with open(verbose_outpath, "a+") as f_out:
        print(*args, **kwargs, flush=True, file=f_out)

# %%
def benefit_calculate(cost_true, cost_estimation, clip_factor = 10.0):
    """
    计算节点拓展的收益情况
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    # print("benefit_calculate: cost_true = {:.2f}. cost_estimation = {:.2f}. clip_factor = {:.2f}.".\
    #     format(cost_true, cost_estimation, clip_factor))
    benefit1 = 0.0   # [0, 1]之间的值
    # 计算plan的增益
    if cost_true * clip_factor <= cost_estimation:
        benefit1 = 1.0
    else:
        benefit1 = (cost_estimation - cost_true) / (cost_true * (clip_factor - 1))

    return benefit1


# %%
def get_signature(input_string, num_out=128):
    hash_func = hashlib.sha256()  # 使用SHA-256哈希函数，可以根据需求选择不同的哈希函数
    
    # 更新哈希函数的输入数据
    hash_func.update(input_string.encode('utf-8'))
    
    # 获取哈希值的字节表示
    hash_bytes = hash_func.digest()
    
    # 计算要提取的字节数
    num_bytes = num_out // 2
    
    # 截取指定字节数并转换为16进制表示
    signature_hex = hash_bytes[:num_bytes].hex()
    
    return signature_hex.upper()


# %% 用于打印函数执行时间的装饰器
def trace(*args, **kwargs):
    print("trace: ", end="")
    print(*args, **kwargs)


from collections import defaultdict
import atexit

func_time_dict = defaultdict(list)
# mode = "realtime"   # 支持realtime/delay
mode = "delay"

def print_func_time():
    if mode == "delay":
        # # 之前的信息打印
        # for k, v in func_time_dict.items():
        #     trace(f"func_exec_time: {k} took {sum(v):.4f} seconds to run. call_times = {len(v)}")

        # 将函数按照总执行时间大小进行打印
        timed_list = [(k, sum(v), len(v)) for k, v in func_time_dict.items()]
        timed_list.sort(key= lambda a:a[1], reverse=True)

        for func_name, time_sum, call_times in timed_list:
            trace(f"func_exec_time: {func_name} took {time_sum:.3f} seconds to run. call_times = {call_times}.")
        
atexit.register(print_func_time)

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time

        if mode == "realtime":
            # print(f"{func.__name__} took {elapsed_time:.6f} seconds to run.")
            trace(f"func_exec_time: {func.__name__} took {elapsed_time:.6f} seconds to run.")
        elif mode == "delay":
            func_time_dict[func.__name__].append(elapsed_time)
        else:
            raise ValueError(f"timing_decorator: Unsupported mode({mode})")
        return result
    return wrapper

def file_content_read(file_path, time_interval, try_times):
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

@timing_decorator
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
    
# %%

def dict_str(in_dict: dict):
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    sorted_keys = sorted(in_dict.keys())
    res_str = ", ".join([f"{k}: {in_dict[k]}" for k in sorted_keys])
    return res_str

# %%

def get_sub_dict(val_dict, key_list):
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
    
    for k in key_list:
        res_dict[k] = val_dict[k]
    return res_dict


# %%

def multi_level_dict(level):
    def recursive_dict(level):
        if level == 1:
            return dict()
        else:
            return defaultdict(lambda: multi_level_dict(level - 1))
        
    # dict_obj = defaultdict(lambda: defaultdict(lambda: dict()))
    dict_obj = recursive_dict(level)
    return dict_obj

# %%

def dict_concatenate(*dict_list):
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    # assert 
    out_dict = defaultdict(list)
    key_list = dict_list[0]
    for k in key_list:
        try:
            for dict_item in dict_list:
                out_dict[k].append(dict_item[k])
        except TypeError as e:
            print(f"dict_concatenate: meet TypeError. k = {k}.")
            raise e

    return dict_apply(out_dict, lambda a: tuple(a))


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

def dict_max(in_dict: dict, val_func = None):
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    in_list = list(in_dict.items())

    if val_func is not None:
        return max(in_list, key=val_func)
    else:
        return max(in_list)
    

def dict_reverse(in_dict: dict):
    """
    {Description}

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    out_dict = {v:k for k, v in in_dict.items()}
    return out_dict

def dict_subset(in_dict: dict, func, mode = "key") -> dict:
    """
    {Description}
    
    Args:
        in_dict:
        func:
        mode:
    Returns:
        out_dict:
        res2:
    """
    if mode == "key":
        out_dict = {k: v for k, v in in_dict.items() if func(k) == True}
    else:
        out_dict = {k: v for k, v in in_dict.items() if func(v) == True}

    return out_dict


# %%

# def extract_card_info(card_dict: dict) -> tuple[dict, dict, dict, dict]:
#     """
#     {Description}

#     Args:
#         card_dict:
#         arg2:
#     Returns:
#         subquery_true:
#         single_table_true:
#         subquery_estimation:
#         single_table_estimation:
#     """
#     try:
#         subquery_true, single_table_true = \
#             card_dict['true']['subquery'], card_dict['true']['single_table']
#         subquery_estimation, single_table_estimation = \
#             card_dict['estimation']['subquery'], card_dict['estimation']['single_table']
#     except Exception as e:
#         print(f"extract_card_info: meet Error. card_dict = {card_dict}.")
#         raise e
#         # return {}, {}, {}, {}
    
#     return subquery_true, single_table_true, \
#         subquery_estimation, single_table_estimation

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

# def pack_card_info(subquery_true, single_table_true, \
#         subquery_estimation, single_table_estimation):
#     """
#     {Description}

#     Args:
#         subquery_true:
#         single_table_true:
#         subquery_estimation: 
#         single_table_estimation: 
#     Returns:
#         out_card_dict:
#         return2:
#     """
#     out_card_dict = {
#         "true": {
#             "subquery": subquery_true,
#             "single_table": single_table_true
#         },
#         "estimation": {
#             "subquery": subquery_estimation,
#             "single_table": single_table_estimation
#         }
#     }
#     return out_card_dict

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

# %% 显示card_dict的具体信息

def display_card_dict(card_dict: dict, func_name = None):
    """
    {Description}

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    subquery_true, single_table_true, subquery_est, single_table_est = extract_card_info(card_dict)
    subquery_keys, single_table_keys = list(set(subquery_true.keys()) | set(subquery_est.keys())), \
        list(set(single_table_true.keys()) | set(single_table_est.keys()))

    subquery_text, single_table_text = {}, {}

    for k in subquery_keys:
        true_card = subquery_true.get(k, "null")
        est_card = subquery_est.get(k, "null")
        subquery_text[k] = f"({true_card}, {est_card})"

    for k in single_table_keys:
        true_card = single_table_true.get(k, "null")
        est_card = single_table_est.get(k, "null")
        single_table_text[k] = f"({true_card}, {est_card})"

    if func_name is None:
        print("display_card_dict: format = (true_card, est_card)")
    else:
        print(f"{func_name}: call display_card_dict. format = (true_card, est_card)")

    print(subquery_text)
    print(single_table_text)

    return subquery_text, single_table_text


def copy_card_dict(card_dict: dict):
    """
    {Description}

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    return deepcopy(card_dict)

# %%

def list_round(float_list, digit = 3):
    """
    {Description}

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    out_list = [round(num, digit) for num in float_list]
    return out_list

def dict_round(float_dict, digit = 3):
    """
    {Description}

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    out_dict = {k: round(v, digit) for k, v in float_dict.items()}
    return out_dict

# %%

def meta_intersection(query_meta1, query_meta2):
    """
    {Description}

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    out_meta = list(set(query_meta1[0]).intersection(set(query_meta2[0]))), \
        list(set(query_meta1[1]).intersection(set(query_meta2[1])))
    return out_meta

def dict_intersection(in_dict1, in_dict2):
    """
    {Description}

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    in_list1 = list(in_dict1.items())
    in_list2 = list(in_dict2.items())

    try:
        out_list = list(set(in_list1).intersection(set(in_list2)))
    except TypeError as e:
        print(f"dict_intersection: in_list1 = {in_list1}. in_list2 = {in_list2}.")
        raise
    out_dict = {k: v for k, v in out_list}
    return out_dict

# %% 处理列表前缀的函数

cmp_func = lambda a, b: a == b[:len(a)]

def prefix_match(prefix_ref, prefix_list):
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    out_list = [item for item in prefix_list if cmp_func(prefix_ref, item)]
    print(f"prefix_match: prefix_ref = {prefix_ref}. prefix_list = {prefix_list}. out_list = {out_list}")
    return out_list

def prefix_aggregation(prefix_ref, prefix_list: list):
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    # print(f"prefix_aggregation: prefix_ref = {prefix_ref}. prefix_list = {prefix_list}.")
    out_dict = defaultdict(list)
    pre_len = len(prefix_ref)
    
    for item in prefix_list:
        try:
            key = item[pre_len]
            out_dict[key].append(item)
        except IndexError as e:
            continue
        
    # print(f"prefix_aggregation: out_dict = {out_dict}.")
    return out_dict

# %%

def get_target_card(query_meta, card_dict, alias_mapping):
    """
    {Description}

    Args:
        arg1:
        arg2:
    Returns:
        card_true:
        card_est:
    """
    schema_list = query_meta[0]
    alias_tuple = tuple(sorted([alias_mapping[s] for s in schema_list]))
    subquery_true, single_table_true, subquery_estimation, \
        single_table_estimation = extract_card_info(card_dict)
    
    subquery_true: dict = subquery_true
    single_table_true: dict = single_table_true 
    subquery_estimation: dict = subquery_estimation
    single_table_estimation: dict = single_table_estimation

    if len(alias_tuple) > 1:
        key = alias_tuple
        card_true = subquery_true.get(key, None)
        card_est = subquery_estimation.get(key, None)
    else:
        key = alias_tuple[0]
        card_true = single_table_true.get(key, None)
        card_est = single_table_estimation.get(key, None)
        
    return card_true, card_est

# %%

def card_dict_normalization(in_card_dict: dict) -> dict:
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    subquery_true, single_table_true, subquery_est, \
        single_table_est = extract_card_info(in_card_dict)
    
    subquery_true = dict_apply(subquery_true, int, mode="value")
    single_table_true = dict_apply(single_table_true, int, mode="value")
    
    subquery_est = dict_apply(subquery_est, int, mode="value")
    single_table_est = dict_apply(single_table_est, int, mode="value")

    out_card_dict = pack_card_info(subquery_true, \
        single_table_true, subquery_est, single_table_est)
    return out_card_dict

# %%
from utility import workload_spec
def construct_alias_list(schema_list, workload):
    """
    {Description}

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    alias_mapping = workload_spec.abbr_option[workload]
    alias_list = [alias_mapping[s] for s in schema_list]
    return alias_list


class SignatureSerializer(object):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self,):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        self.signature_dict = {}

    def reset(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        self.signature_dict = {}

    def get_new_id(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        return len(self.signature_dict)

    def add_signature(self, new_sig):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        new_id = self.get_new_id()
        self.signature_dict[new_sig] = new_id
        return self.signature_dict


    def translate_signature(self, target_sig):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        if target_sig in self.signature_dict:
            return self.signature_dict[target_sig]
        else:
            self.add_signature(target_sig)
            return self.signature_dict[target_sig]

    def translate_list(self, sig_list):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        return [self.translate_signature(sig) for sig in sig_list]
    
# %%
def clean_mv_cache(workload, mv_cache_dir = "/home/lianyuan/Research/CE_Evaluator/mv_cache"):
    """
    清理mv_cache
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    target_dir = p_join(mv_cache_dir, workload)
    if os.path.isdir(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)

# %%

def card_dict_valid_check(subquery_dict: dict, single_table_dict: dict):
    """
    {Description}

    Args:
        arg1:
        arg2:
    Returns:
        flag: True代表合法，False代表不合法
        return2:
    """
    for val in subquery_dict.values():
        if isinstance(val, (float, int)) == False or val < -1e-6:
            print(f"card_dict_valid_check: 出现超时的情况! 接下来会直接删除任务. values = {subquery_dict.values()}.")
            return False
    
    for val in single_table_dict.values():
        if isinstance(val, (float, int)) == False or val < -1e-6:
            print(f"card_dict_valid_check: 出现超时的情况! 接下来会直接删除任务. values = {single_table_dict.values()}.")
            return False
        
    return True

# %%

def modify_template_meta(workload, ce_method, signature, template_list):
    """
    {Description}

    Args:
        workload:
        ce_method:
        signature:
        template_list:
    Returns:
        return1:
        return2:
    """
    file_dir = "/home/lianyuan/Research/CE_Evaluator/intermediate"
    file_name = "modify_template_meta.py"
    curr_dir = os.getcwd()
    python_path = "/home/lianyuan/anaconda3/envs/CE_Evaluator/bin/python"
    cmd_template = f"{python_path} {file_name}" + " --workload {workload} "\
        "--ce_method {ce_method} --signature {signature} --template_list {template_list}"
    params_dict = {
        "workload": workload,
        "ce_method": ce_method,
        "signature": signature,
        "template_list": " ".join(template_list) if \
            isinstance(template_list, list) else template_list
    }
    os.chdir(file_dir)
    cmd_content = cmd_template.format(**params_dict)
    print(f"modify_template_meta: cmd_content = {cmd_content}")
    os.system(cmd_content)
    os.chdir(curr_dir)

    
# %%
