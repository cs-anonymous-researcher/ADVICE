#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from os.path import join as p_join
from matplotlib import pyplot as plt
import json, os, time, re, math
import pickle
import hashlib, shutil
from timeout_decorator import timeout, timeout_decorator
# %%

import random
from copy import deepcopy
from query.query_construction import abbr_option
from query.query_construction import JOB_abbr, STATS_abbr, RELEASE_abbr, DSB_abbr
from collections import defaultdict
from utility.utils import load_json, predicate_location, predicate_transform
from data_interaction import data_management
from utility import workload_spec, utils, common_config

# %%

def tuple_hash(in_tuple):
    """
    {Description}
    
    Args:
        in_tuple:
    Returns:
        hash_val:
    """
    h = hashlib.blake2b()
    for i in in_tuple:
        if i is None:
            continue
        elif isinstance(i, str):
            h.update(bytes(i, encoding="utf8"))
        elif isinstance(i, (int, float, np.int64)):
            # print("i = {}.".format(str(i)))
            # h.update(bytes(i))
            # 之前的代码没法处理负数，作为应对直接转成str类型
            h.update(bytes(str(i), encoding="utf-8"))
        else:
            raise ValueError("Unsupported data type: {}".format(type(i)))
    return h.hexdigest()

def condition_encoding(schema_list: list, filter_list: list):
    """
    {Description}
    
    Args:
        schema_list:
        filter_list:
        column_list:
    Returns:
        mv_name:
    """
    schema_list = sorted(schema_list)
    filter_list = sorted(filter_list)

    schema_name = "#".join(schema_list)

    h = hashlib.blake2b()
    for item in filter_list:
        h.update(bytes(tuple_hash(item), encoding="utf8"))
    filter_name = h.hexdigest()[:8]
    # column_name = ""
    mv_name = "#".join([schema_name, filter_name])

    return mv_name

def process_single_condition(in_df, cond_tuple):
    """
    {Description}
    TODO: 增加对空值的支持，扩展代码的泛用性

    Args:
        in_df:
        cond_tuple:
    Returns:
        out_df:
    """
    tbl_abbr, col_name, lower_bound, upper_bound = cond_tuple
    
    df_col_name = "{}_{}".format(tbl_abbr, col_name)
    # print("print dataframe head:")
    # print(in_df.head())
    if lower_bound is not None:
        out_df = in_df[in_df[df_col_name] >= lower_bound]
    else:
        out_df = in_df  # 保持变量一致性
    if upper_bound is not None:
        out_df = out_df[out_df[df_col_name] <= upper_bound]
    out_df = out_df.dropna(subset = [df_col_name], how='any')  # 删除NA的项
    return out_df


def conditions_apply(in_df, filter_list, column_subset = []):
    """
    {Description}
    
    Args:
        in_df: 输入的DataFrame
        filter_list: 过滤列表
        column_subset: 列的子集
    Returns:
        out_df:
    """
    out_df = in_df
    for filter in filter_list:
        out_df = process_single_condition(\
            out_df, cond_tuple = filter)
    return out_df

def meta_merge(left_meta, right_meta):
    """
    {Description}
    
    Args:
        left_meta:
        right_meta:
    Returns:
        merged_meta:
    """
    left_schema, left_filter = left_meta
    right_schema, right_filter = right_meta

    # print("mv_management.meta_merge: left_meta = {}. right_meta = {}.".format(left_meta, right_meta))
    merged_meta = (list(left_schema) + list(right_schema), list(left_filter) + list(right_filter))  
    merged_meta[0].sort()
    merged_meta[1].sort()
    # 结果进行去重
    compact_schema, compact_filter = [], []

    for idx, item in enumerate(merged_meta[0]):
        if idx == 0 or compact_schema[-1] != item:
            compact_schema.append(item)

    for idx, item in enumerate(merged_meta[1]):
        if idx == 0 or compact_filter[-1] != item:
            compact_filter.append(item)

    merged_meta = compact_schema, compact_filter
    return merged_meta


def meta_subset(in_meta, schema_subset, abbr_mapping = "job"):
    """
    获得元信息的一个子集，用于处理子查询的场景
    
    Args:
        in_meta:
        schema_subset: 关系的子集
        abbr_mapping:
    Returns:
        schema_out: 
        filter_out:
    """
    if isinstance(abbr_mapping, dict):
        abbr_mapping = abbr_mapping
    elif isinstance(abbr_mapping, str):
        if abbr_mapping == "job":
            abbr_mapping = JOB_abbr
        elif abbr_mapping == "stats":
            abbr_mapping = STATS_abbr
        elif abbr_mapping == "release":
            abbr_mapping = RELEASE_abbr
        elif abbr_mapping == "dsb":
            abbr_mapping = DSB_abbr
        else:
            raise ValueError("meta_subset: Unsupported abbr_mapping({})".format(abbr_mapping))
    else:
        raise TypeError("Unsupported abbr_mapping type: {}".format(type(abbr_mapping)))

    schema_list, filter_list = in_meta
    schema_out, filter_out = [], []

    for s in schema_list:
        if abbr_mapping[s] in schema_subset:
            schema_out.append(s)

    for f in filter_list:
        abbr, col, low, upper = f
        if abbr in schema_subset:
            filter_out.append(f)
            
    return schema_out, filter_out

def meta_filter_add(in_meta, filter):
    """
    在meta信息添加单个筛选条件
    
    Args:
        in_meta:
        filter:
    Returns:
        in_schema:
        in_filter:
    """
    in_schema, in_filter = in_meta
    f = filter
    if f not in in_filter:
        in_filter.append(f)
    return in_schema, in_filter

def meta_filter_append(in_meta, filter_list):
    """
    添加一组筛选条件
    
    Args:
        in_meta:
        filter_list:
    Returns:
        out_meta: 结果元信息
    """
    in_schema, in_filter = in_meta
    # TODO: 条件去重，添加对谓词的考虑
    # in_filter = in_filter + filter_list
    in_filter = list(in_filter)
    for f in filter_list:
        if f not in in_filter:
            in_filter.append(f)
    return in_schema, tuple(in_filter)

def meta_schema_add(in_meta, schema):
    """
    在meta中添加单个schema
    
    Args:
        in_meta:
        schema:
    Returns:
        out_meta:
    """
    in_schema, in_filter = in_meta
    # print("meta_schema_add: in_meta = {}.".format(in_meta))
    in_schema = list(in_schema)
    # 条件去重
    if schema not in in_schema:
        in_schema.append(schema)

    return in_schema, in_filter

def meta_schema_append(in_meta, schema_list):
    """
    在meta中添加一个schema列表
    
    Args:
        in_meta:
        schema_list:
    Returns:
        out_meta:
    """
    in_schema, in_filter = in_meta
    # 条件去重
    for s in schema_list:
        if s not in in_schema:
            in_schema.append(s)

    return in_schema, in_filter

def meta_comparison(meta1, meta2):
    """
    meta信息的比较
    
    Args:
        meta1:
        meta2:
    Returns:
        flag:
    """
    flag = True
    # 直接比较repr是否相等
    repr1 = condition_encoding(meta1[0], meta1[1])
    repr2 = condition_encoding(meta2[0], meta2[1])
    # print("repr1 = {}. repr2 = {}.".format(repr1, repr2))
    if repr1 == repr2:
        flag = True
    else:
        flag = False
    return flag

def meta_copy(in_meta):
    """
    复制meta的信息
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    out_meta = deepcopy(in_meta[0]), deepcopy(in_meta[1])
    return out_meta


def meta_repr(in_meta, workload = "job"):
    """
    关于query_meta对应的schema表示
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    abbr_mapping = abbr_option[workload]
    res_list = []
    schema_list, filter_list = in_meta
    for s in schema_list:
        res_list.append(abbr_mapping[s])

    return tuple(sorted(res_list))

def meta_key_repr(in_meta, workload = "job"):
    """
    meta信息转成一个字符串，用于字典key的表示

    Args:
        in_meta: 查询meta信息
        workload: meta所属的工作负载
    Returns:
        meta_key_str:
    """
    schema_list, filter_list = in_meta
    abbr_mapping = abbr_option[workload]
    
    def condition_str(alias, col_name, lower_bound, upper_bound):
        if lower_bound is None and upper_bound is not None:
            return "{}.{}<={}".format(alias, col_name, upper_bound)
        elif lower_bound is not None and upper_bound is None:
            return "{}.{}>={}".format(alias, col_name, lower_bound)
        elif lower_bound is not None and upper_bound is not None:
            if lower_bound == upper_bound:
                return "{}.{}={}".format(alias, col_name, lower_bound)
            else:
                return "{}<={}.{}<={}".format(lower_bound, alias, col_name, upper_bound)
        else:
            raise ValueError("meta_key_repr: lower_bound and upper_bound are None!")

    schemas = ",".join([abbr_mapping[s] for s in sorted(schema_list)])
    conditions = ",".join([condition_str(*item) for item in sorted(filter_list)])

    meta_key_str = "({schemas}),{conditions}".format(schemas = schemas, conditions = conditions)
    return meta_key_str

def meta_decompose(in_meta, workload = "job"):
    meta_list = []
    schema_list, filter_list = in_meta
    alias_mapping = abbr_option[workload]
    
    def aggregate_corr_cond(schema_name, filter_list):
        schema_alias = alias_mapping[schema_name]
        res_list = []
        for item in filter_list:
            if item[0] == schema_alias:
                res_list.append(item)
        return res_list
    
    for schema in schema_list:
        local_schema_list = [schema,]
        local_filter_list = aggregate_corr_cond(schema, filter_list)
        meta_list.append((local_schema_list, local_filter_list))

    return meta_list


def meta_standardization(in_meta):
    """
    元信息规范化

    Args:
        in_meta:
    Returns:
        out_meta:
    """
    out_meta = in_meta[0], []

    for item in in_meta[1]:
        alias, column, lower_bound, upper_bound = item
        lower_new, upper_new = 0, 0

        # 目前只考虑int类型的数据
        if lower_bound is None:
            lower_new = None
        else:
            lower_new = int(lower_bound)

        if upper_bound is None:
            upper_new = None
        else:
            upper_new = int(upper_bound)
        
        out_meta[1].append((alias, column, lower_new, upper_new))
    return out_meta

# %%

class MaterializedViewManager(object):
    """
    物化视图管理

    Members:
        field1:
        field2:
    """

    def __init__(self, workload = "job", cache_dir = 
        "/home/lianyuan/Research/CE_Evaluator/mv_cache", meta_name = "mv_meta.json"):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        # if workload == "job":
        #     self.abbr_mapping = abbr_option[workload]   # 设置映射
        #     workload = workload.upper()
        # elif workload == "stats":
        #     pass
        
        self.abbr_mapping = abbr_option[workload.lower()]   # 设置映射
        self.mv_dir = p_join(cache_dir, workload)
        meta_path = p_join(self.mv_dir, meta_name)
        
        # 加载元数据
        # self.mv_meta_dict = load_json(meta_path)
        self.mv_meta_dict = {}
        self.mv_meta_path = meta_path

    def clean_historical_mv(self,):
        """
        清理历史的物化视图

        Args:
            None
        Returns:
            historical_meta_dict:
        """
        historical_meta_dict = deepcopy(self.mv_meta_dict)
        self.mv_meta_dict = {}
        for f_name in os.listdir(self.mv_dir):
            f_path = p_join(self.mv_dir, f_name)
            if os.path.isfile(f_path) == True:
                os.remove(f_path)

        return historical_meta_dict

    def column_rename(self, in_df, table_abbr):
        """
        DataFrame列的重命名
        
        Args:
            in_df:
            column_abbr: 列的缩写
        Returns:
            out_df:
        """
        df_columns = in_df.columns
        prefix = table_abbr + "_"
        valid = True
        mapping_dict = {}
        for col in df_columns:
            mapping_dict[col] = prefix + col

        out_df = in_df.rename(columns = mapping_dict)
        return out_df


    def get_initial_meta(self, schema_list, filter_list):
        """
        获得初始的元信息
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        return (schema_list, filter_list)

    def flush_mv_meta(self,):
        """
        {Description}
        
        Args:
            None
        Returns:
            None
        """
        # print("mv_meta_dict = {}".format(self.mv_meta_dict))
        with open(self.mv_meta_path, "w") as f_out:
            json.dump(self.mv_meta_dict, f_out, indent = 4)


    def update_mv_meta(self, joined_mv_name, joined_meta):
        """
        {Description}
        
        Args:
            joined_mv_name:
            joined_meta:
        Returns:
            None
        """
        self.mv_meta_dict[joined_mv_name] = joined_meta
        self.flush_mv_meta()
    
    def column_prefix_eval(self, in_df, in_meta):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        if len(in_meta[0]) > 1:
            # print("警告，")
            return in_df

        schema = in_meta[0][0]
        abbr = self.abbr_mapping[schema]
        prefix = "{}_".format(abbr)     # 预期的前缀
        flag = True
        for col in in_df.columns:
            if col.startswith(prefix) == False:
                flag = False

        if flag == False:
            in_df = self.column_rename(in_df, abbr)

        return in_df


    def generate_joined_mv(self, left_df, right_df, left_on, right_on, \
        left_cond, right_cond, left_meta, right_meta):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            joined_path:
            joined_df:
            joined_meta:
        """
        # 在列名上添加前缀
        left_df = self.column_prefix_eval(left_df, left_meta)
        right_df = self.column_prefix_eval(right_df, right_meta)

        left_df = conditions_apply(left_df, filter_list = left_cond)
        right_df = conditions_apply(right_df, filter_list = right_cond)

        joined_df = pd.merge(left = left_df, right = right_df, how="inner", \
            left_on = left_on, right_on = right_on)
        joined_meta = meta_merge(left_meta, right_meta)
        joined_meta = meta_filter_append(joined_meta, left_cond)
        joined_meta = meta_filter_append(joined_meta, right_cond)
        joined_mv_name = condition_encoding(*joined_meta)
        
        # print("generate_joined_mv: joined_name = {}. joined_meta = {}.".\
        #     format(joined_mv_name, joined_meta))
        self.update_mv_meta(joined_mv_name, joined_meta)
        joined_path = self.dump_mv(joined_df, joined_mv_name)

        return joined_path, joined_df, joined_meta

    def generate_single_mv(self, in_df, conditions, in_meta):
        """
        生成单表的物化视图

        Args:
            in_df: 数据对应的dataframe
            conditions: 新增的Meta条件
            in_meta: 原有的df对应的meta
        Returns:
            out_path:
            out_df:
            out_meta:
        """
        in_df = self.column_prefix_eval(in_df, in_meta)             # 添加前缀
        out_df = conditions_apply(in_df, filter_list = conditions)  # 获得过滤后的df

        out_meta = meta_filter_append(in_meta, filter_list = conditions)
        out_name = condition_encoding(*out_meta)
        # print("generate_single_mv: out_name = {}. out_meta = {}.".\
        #     format(out_name, out_meta))
        self.update_mv_meta(out_name, out_meta)
        out_path = self.dump_mv(out_df, out_name)
        return out_path, out_df, out_meta
    
    def generate_external_mv(self, ext_df, ext_meta):
        """
        保存从外部生成的物化视图
        
        Args:
            ext_df: 外部的DataFrame
            ext_meta: 外部的meta信息
        Returns:
            out_path: 
            out_df: 
            out_meta:
        """
        # 处理res_meta的数据类型，保证可以被序列化
        out_meta, out_df = meta_standardization(in_meta=ext_meta), ext_df
        # print("generate_external_mv: out_meta = {}.".format(out_meta))

        out_name = condition_encoding(*out_meta)
        self.update_mv_meta(out_name, out_meta)
        out_path = self.dump_mv(out_df, out_name)
        # print("generate_external_mv: out_name = {}. out_meta = {}.".\
        #     format(out_name, out_meta))

        return out_path, out_df, out_meta


    def search_available_mv(self, schemas, conditions):
        """
        搜索可用的物化视图
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        flag = False
        res_name = ""
        target_meta = (schemas, conditions)
        # print("target_meta = {}.".format(target_meta))
        # print("mv_meta_dict = {}.".format(self.mv_meta_dict))

        for name, meta in self.mv_meta_dict.items():
            if meta_comparison(meta, target_meta) == True:
                res_name = name
                flag = True
                break

        return flag, res_name

    def load_mv_from_meta(self, query_meta):
        """
        {Description}
    
        Args:
            query_meta:
        Returns:
            res_mv:
        """
        schemas, conditions = query_meta
        # print("schemas = {}. conditions = {}.".format(schemas, conditions))
        return self.load_mv(schemas, conditions)

    def load_mv_from_name(self, mv_name):
        """
        根据名字加载物化视图
        
        Args:
            mv_name:
        Returns:
            res_mv:
        """
        mv_path = p_join(self.mv_dir, "{}.pkl".format(mv_name))
        if os.path.isfile(mv_path):
            return pd.read_pickle(mv_path)
        else:
            print("load_mv_from_name: 文件名不存在")
            return None


    def load_mv(self, schemas, conditions):
        """
        加载物化视图
        
        Args:
            schemas:
            conditions:
        Returns:
            res1:
            res2:
        """
        flag, mv_name = self.search_available_mv(schemas, conditions)
        if flag == True:
            mv_path = p_join(self.mv_dir, "{}.pkl".format(mv_name))
            return pd.read_pickle(mv_path)
        else:
            return None


    def dump_mv(self, out_df, out_name):
        """
        {Description}
        
        Args:
            out_df:
            out_name:
        Returns:
            out_path:
        """
        out_path = p_join(self.mv_dir, "{}.pkl".format(out_name))
        out_df.to_pickle(out_path)
        return out_path

# %%

class MaterializedViewBuilder(object):
    """
    直接使用目标的meta信息构造出物化视图

    Members:
        field1:
        field2:
    """

    def __init__(self, workload, data_manager_ref: data_management.DataManager, \
        mv_manager_ref:MaterializedViewManager = None):
        """
        {Description}

        Args:
            workload:
            data_manager_ref:
            mv_manager_ref:
        """
        self.workload = workload
        self.data_manager = data_manager_ref
        self.mv_manager = mv_manager_ref
        self.alias_mapping = data_manager_ref.tbl_abbr
    
        self.cache_dir = "/home/lianyuan/Research/CE_Evaluator/mv_cache"

    def constraint_eval(self, curr_time, curr_card, constraint):
        # 判断
        flag = True
        if 'time' in constraint.keys():
            flag = flag and curr_time <= constraint['time']

        if 'cardinality' in constraint.keys():
            # 如果cardinality达不到要求，那就不退出
            flag = flag or curr_card <= constraint['cardinality']

        return flag
        
    def cond_bound_check(self, cond_bound_dict:dict):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        for k, v in cond_bound_dict.items():
            if isinstance(k, tuple):
                print(f"cond_bound_check: k = {k}. v = {len(v)}.")
            else:
                print(f"cond_bound_check: k = {k}. v = {v}.")

    def dynamic_mv_signature(self, target_meta: tuple, cond_bound_dict: dict):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        cond_bound_dict = utils.dict_apply(cond_bound_dict, str, mode="key")
        cond_bound_dict = utils.dict_apply(cond_bound_dict, str, mode="value")
        
        meta_data = str(sorted(target_meta[1])).encode()
        json_data = json.dumps(cond_bound_dict, sort_keys=True).encode()
        out_signature = hashlib.sha256(meta_data + json_data).hexdigest()
        return out_signature
    
    def build_dynamic_mv_on_bound(self, target_meta: tuple, cond_bound_dict: dict, \
        constraint: dict, use_cache = True):
        """
        根据外部已经构建好的bound_dict动态的构建mv
    
        Args:
            target_meta:
            cond_bound_dict:
            constraint:
            use_cache:
        Returns:
            merged_df: 
            extra_meta_filter: 
            cond_idx:
        """
        print(f"build_dynamic_mv_on_bound: target_meta = {target_meta[0]}. " \
              f"cond_bound_dict = {cond_bound_dict}. constraint = {constraint}.")
        max_cond_idx, cond_idx = cond_bound_dict['max_length'], -1
        # self.cond_bound_check(cond_bound_dict)

        def apply_config(cond_idx):
            res_meta_list = []
            for curr_meta in origin_meta_list:
                new_meta = meta_copy(curr_meta)
                for k, val_list in cond_bound_dict.items():
                    if new_meta[0][0] == k[0]:
                        # 表的名字匹配成功，添加新的条件
                        try:
                            start_val, end_val = val_list[cond_idx]
                        except IndexError as e:
                            print(f"cond_idx = {cond_idx}. max_cond_idx = {max_cond_idx}. len(val_list) = {len(val_list)}")
                            raise e
                        
                        alias = self.alias_mapping[k[0]]
                        new_meta[1].append((alias, k[1], start_val, end_val))    # 添加新的condition
                res_meta_list.append(new_meta)

            # print(f"apply_config: cond_idx = {cond_idx}. res_meta_list = {res_meta_list}.")
            return res_meta_list
        
        mv_cache_path = p_join(self.cache_dir, self.workload, f"{self.dynamic_mv_signature(target_meta, cond_bound_dict)}.pkl")

        if use_cache == True and os.path.isfile(mv_cache_path) == True:
            #
            ts = time.time()
            merged_df, cond_idx = utils.load_pickle(mv_cache_path)
            ts = time.time()
        else:
            #
            origin_meta_list = self.meta_decompose(target_meta)
            join_order_list = self.determine_merge_order(origin_meta_list)
            
            test_cnt, valid_flag = 10, True    # 进行三次迭代测试
            while True:
                if test_cnt == 0:
                    break
                test_cnt -= 1
                cond_idx += 1

                # current_meta_list = apply_config(cond_idx)    # 
                # current_data_list = self.fetch_data_list(src_meta_list = current_meta_list)
                new_meta_list = apply_config(cond_idx)    # 
                new_data_list = self.fetch_data_list(src_meta_list = new_meta_list)

                # 这里存在内存超限导致进程直接被Kill的问题，解决的策略是创建一个cache，每次把结果保存在cache里，
                # 如果下一次失败了，上一次的结果会存在cache里，直接读取就行
                try:
                    ts = time.time()
                    # temp_df = self.merge_all_tables_under_limit(single_meta_list = new_meta_list, \
                    #     data_list = new_data_list, order_list = join_order_list)

                    # 2024-03-22: 改写merge_all_tables的实现，支持动态调整timeout
                    merge_all_tables = timeout(seconds=common_config.merge_time_limit)(self.merge_all_tables)
                    temp_df = merge_all_tables(new_meta_list, new_data_list, join_order_list)
                    te = time.time()

                    # 完成迭代赋值，并表示已创建merged_df对象
                    merged_df, valid_flag = temp_df, True
                    # current_meta_list, current_data_list = new_meta_list, new_data_list
                except timeout_decorator.TimeoutError as e1:
                    # 出现超时的问题，回退部分信息
                    cond_idx -= 1
                    print(f"build_dynamic_mv_on_bound: timeout. cond_idx = {cond_idx}")
                    if valid_flag == True:
                        break
                    else:
                        time.sleep(5)
                        continue
                except Exception as e2:
                    cond_idx -= 1
                    print(f"build_dynamic_mv_on_bound: error_type = {type(e2)}. cond_idx = {cond_idx}")
                    if valid_flag == True:
                        break
                    else:
                        time.sleep(5)
                        continue

                delta_time = te - ts

                if self.constraint_eval(delta_time, len(merged_df), constraint) == False:
                    # 满足条件以后直接退出
                    print("build_dynamic_mv_on_bound: self.constraint_eval(delta_time, len(merged_df), constraint) == False")
                    break
                
                if cond_idx + 1 >= max_cond_idx:
                    # 循环迭代超过上限，直接退出
                    print("build_dynamic_mv_on_bound: cond_idx + 1 >= max_cond_idx")
                    break

                # 完成config的迭代
                curr_metrics = {
                    "time": te - ts,
                    "cardinality": len(merged_df)
                }
                # print(f"build_dynamic_mv_on_bound: cond_idx = {cond_idx}. curr_metrics = {curr_metrics}")
                if use_cache == True:
                    # 使用cache的条件下，保存结果
                    ts = time.time()
                    utils.dump_pickle((merged_df, cond_idx), data_path=mv_cache_path)
                    te = time.time()
                    obj_length = len(merged_df)
                    print(f"build_dynamic_mv_on_bound: save mv_object. delta_time = {te - ts:.2f}. "\
                          f"obj_length = {obj_length}. cond_idx = {cond_idx}. mv_cache_path = {mv_cache_path}.")

        extra_meta_filter = []
        for k, val_list in cond_bound_dict.items():
            try:
                schema_name, column_name = k
                alias_name = self.alias_mapping[schema_name]
                start_val, end_val = val_list[cond_idx]

                print(f"build_dynamic_mv.final: k = {k}. start_val = {start_val}. end_val = {end_val}.")
                extra_meta_filter.append((alias_name, column_name, start_val, end_val))
            except ValueError:
                print(f"build_dynamic_mv_on_bound: meet error. k = {k}")

        try:
            return merged_df, extra_meta_filter, cond_idx
        except UnboundLocalError as e:
            return pd.DataFrame(), {}, 0


    def build_dynamic_mv_on_constraint(self, target_meta, growth_config: dict, grid_config: dict, constraint: dict, mv_hint: dict = None):
        """
        在限制条件下构建动态的mv
        
        Args:
            target_meta: 目标的元信息
            growth_config: 不同的列增长策略，key是column，value是(start_idx, ratio, rate)
            grid_config: 被选择的列相关grid信息，key是column，value是(bins_list, marginal_list)
            constraint: 限制条件，目前能想到的是card constraint和time constriant
            mv_hint: mv生成提示信息
        Returns:
            mv_object: 结果的mv对象
            extra_meta_filter: 额外的元数据信息
            extra_config: 
        """
        origin_meta_list = self.meta_decompose(target_meta)

        def apply_config(curr_config):
            """
            将配置运用在原始的meta_list上
            
            Args:
                arg1:
                arg2:
            Returns:
                res1:
                res2:
            """
            # print("apply_config: curr_config = {}.".format(curr_config))
            res_meta_list = []
            for curr_meta in origin_meta_list:
                new_meta = meta_copy(curr_meta)
                for k, v in curr_config.items():
                    if new_meta[0][0] == k[0]:
                        # 表的名字匹配成功，添加新的条件
                        ratio, start_idx, end_idx = v
                        bins_list = grid_config[k][0]
                        start_val, end_val = predicate_transform(bins_list, start_idx, end_idx)
                        alias = self.alias_mapping[k[0]]
                        new_meta[1].append((alias, k[1], start_val, end_val))    # 添加新的condition
                        # print(f"apply_config: k = {k}. start_idx = {start_idx}. end_idx = {end_idx}. total_len = {len(bins_list)} "
                        #     f"start_val = {start_val}. end_val = {end_val}. ratio = {ratio}.")
                res_meta_list.append(new_meta)
            return res_meta_list

        def end_location(start_idx, ratio, marginal_list):
            """
            确定结束的索引位置
            
            Args:
                start_idx:
                ratio:
                marginal_list:
            Returns:
                end_idx:
                actual_ratio:
            """
            
            total_length = sum(marginal_list)
            selected_length = int(total_length * ratio)

            for end_idx in range(start_idx + 1, len(marginal_list) + 1):
                curr_sum = np.sum(marginal_list[start_idx: end_idx])
                if curr_sum >= selected_length:
                    actual_ratio = curr_sum / selected_length
                    break
            actual_ratio = 1.0
            return end_idx, actual_ratio
        
        def get_config_init():
            """
            获得最初始的config
            
            Args:
                None
            Returns:
                res_config:
            """
            res_config = {}
            for k, v in growth_config.items():
                start_idx, ratio, rate = v
                bins_list, marginal_list = grid_config[k]
                end_idx, _ = end_location(start_idx=start_idx, ratio=ratio, marginal_list=marginal_list)
                res_config[k] = ratio, start_idx, end_idx

            return res_config

        iter_pos = 0
        def config_iterate(curr_config:dict, curr_metrics:dict):
            """
            迭代当前的配置，如果迭代的预期过大，直接退出结果
            为了保证公平性，对于dict.items()的结果做一个random.shuffle

            Args:
                curr_config: 当前的配置
                curr_metrics: 当前配置下的指标
            Returns:
                res1:
                res2:
            """
            nonlocal iter_pos
            res_config = {}
            # 表示是否迭代结束
            complete_flag = True
            expect_metrics = deepcopy(curr_metrics)

            accumulative_ratio = 1.0
            item_list = list(curr_config.items())
            # random.shuffle(item_list)

            print(f"build_dynamic_mv_on_constraint: curr_metrics = {curr_metrics}. curr_config = {curr_config}")
            for k, v in item_list[iter_pos:]:
                ratio, start_idx, _ = v
                ratio *= growth_config[k][2]    # 调整ratio
                accumulative_ratio *= growth_config[k][2]
                if ratio >= 1:
                    # 如果超过1，就调整成1
                    accumulative_ratio *= 1 / ratio
                    ratio = 1
                else:
                    complete_flag = False

                end_idx, actual_ratio = end_location(start_idx=start_idx, ratio=ratio, marginal_list=grid_config[k][1])
                res_config[k] = (ratio, start_idx, end_idx)

                print(f"config_iterate: k = {k}. start_idx = {start_idx}. end_idx = {end_idx}. "
                    f"total_len = {len(grid_config[k][0])}. ratio = {ratio}. actual_ratio = {actual_ratio}")

                # 2/3在这里是一个magic_number
                # print("accumulative_ratio = {}.".format(accumulative_ratio))
                # if constraint_eval(expect_metrics['time'] * accumulative_ratio * 1/3, 
                #                    expect_metrics['cardinality'] * accumulative_ratio * 1/3) == True:
                #     iter_pos = (iter_pos + 1) % len(item_list)  # 迭代位置重定向
                #     break
            
            # 补全res_config
            for k, v in curr_config.items():
                if k not in res_config.keys():
                    res_config[k] = v

            return res_config, complete_flag


        def constraint_eval(curr_time, curr_card):
            # 判断
            flag = True
            if 'time' in constraint.keys():
                flag = flag and curr_time <= constraint['time']

            if 'cardinality' in constraint.keys():
                # 如果cardinality达不到要求，那就不退出
                flag = flag or curr_card <= constraint['cardinality']

            return flag

        join_order_list = self.determine_merge_order(origin_meta_list)

        init_config = get_config_init()
        current_config = init_config
        # print("build_dynamic_mv_on_constraint: current_config = {}.".format(current_config))
        # test_cnt = 10    # 进行三次迭代测试
        flag = False

        while True:
            # if test_cnt == 0:
            #     break
            # test_cnt -= 1

            current_meta_list = apply_config(current_config)
            current_data_list = self.fetch_data_list(src_meta_list = current_meta_list)

            # print("current_meta_list = {}.".format(current_meta_list))
            # print("current_data_list's length = {}.".format([len(item) for item in current_data_list]))

            ts = time.time()
            merged_df = self.merge_all_tables(single_meta_list = current_meta_list, \
                data_list = current_data_list, order_list = join_order_list)
            te = time.time()
            delta_time = te - ts

            # print("delta_time = {}. len(merged_df) = {}.".format(delta_time, len(merged_df)))
            if constraint_eval(delta_time, len(merged_df)) == False:
                # 满足条件以后直接退出
                break
            
            if flag == True:
                # 已达成要求，直接退出
                break
            # 完成config的迭代
            curr_metrics = {
                "time": te - ts,
                "cardinality": len(merged_df)
            }
            current_config, flag = config_iterate(curr_config=current_config, curr_metrics=curr_metrics)

        extra_meta_filter = []
        for k, v in current_config.items():
            schema_name, column_name = k
            ratio, start_idx, end_idx = v
            alias_name = self.alias_mapping[schema_name]
            start_val, end_val = predicate_transform(grid_config[k][0], start_idx, end_idx)

            # print(f"build_dynamic_mv.final: k = {k}. start_idx = {start_idx}. end_idx = {end_idx}. total_len = {len(grid_config[k][0])} "
            #       f"start_val = {start_val}. end_val = {end_val}. ratio = {ratio}.")
            
            extra_meta_filter.append((alias_name, column_name, start_val, end_val))

        return merged_df, extra_meta_filter, current_config
        # return None, extra_meta

    def build_mv_on_meta(self, target_meta, mv_hint = None):
        """
        {Description}
        
        Args:
            target_meta:
            mv_hint:
        Returns:
            merged_df:
        """
        # merged_df = pd.DataFrame([])
        # 创建所有单表的meta
        single_meta_list = self.meta_decompose(target_meta)
        # print("mv_meta = {}.".format(target_meta))
        # print("single_meta_list = {}.".format(single_meta_list))

        single_table_list = self.fetch_data_list(src_meta_list = single_meta_list)
        join_order_list = self.determine_merge_order(single_meta_list)

        merged_df = self.merge_all_tables(single_meta_list = single_meta_list, \
            data_list = single_table_list, order_list = join_order_list)
        
        return merged_df

    def meta_decompose(self, target_meta):
        """
        将一个复合的meta信息，分解成单表的meta

        Args:
            target_meta:
        Returns:
            meta_list:
        """
        meta_list = []
        schema_list, filter_list = target_meta
        def aggregate_corr_cond(schema_name, filter_list):
            schema_alias = self.alias_mapping[schema_name]
            res_list = []
            for item in filter_list:
                if item[0] == schema_alias:
                    res_list.append(item)
            return res_list
        
        for schema in schema_list:
            local_schema_list = [schema,]
            local_filter_list = aggregate_corr_cond(schema, filter_list)
            meta_list.append((local_schema_list, local_filter_list))

        return meta_list

    def determine_merge_order(self, single_meta_list):
        """
        决定合并的顺序

        Args:
            single_meta_list:
        Returns:
            index_list:
        """
        index_list = []
        src_meta_list = single_meta_list
        index_all = range(len(src_meta_list))
        foreign_mapping = workload_spec.get_spec_foreign_mapping(workload_name=self.workload)

        # 首先判断两表之前是否可以连通，然后每一次都加一张可以和当前表连接的表

        def build_join_graph(src_meta_list):
            edge_set = set()
            for i in range(len(src_meta_list)):
                for j in range(i + 1, len(src_meta_list)):
                    left_meta, right_meta = src_meta_list[i], src_meta_list[j]
                    left_on, right_on = self.data_manager.infer_join_conditions(\
                        left_meta, right_meta, foreign_mapping=foreign_mapping)
                    if left_on is not None and len(left_on) > 0:
                        edge_set.add((i, j))
                        edge_set.add((j, i))
            return edge_set

        edge_set = build_join_graph(src_meta_list)
        # print(f"determine_merge_order: edge_set = {edge_set}. schema_list = {[item[0] for item in src_meta_list]}.")

        def can_connect(idx, edge_set):
            flag = False
            for src_idx in index_list:
                if (src_idx, idx) in edge_set:
                    flag = True
                    break
            return flag

        def find_next_obj_index(index_list):
            for idx in index_all:
                if idx in index_list:
                    # 已经在里面了，直接跳过
                    continue
                elif can_connect(idx, edge_set=edge_set) == True:
                    # 判断能否join起来
                    return idx
            # 全部失败，理论上不可能存在的情况
            return -1

        # 直接添加第一个对象
        index_list.append(0)
        for _ in src_meta_list[1:]:
            # meta_aggregation
            next_obj_index = find_next_obj_index(index_list)
            index_list.append(next_obj_index)
        return index_list
    
    # @timeout(seconds=60)    # 限制程序最长的运行时间 
    def merge_all_tables(self, single_meta_list, data_list, order_list):
        """
        合并所有的单表
        
        Args:
            single_meta_list:
            data_list:
            order_list:
        Returns:
            all_tables_df:
        """
        first_idx = order_list[0]
        all_tables_df = data_list[first_idx]    # 初始的第一张表
        src_meta_list = single_meta_list

        all_tables_meta = src_meta_list[first_idx]
        foreign_mapping = workload_spec.get_spec_foreign_mapping(workload_name=self.workload)

        for curr_index in order_list[1:]:
            curr_meta = src_meta_list[curr_index]
            left_on, right_on = self.data_manager.infer_join_conditions(left_meta = all_tables_meta, \
                right_meta = curr_meta, foreign_mapping=foreign_mapping, allow_empty=False)          # 推断两边join的条件
            all_tables_meta = meta_merge(all_tables_meta, curr_meta)
            curr_df = data_list[curr_index]     # 获取待join的dataframe
            all_tables_df = self.merge_one_step(left_obj = all_tables_df, \
                right_obj = curr_df, left_on = left_on, right_on = right_on)

        return all_tables_df
    
    @timeout(seconds=60)    # 限制程序最长的运行时间 
    def merge_all_tables_under_limit(self, single_meta_list, data_list, order_list):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        return self.merge_all_tables(single_meta_list, data_list, order_list)

    def merge_one_step(self, left_obj: pd.DataFrame, right_obj: pd.DataFrame, left_on, right_on):
        """
        {Description}
    
        Args:
            left_obj:
            right_obj:
        Returns:
            joined_obj:
        """
        # joined_obj = pd.DataFrame([])
        # print("merge_one_step: left_on = {}. right_on = {}.".format(left_on, right_on))
        # print("left_obj = {}".format(left_obj.head()))
        # print("right_obj = {}".format(right_obj.head()))
        # print("left_obj's length = {}".format(len(left_obj)))
        # print("right_obj's length = {}".format(len(right_obj)))

        # 处理left_obj和right_obj，执行join_key上的dropna操作
        left_obj = left_obj.dropna(how="any", subset=left_on)
        right_obj = right_obj.dropna(how="any", subset=right_on)

        try:
            joined_obj = pd.merge(left = left_obj, right = right_obj, how="inner", 
                left_on=left_on, right_on=right_on)
        except IndexError as e:
            print(f"merge_one_step: meet IndexError. left_on = {left_on}. right_on = {right_on}. "\
                  f"left_cols = {left_obj.columns}. right_cols = {right_obj.columns}.")
            raise e
        return joined_obj


    def fetch_data_list(self, src_meta_list):
        """
        获取具体的数据到内存(保存成列表)，根据meta的性质去选择从data_manager
        或者mv_manager中获取数据

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        data_list = []
        for curr_meta in src_meta_list:
            tbl_name = curr_meta[0][0]
            table_df = self.data_manager.load_table_with_prefix(tbl_name)   # 加载表的dataframe

            # 删除字符串类型的数据
            cols_to_drop = table_df.select_dtypes(include=['object']).columns
            table_df.drop(columns=cols_to_drop, inplace=True)

            # print("mv_builder.fetch_data_list: curr_meta = {}.".format(curr_meta))
            # print("mv_builder.fetch_data_list: table_df = \n{}.".format(table_df.head()))

            # 应用filter信息筛选数据
            filter_df = conditions_apply(table_df, curr_meta[1])
            # 判断数据单表获取的结果正确性
            # print("len(filter_df) = {}. curr_meta = {}.".format(len(filter_df), curr_meta))
            data_list.append(filter_df)    # 从data_manager中进行加载

        return data_list


# %%
