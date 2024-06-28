#!/usr/bin/env python
# -*- coding: utf-8 -*-


# %%

import numpy as np
import pandas as pd
from os.path import join as p_join
from matplotlib import pyplot as plt
import json, os, time, re, math
import pickle

# %%
from utility import workload_spec, utils
from utility.workload_spec import dsb_join_origin, dsb_join_constructor, dsb_schema_constructor

# %%
def JOB_join_constructor(schema_list, abbr_mapping):
    """
    针对JOB-light workload的连接条件构建
    
    Args:
        schema_list:
        abbr_mapping:
    Returns:
        item_list:
    """
    if len(schema_list) == 1:
        # 只有一张表就直接返回
        return []
    item_list = []
    for s in schema_list:
        if s == 'title':
            continue
        else:
            item_list.append("{}.movie_id=t.id".format(abbr_mapping[s]))
    # return " AND ".join(item_list)
    return item_list

# %%

"""
出于简化起见，在这里我们只考虑一张表最多包含一个外键的情况

stats workload下主要包含了

TODO: 增加一个表具有多个外键的情况
"""

# %%

def JOB_schema_constructor(schema_list: list, abbr_mapping: dict):
    """
    直接采用取巧的方法
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    schema_res = deepcopy(schema_list)
    if len(schema_res) == 1:
        # 单表情况下特判
        return schema_res
    
    if 'title' not in schema_res:
        schema_res.append('title')
    return schema_res


def stats_schema_constructor(schema_list, abbr_mapping):
    """
    stats workload中需要添加多少的主键表
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    schema_res = deepcopy(schema_list)
    if len(schema_res) == 1:
        # 单表情况下特判
        return schema_res
    
    for src_tbl, v in workload_spec.stats_foreign_mapping.items():
        src_col, ref_tbl, ref_col = v
        
        if src_tbl == "posts":
            # 出现posts的时候直接跳过，依靠其他schema添加user表的reference
            continue
        if src_tbl in schema_list and ref_tbl not in schema_res:    # 保证只添加一次
            # 如果主键表不存在的话，直接添加
            # schema_res.append(abbr_mapping[ref_tbl])
            schema_res.append(ref_tbl)
    return schema_res



# %%


def stats_join_constructor(schema_list, abbr_mapping):
    """
    针对stats workload的连接条件构建
    
    Args:
        schema_list:
        abbr_mapping:
    Returns:
        item_list:
    """
    # print("current stats_foreign_mapping = {}.".format(workload_spec.stats_foreign_mapping))
    if len(schema_list) == 1:
        # 只有一张表就直接返回
        return []
    item_list = []
    
    # 判断posts和users是否同时出现
    if "posts" in schema_list and "users" in schema_list:
        item_list.append("p.owneruserid=u.id")

    # 将所有表和foreign table连接起来
    for s in schema_list:
        if s in ["posts", "users"]:
            continue
        else:
            src_col, ref_table, ref_col = workload_spec.stats_foreign_mapping[s]
            # 添加和主键表的连接
            item_list.append("{}.{}={}.{}".\
                format(abbr_mapping[s], src_col, abbr_mapping[ref_table], ref_col)) 
    return item_list


def job_light_join_origin(schema_list, abbr_mapping):
    """
    针对job-light的workload，不考虑主键表缺失的问题，直接把现有的表连接起来，join等价类中第一个表会被视为主键表
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    item_list = []
    if "title" in schema_list:
        # 包含title的情况
        for s in schema_list:
            if s == "title":
                continue
            else:
                src_col, ref_table, ref_col = workload_spec.job_light_foreign_mapping[s]
                item_list.append("{}.{}={}.{}".format(abbr_mapping[s], src_col, abbr_mapping[ref_table], ref_col))
    else:
        # 不包含title的情况
        ref_col_dict = {}

        for s in schema_list:
            try:
                src_col, ref_table, ref_col = workload_spec.job_light_foreign_mapping[s]
            except KeyError as e:
                print(f"job_light_join_origin: meet KeyError. schema_list = {schema_list}.")
                raise e
            # 添加和主键表的连接
            if (ref_table, ref_col) not in ref_col_dict.keys():
                # 如果是第一次出现的等价类，设置连接的参考表
                ref_col_dict[(ref_table, ref_col)] = s, src_col
            else:
                join_tbl, join_col = ref_col_dict[(ref_table, ref_col)]
                item_list.append("{}.{}={}.{}".\
                    format(abbr_mapping[s], src_col, abbr_mapping[join_tbl], join_col)) 

    return item_list


def stats_join_origin(schema_list, abbr_mapping):
    """
    针对stats的workload，不考虑主键表缺失的问题，直接把现有的表连接起来，join等价类中第一个表会被视为主键表
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    # print("current stats_foreign_mapping = {}.".format(workload_spec.stats_foreign_mapping))
    # print("schema_list = {}".format(schema_list))

    if len(schema_list) == 1:
        # 只有一张表就直接返回
        return []
    item_list = []
    
    # 判断posts和users是否同时出现
    if "posts" in schema_list and "users" in schema_list:
        item_list.append("p.owneruserid=u.id")

    ref_col_dict = {}
    # 将所有表和foreign table连接起来
    for s in schema_list:
        if s in ["posts", "users"]:
            continue
        else:
            src_col, ref_table, ref_col = workload_spec.stats_foreign_mapping[s]
            # print("stats_join_origin: src_table = {}, src_col = {}, ref_table = {}, ref_col = {}".\
            #     format(s, src_col, ref_table, ref_col))
            # 判断参考的表是否存在
            if ref_table in schema_list: 
                # 如果存在的话，照之前的策略处理
                item_list.append("{}.{}={}.{}".\
                    format(abbr_mapping[s], src_col, abbr_mapping[ref_table], ref_col)) 
            else:
                # 添加和主键表的连接
                if (ref_table, ref_col) not in ref_col_dict.keys():
                    # 如果是第一次出现的等价类，设置连接的参考表
                    ref_col_dict[(ref_table, ref_col)] = s, src_col
                else:
                    join_tbl, join_col = ref_col_dict[(ref_table, ref_col)]
                    item_list.append("{}.{}={}.{}".\
                        format(abbr_mapping[s], src_col, abbr_mapping[join_tbl], join_col)) 

    # print("ref_col_dict = {}.".format(ref_col_dict))
    # 特殊处理posts的问题
    user_exist = False
    foreign_tbl, foreign_col = "", ""
    for ref_table, ref_col in ref_col_dict.keys():
        if ref_table == "users" and ref_col == "id":
            user_exist = True
            foreign_tbl, foreign_col = ref_col_dict[(ref_table, ref_col)]
            break

    if user_exist == True and "posts" in schema_list:
        item_list.append("p.owneruserid={}.{}".format(abbr_mapping[foreign_tbl], foreign_col))

    return item_list



def build_filter_list(table_abbr, column_list, param_list):
    """
    构造conditions，generate_single_query的辅助函数
    """
    filter_list = []
    for col, param in zip(column_list, param_list):
        filter_list.append((table_abbr, \
            "{}_{}".format(table_abbr, col), param[0], param[1]))
    return filter_list

def empty_meta():
    return deepcopy([]), deepcopy([])

JOB_abbr = {
    "cast_info": "ci",
    "movie_keyword": "mk",
    "movie_companies": "mc",
    "company_type": "ct",
    "title": "t",
    "company_name": "cn",
    "keyword": "k",
    "movie_info_idx": "mi_idx",
    "info_type": "it",
    "movie_info": "mi"
}

STATS_abbr = {
    "posts": "p",
    "votes": "v",
    "badges": "b",
    "users": "u",
    "posthistory": "ph",
    "postlinks": "pl",
    "tags": "t",
    "comments": "c"
}

RELEASE_abbr = {
    "release": "r",
    "medium": "m",
    "release_country": "rc",
    "release_tag": "rt",
    "release_label": "rl",
    # 
    "release_group": "rg",
    "artist_credit": "ac",
    "release_meta": "rm"
}

DSB_abbr = {
    "store_sales": "ss",
    "catalog_sales": "cs",
    "web_sales": "ws",
    "store_returns": "sr",
    "catalog_returns": "cr",
    "web_returns": "wr",
    "customer_demographics": "cd",
    "customer": "c",
    "customer_address": "ca"
}

abbr_option = {
    "job": JOB_abbr,
    "stats": STATS_abbr,
    "release": RELEASE_abbr,
    "dsb": DSB_abbr
}


def get_alias_mapping(workload):
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    return abbr_option[workload]

def get_alias_list(query_meta, workload):
    """
    {Description}

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    alias_mapping = abbr_option[workload]
    return [alias_mapping[s] for s in query_meta[0]]


# %%
from copy import copy, deepcopy

query_template = """SELECT COUNT(*) FROM {schemas} WHERE {{conditions}};"""

# %%

def parse_compound_column_name(col_str, workload="job", alias_reverse:dict = None):
    if alias_reverse is None:
        alias_mapping = abbr_option[workload]
    else:
        alias_mapping = {}
        for k, v in alias_reverse.items():
            alias_mapping[v] = k
            
    curr_alias, curr_schema = "", ""
    for k, v in alias_mapping.items():
        if col_str.startswith(v) and len(curr_alias) < len(v):
            curr_alias, curr_schema = v, k

    col_name = col_str[len(curr_alias) + 1: ]
    # print("curr_schema = {}. col_name = {}.".format(curr_schema, col_name))
    return curr_schema, col_name


def parse_compound_column_list(col_str_list, workload="job", alias_reverse:dict = None):
    return [parse_compound_column_name(col_str, workload, alias_reverse) for col_str in col_str_list]

# %% 生成原始的查询

# @utils.timing_decorator
def construct_origin_query(query_meta, workload):
    """
    生成原始的查询，不考虑主键缺失的问题，主要是最后的查询
    不应该包含主键表。
    
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    schema_list, filter_list = query_meta
    if workload == "stats":
        abbr_mapping = STATS_abbr
        local_join_func = stats_join_origin
    elif workload == "job":
        abbr_mapping = JOB_abbr
        local_join_func = job_light_join_origin
    elif workload == "dsb":
        abbr_mapping = DSB_abbr
        local_join_func = dsb_join_origin
    else:
        raise ValueError(f"construct_origin_query: Unsupported workload = {workload}.")
    
    # condition_str的构造
    item_list = []
    for cond_tuple in filter_list:
        try:
            table_abbr, col_name, lower_bound, upper_bound = cond_tuple
        except ValueError as e:
            print(f"construct_origin_query: cond_tuple = {cond_tuple}")
            raise e
        if lower_bound == upper_bound:  # 考虑等于的情况
            item_list.append("{}.{}={}".format(table_abbr, col_name, lower_bound))
            continue
        else:
            if lower_bound is not None:
                item_list.append("{}.{}>={}".format(table_abbr, col_name, lower_bound))
            if upper_bound is not None:
                item_list.append("{}.{}<={}".format(table_abbr, col_name, upper_bound))

    # 单表的情况
    if len(query_meta[0]) == 1:
        if len(item_list) == 0:
            # 没有condition的情况
            template = "SELECT COUNT(*) FROM {schemas}"
            s = query_meta[0][0]
            schema_result = "{} {}".format(s, abbr_mapping[s])
            return template.format(schemas = schema_result)
        else:
            # 存在condition的情况
            template = "SELECT COUNT(*) FROM {schemas} WHERE {all_conditions};"
            s = query_meta[0][0]
            schema_result = "{} {}".format(s, abbr_mapping[s])
            return template.format(schemas = schema_result, all_conditions = " AND ".join(item_list))
    else:
        # 多表的情况
        template = "SELECT COUNT(*) FROM {schemas} WHERE {all_conditions};"
        # schema_str的构造
        schema_result = ",".join(["{} {}".format(s, abbr_mapping[s]) for s in query_meta[0]])
        # join_str的构造
        join_result = local_join_func(schema_list=schema_list, abbr_mapping=abbr_mapping)
        all_conditions = item_list + join_result
        return template.format(schemas = schema_result, all_conditions = " AND ".join(all_conditions))

# %%

def construct_verification_query(query_meta, workload):
    """
    构造用于验证行数的查询

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    schema_list, filter_list = query_meta
    if workload == "stats":
        abbr_mapping = STATS_abbr
        local_join_func = stats_join_origin
    elif workload == "job":
        abbr_mapping = JOB_abbr
        local_join_func = job_light_join_origin
    elif workload == "dsb":
        abbr_mapping = DSB_abbr
        local_join_func = dsb_join_origin
    else:
        raise ValueError(f"construct_origin_query: Unsupported workload = {workload}.")
    
    # condition_str的构造
    item_list = []
    for cond_tuple in filter_list:
        try:
            table_abbr, col_name, lower_bound, upper_bound = cond_tuple
        except ValueError as e:
            print(f"construct_origin_query: cond_tuple = {cond_tuple}")
            raise e
        item_list.append("{}.{} is not NULL".format(table_abbr, col_name))

    # 单表的情况
    if len(query_meta[0]) == 1:
        if len(item_list) == 0:
            # 没有condition的情况
            template = "SELECT COUNT(*) FROM {schemas}"
            s = query_meta[0][0]
            schema_result = "{} {}".format(s, abbr_mapping[s])
            return template.format(schemas = schema_result)
        else:
            # 存在condition的情况
            template = "SELECT COUNT(*) FROM {schemas} WHERE {all_conditions};"
            s = query_meta[0][0]
            schema_result = "{} {}".format(s, abbr_mapping[s])
            return template.format(schemas = schema_result, all_conditions = " AND ".join(item_list))
    else:
        # 多表的情况
        template = "SELECT COUNT(*) FROM {schemas} WHERE {all_conditions};"
        # schema_str的构造
        schema_result = ",".join(["{} {}".format(s, abbr_mapping[s]) for s in query_meta[0]])
        # join_str的构造
        join_result = local_join_func(schema_list=schema_list, abbr_mapping=abbr_mapping)
        all_conditions = item_list + join_result
        return template.format(schemas = schema_result, all_conditions = " AND ".join(all_conditions))


# %%

class SingleWrapper(object):
    """
    根据内存中的DataFrame批量的生成查询(单个DataFrame)，主要针对的是需要获取基数的情况，因为
    某些基数估计方法必须保证主键的存在

    Members:
        field1:
        field2:
    """

    def __init__(self, single_df, single_meta, schema_func = JOB_schema_constructor, join_func = \
        JOB_join_constructor, abbr_mapping = JOB_abbr, mode = "DeepDB"):
        """
        {Description}

        Args:
            single_df: 
            single_meta: 
            schema_func: schema函数，用于补充缺失的schema
            join_func: join函数，用来构建JOIN条件
            abbr_mapping: 
            mode: 
        """
        self.schema_func = schema_func
        self.join_func = join_func

        self.single_df = single_df
        self.single_meta = single_meta

        self.existing_join_list = []
        self.existing_condition_list = []
        self.abbr_mapping = abbr_mapping
        if mode == "DeepDB":
            print("mode = {}".format(mode))
        elif mode == "NeuroCard":
            print("mode = {}".format(mode))
        elif mode == "Default":
            print("mode = {}".format(mode))
        else:
            raise ValueError("Unsupported mode: {}".format(mode))

        self.mode = mode
        self.parse_meta(single_meta, mode)

    def replace_df(self, new_df):
        """
        更新当前的数据
        
        Args:
            new_df:
        Returns:
            None
        """
        self.single_df = new_df

    def replace_meta(self, new_meta):
        """
        更新当前的元信息
        
        Args:
            new_meta:
        Returns:
            existing_condition_list:
        """
        self.existing_join_list = []
        self.existing_condition_list = []
        self.parse_meta(new_meta, self.mode)
        return self.existing_condition_list


    def parse_meta(self, single_meta, mode):
        """
        解析元信息

        Args:
            single_meta:
        Returns:
            existing_condition_list:
        """
        schema_list, filter_list = deepcopy(single_meta)
        self.schema_list = schema_list
        self.filter_list = filter_list
        mapping = self.abbr_mapping

        # 如果出现两张表以上，需要考虑加入主键表
        schema_res = self.schema_func(schema_list=schema_list, \
            abbr_mapping = self.abbr_mapping)
        if len(schema_list) >= 2:
            schema_str = ",".join(["{} {}".\
                format(s, mapping[s]) for s in schema_res])
        else:
            s = schema_list[0]
            schema_str = "{} {}".format(s, mapping[s])

        self.template = query_template.format(schemas = schema_str)
        # 添加join的信息
        # join_cond_list = self.construct_join(schema_list)
        join_cond_list = self.construct_join(schema_res)    # 新加主键表，再考虑join的问题

        self.existing_condition_list.extend(join_cond_list)

        # 添加已有filter的信息
        column_cond_list = self.construct_filter(filter_list)
        self.existing_condition_list.extend(column_cond_list)
            
        return self.existing_condition_list


    def generate_batch_queries(self, filter_list_batch):
        """
        生成批量的查询
        
        Args:
            filter_list_batch:
        Returns:
            query_list:
        """
        query_list = []
        for filter_list in filter_list_batch:
            curr_query = self.generate_single_query(filter_list)
            query_list.append(curr_query)
        return query_list

    def generate_single_query(self, filter_list):
        """
        根据条件生成单个query
        
        Args:
            filter_list:
        Returns:
            res1:
            res2:
        """
        query_text = self.template
        curr_condition_list = copy(self.existing_condition_list)
        curr_condition_list.extend(self.construct_filter(filter_list))
        query_text = query_text.format(conditions = \
            " AND ".join(curr_condition_list))

        return query_text


    def generate_current_query(self):
        """
        获得当前状态下的query
        
        Args:
            arg1:
            arg2:
        Returns:
            query_text:
        """
        return self.generate_single_query(filter_list = [])


    def construct_filter(self, filter_list):
        """
        {Description}
        
        Args:
            filter_list:
        Returns:
            item_list:
        """
        item_list = []
        mode = self.mode

        for cond_tuple in filter_list:
            table_abbr, col_name, lower_bound, upper_bound = cond_tuple
            # TODO: 处理None和=的情况
            if lower_bound == upper_bound:  # 考虑等于的情况
                item_list.append("{}.{}={}".format(table_abbr, col_name, lower_bound))
                continue
            else:
                if lower_bound is not None:
                    item_list.append("{}.{}>={}".format(table_abbr, col_name, lower_bound))
                if upper_bound is not None:
                    item_list.append("{}.{}<={}".format(table_abbr, col_name, upper_bound))

        return item_list


    def construct_join(self, schema_list):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        return self.join_func(schema_list=schema_list, \
            abbr_mapping=self.abbr_mapping)

# %%


def get_single_wrapper_instance(single_df, single_meta, workload="job", mode="DeepDB"):
    """
    获得single_wrapper的实例
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    if workload == "job":
        result_wrapper = SingleWrapper(single_df=single_df, single_meta=single_meta, \
            schema_func=JOB_schema_constructor, join_func=JOB_join_constructor, \
            abbr_mapping=JOB_abbr, mode=mode)
    elif workload == "stats":
        result_wrapper = SingleWrapper(single_df=single_df, single_meta=single_meta, \
            schema_func=stats_schema_constructor, join_func=stats_join_constructor, \
            abbr_mapping=STATS_abbr, mode=mode)
    elif workload == "dsb":
        result_wrapper = SingleWrapper(single_df=single_df, single_meta=single_meta, \
            schema_func=dsb_schema_constructor, join_func=dsb_join_constructor, \
            abbr_mapping=DSB_abbr, mode=mode)
    else:
        raise ValueError("get_single_wrapper_instance. Unsupported workload: {}".format(workload))
    return result_wrapper

# %%
