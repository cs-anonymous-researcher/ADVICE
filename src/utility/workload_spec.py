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

# %% 不同workload下schema的缩写

JOB_abbr = {
    "cast_info": "ci", "movie_keyword": "mk",
    "movie_companies": "mc", "company_type": "ct",
    "title": "t", "company_name": "cn",
    "keyword": "k", "movie_info_idx": "mi_idx",
    "info_type": "it", "movie_info": "mi"
}

STATS_abbr = {
    "posts": "p", "votes": "v",
    "badges": "b", "users": "u",
    "posthistory": "ph", "postlinks": "pl",
    "tags": "t", "comments": "c"
}

RELEASE_abbr = {
    "release": "r", "medium": "m", "release_country": "rc",
    "release_tag": "rt", "release_label": "rl",
    # 
    "release_group": "rg", "artist_credit": "ac", "release_meta": "rm"
}

DSB_abbr = {
    "store_sales": "ss", "catalog_sales": "cs", "web_sales": "ws",
    "store_returns": "sr", "catalog_returns": "cr",
    "web_returns": "wr", "customer_demographics": "cd",
    "customer": "c", "customer_address": "ca"
}

abbr_option = {
    "job": JOB_abbr, "stats": STATS_abbr,
    "release": RELEASE_abbr, "dsb": DSB_abbr
}

def get_alias_inverse(workload):
    """
    获得workload对应的alias反向映射
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    alias_mapping = abbr_option[workload]
    alias_inverse = {v: k for k, v in alias_mapping.items()}
    return alias_inverse


def list_alias_to_schema(alias_iter, workload):
    """
    {Description}

    Args:
        alias_iter:
        arg2:
    Returns:
        schema_tuple:
        return2:
    """
    alias_inverse = get_alias_inverse(workload)
    schema_tuple = tuple([alias_inverse[a] for a in alias_iter])
    return schema_tuple


# %% 表示workload下所有的schema

total_schema_dict = {
    "job": ["title", "movie_keyword", "movie_info", "movie_info_idx", "cast_info", "movie_companies"],
    "stats": ["badges", "comments", "posthistory", "postlinks", \
            "posts", "tags", "users", "votes"],
    "release": ["artist_credit", "medium", "release", "release_country", "release_group",\
                "release_label", "release_meta", "release_tag"],
    "dsb": ["store_sales", "catalog_sales", "web_sales", "store_returns", \
            "catalog_returns", "web_returns", "customer_demographics", "customer"]
}

workload2database = {
    "job": "imdbload",
    "stats": "stats",
    "release": "release_eight_tables",
    # "dsb": "dsb_2g"
    "dsb": "dsb_5g"
}

# %%

"""
之前stats_foreign_mapping，后来感觉link到users上可以获得更好的结果
"""

# stats_foreign_mapping = {
#     "tags": ("excerptpostid", "posts", "id"),
#     "postlinks": ("postid", "posts", "id"),
#     "posthistory": ("postid", "posts", "id"),
#     "comments": ("postid", "posts", "id"),
#     "posts": ("owneruserid", "users", "id"),   # posts在这里可以不依赖users存在
#     "badges": ("userid", "users", "id"),
#     "votes": ("userid", "users", "id")  
# }
# equivalent_stats = {
#     "users": ["badges", "votes"], 
#     "posts": ["tags", "postlinks", "posthistory", "comments"]
# }

# %%

# 比较公平的主/外键分配方式
stats_foreign_mapping = {
    "tags": ("excerptpostid", "posts", "id"),
    "postlinks": ("postid", "posts", "id"),
    "posthistory": ("postid", "posts", "id"),
    "comments": ("userid", "users", "id"),
    "posts": ("owneruserid", "users", "id"),   # posts在这里可以不依赖users存在
    "badges": ("userid", "users", "id"),
    "votes": ("postid", "posts", "id"),
}


equivalent_stats = {
    "users": ["badges", "comments"], 
    "posts": ["tags", "posthistory", "votes", "postlinks"]
}



def update_stats_foreign_mapping(tbl_name, col_name):
    """
    更新当前的foreign_mapping
    
    Args:
        tbl_name:
        col_name:
    Returns:
        current_foreign_mapping:
    """
    # print("update_stats_foreign_mapping: tbl_name = {}. col_name = {}.".\
    #     format(tbl_name, col_name))

    for src_tbl, src_col, ref_tbl, ref_col in stats_foreign_full:
        # print("src_tbl = {}. src_col = {}.".format(src_tbl, src_col))
        if tbl_name.lower() == src_tbl and src_col == col_name.lower():
            if stats_foreign_mapping[src_tbl][0] != src_col:
                pass
                # print("stats_foreign_mapping change:\nsrc_tbl = {}. src_col = \
                #     {}. ref_tbl = {}. ref_col = {}.".format(src_tbl, src_col, ref_tbl, ref_tbl))
            stats_foreign_mapping[src_tbl] = (src_col, ref_tbl, ref_col)
    return stats_foreign_mapping


def update_stats_multi_steps(tbl_col_list):
    """
    多步关于foreign_mapping的更新
    
    Args:
        tbl_col_list:
    Returns:
        res_foreign_mapping:
    """
    # print("tbl_col_list = {}.".format(tbl_col_list))
    for tbl_name, col_name in tbl_col_list:
        res_foreign_mapping = update_stats_foreign_mapping(tbl_name, col_name)
    # print("res_foreign_mapping = {}.".format(res_foreign_mapping))
    return res_foreign_mapping


# 完整的stats workload下的映射
stats_foreign_full = [
    ('tags', 'excerptpostid', 'posts', 'id'),
    ('postlinks', 'postid', 'posts', 'id'),
    ('postlinks', 'relatedpostid', 'posts', 'id'),
    ('posthistory', 'postid', 'posts', 'id'),
    ('votes', 'postid', 'posts', 'id'),
    ('comments', 'postid', 'posts', 'id'),

    ('badges', 'userid', 'users', 'id'),
    ('votes', 'userid', 'users', 'id'),
    ('posthistory', 'userid', 'users', 'id'),
    ('posts', 'owneruserid', 'users', 'id'),
    ('posts', 'lasteditoruserid', 'users', 'id'),
    ('comments', 'userid', 'users', 'id')
]

# %% JOB相关配置

job_light_foreign_mapping = {
    "cast_info" : ("movie_id", "title", "id"),
    "movie_info" : ("movie_id", "title", "id"),
    "movie_info_idx" : ("movie_id", "title", "id"),
    "movie_companies" : ("movie_id", "title", "id"),
    "movie_keyword" : ("movie_id", "title", "id")
}

equivalent_job_light = {
    "title": ["cast_info", "movie_info", "movie_info_idx", \
        "movie_companies", "movie_keyword"]
}


# %% Release数据集相关配置

release_foreign_mapping = {
    "release_label": ("release", "release", "id"),
    "release_tag": ("release", "release", "id"),
    "release_country": ("release", "release", "id"),
    "medium": ("release", "release", "id"),
    "release_meta": ("id", "release", "id"),
    # "release": ("release_group", "release_group", "id"), 采用反向避免dict引发的问题
    # "release": ("artist_credit", "artist_credit", "id")
    "release_group": ("id", "release", "release_group"),
    "artist_credit": ("id", "release", "artist_credit")
}

equivalent_release = {
    "release": ["release_label", "release_tag", "release_country", "medium"],
    "release_group": ["release"],
    "artist_credit": ["release"]
}
# %% DSB数据集相关配置

dsb_foreign_mapping = {
    "store_sales" : ("ss_cdemo_sk", "customer_demographics", "cd_demo_sk"),
    "catalog_sales" : ("cs_bill_cdemo_sk", "customer_demographics", "cd_demo_sk"),
    "catalog_returns" : ("cr_refunded_cdemo_sk", "customer_demographics", "cd_demo_sk"),
    "web_sales" : ("ws_bill_cdemo_sk", "customer_demographics", "cd_demo_sk"),
    "web_returns" : ("wr_refunded_cdemo_sk", "customer_demographics", "cd_demo_sk"),
    "store_returns": ("sr_cdemo_sk", "customer_demographics", "cd_demo_sk"),
    "customer": ("c_current_cdemo_sk", "customer_demographics", "cd_demo_sk"),
}


equivalent_dsb = {
    "customer_demographics": ["store_sales", "catalog_sales", "web_sales", "store_returns", \
        "catalog_returns", "web_returns", "customer"]
}

"""
select count(*) from store_returns, customer, web_returns, web_sales, catalog_returns where customer.c_current_cdemo_sk=store_returns.sr_cdemo_sk
and customer.c_current_cdemo_sk=web_returns.wr_refunded_cdemo_sk and web_sales.ws_bill_cdemo_sk=web_returns.wr_refunded_cdemo_sk
and catalog_returns.cr_refunded_cdemo_sk=customer.c_current_cdemo_sk;
"""

# %%

def get_spec_foreign_mapping(workload_name):
    if workload_name == "job" or workload_name == "job-light":
        return job_light_foreign_mapping
    elif workload_name == "stats":
        return stats_foreign_mapping
    elif workload_name == "release":
        return release_foreign_mapping
    elif workload_name == "dsb":
        return dsb_foreign_mapping
    else:
        raise ValueError(f"get_spec_foreign_mapping: workload = {workload_name}")

# %% release数据集查询生成构造的配置


def release_light_join_origin(schema_list, abbr_mapping):
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    item_list = []
    if "release" in schema_list:
        # 包含release的情况
        for s in schema_list:
            if s == "release":
                continue
            else:
                src_col, ref_table, ref_col = release_foreign_mapping[s]
                item_list.append("{}.{}={}.{}".format(abbr_mapping[s], src_col, abbr_mapping[ref_table], ref_col))
    else:
        # 不包含release的情况
        ref_col_dict = {}

        for s in schema_list:
            src_col, ref_table, ref_col = release_foreign_mapping[s]
            # 添加和主键表的连接
            if (ref_table, ref_col) not in ref_col_dict.keys():
                # 如果是第一次出现的等价类，设置连接的参考表
                ref_col_dict[(ref_table, ref_col)] = s, src_col
            else:
                join_tbl, join_col = ref_col_dict[(ref_table, ref_col)]
                item_list.append("{}.{}={}.{}".\
                    format(abbr_mapping[s], src_col, abbr_mapping[join_tbl], join_col)) 

    return item_list

def release_join_constructor(schema_list, abbr_mapping):
    """
    针对release workload的连接条件构建
    20230807: 需要额外考虑两张表
    
    Args:
        schema_list:
        abbr_mapping:
    Returns:
        item_list:
    """
    # print("release_join_constructor: stats_foreign_mapping = {}.".format(stats_foreign_mapping))
    if len(schema_list) == 1:
        # 只有一张表就直接返回
        return []
    item_list = []
    for s in schema_list:
        if s == 'release':
            continue
        else:
            #
            # item_list.append("{}.release=r.id".format(abbr_mapping[s]))
            src_col, ref_table, ref_col = release_foreign_mapping[s]
            # 添加和主键表的连接
            item_list.append("{}.{}={}.{}".\
                format(abbr_mapping[s], src_col, abbr_mapping[ref_table], ref_col)) 
    # return " AND ".join(item_list)
    return item_list


def release_schema_constructor(schema_list, abbr_mapping):
    """
    release workload中需要添加多少的主键表
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    schema_res = deepcopy(schema_list)
    for src_tbl, v in release_foreign_mapping.items():
        src_col, ref_tbl, ref_col = v
        if src_tbl in schema_list and ref_tbl not in schema_res:    # 保证只添加一次
            # 如果主键表不存在的话，直接添加
            # schema_res.append(abbr_mapping[ref_tbl])
            schema_res.append(ref_tbl)
    return schema_res

# %% DSB数据集查询生成构造的配置


def dsb_join_origin(schema_list, abbr_mapping):
    """
    {Description}

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    item_list = []
    if "customer_demographics" in schema_list:
        # 包含release的情况
        for s in schema_list:
            if s == "customer_demographics":
                continue
            else:
                src_col, ref_table, ref_col = dsb_foreign_mapping[s]
                item_list.append("{}.{}={}.{}".format(abbr_mapping[s], src_col, abbr_mapping[ref_table], ref_col))
    else:
        # 不包含release的情况
        ref_col_dict = {}

        for s in schema_list:
            src_col, ref_table, ref_col = dsb_foreign_mapping[s]
            # 添加和主键表的连接
            if (ref_table, ref_col) not in ref_col_dict.keys():
                # 如果是第一次出现的等价类，设置连接的参考表
                ref_col_dict[(ref_table, ref_col)] = s, src_col
            else:
                join_tbl, join_col = ref_col_dict[(ref_table, ref_col)]
                item_list.append("{}.{}={}.{}".\
                    format(abbr_mapping[s], src_col, abbr_mapping[join_tbl], join_col)) 

    return item_list


def dsb_join_constructor(schema_list, abbr_mapping):
    """
    {Description}

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    # print("release_join_constructor: stats_foreign_mapping = {}.".format(stats_foreign_mapping))
    if len(schema_list) == 1:
        # 只有一张表就直接返回
        return []
    item_list = []
    for s in schema_list:
        if s == 'customer_demographics':
            continue
        else:
            #
            # item_list.append("{}.release=r.id".format(abbr_mapping[s]))
            src_col, ref_table, ref_col = dsb_join_constructor[s]
            # 添加和主键表的连接
            item_list.append("{}.{}={}.{}".\
                format(abbr_mapping[s], src_col, abbr_mapping[ref_table], ref_col)) 
    return item_list


def dsb_schema_constructor(schema_list, abbr_mapping):
    """
    {Description}

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    schema_res = deepcopy(schema_list)

    if len(schema_res) == 1:
        # 单表情况下特判
        return schema_res
    
    for src_tbl, v in dsb_foreign_mapping.items():
        src_col, ref_tbl, ref_col = v
        if src_tbl in schema_list and ref_tbl not in schema_res:    # 保证只添加一次
            # 如果主键表不存在的话，直接添加
            # schema_res.append(abbr_mapping[ref_tbl])
            schema_res.append(ref_tbl)
    return schema_res

# %%
