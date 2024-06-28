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

from collections import OrderedDict

col_name_mapping_stats = OrderedDict({'reputation': 'Reputation', 
        'creationdate': 'CreationDate', 'views': 'Views', 'upvotes': 'UpVotes', 
        'downvotes': 'DownVotes', 'posttypeid': 'PostTypeId', 'score': 'Score', 
        'viewcount': 'ViewCount', 'owneruserid': 'OwnerUserId', 'answercount': 'AnswerCount', 
        'commentcount': 'CommentCount', 'favoritecount': 'FavoriteCount', 
        'lasteditoruserid': 'LastEditorUserId', 'excerptpostid': 'ExcerptPostId', 
        'relatedpostid': 'RelatedPostId', 'linktypeid': 'LinkTypeId', 
        'posthistorytypeid': 'PostHistoryTypeId', 'userid': 'UserId', 
        'votetypeid': 'VoteTypeId', 'bountyamount': 'BountyAmount', 'date': 'Date', 
        'count': 'Count', 'postid': 'PostId', 'id': 'Id'})

tbl_name_mapping_stats = OrderedDict({
    "postlinks": "postLinks",
    "posthistory": "postHistory"
})

col_name_inverse_stats = OrderedDict({})
tbl_name_inverse_stats = OrderedDict({})

for k, v in col_name_mapping_stats.items():
    col_name_inverse_stats[v] = k

for k, v in tbl_name_mapping_stats.items():
    tbl_name_inverse_stats[v] = k


def function_builder(col_name_mapping, tbl_name_mapping):
    def func_res(query_content):
        def kw_sub(in_str: str):
            for k, v in col_name_mapping.items():
                in_str = in_str.replace(k, v)

            for k, v in tbl_name_mapping.items():
                in_str = in_str.replace(k, v)
            return in_str

        if isinstance(query_content, (list, tuple)):
            return [kw_sub(query) for query in query_content]
        elif isinstance(query_content, str):
            return kw_sub(query_content)
        
    return func_res

def stats_name_mapping(query_content):
    """
    stats workload下对于名字的映射，从小写转换成大写的形式
    
    Args:
        query_content: 查询实例或者查询列表
        arg2:
    Returns:
        res1:
        res2:
    """
    col_name_mapping = col_name_mapping_stats
    tbl_name_mapping = tbl_name_mapping_stats

    # tbl_name_mapping = {
    #     "postlinks": "postLinks",
    #     "posthistory": "postHistory"
    # }

    def kw_sub(in_str: str):
        for k, v in col_name_mapping.items():
            in_str = in_str.replace(k, v)

        for k, v in tbl_name_mapping.items():
            in_str = in_str.replace(k, v)
        return in_str

    if isinstance(query_content, (list, tuple)):
        return [kw_sub(query) for query in query_content]
    elif isinstance(query_content, str):
        return kw_sub(query_content)


stats_name_inverse = function_builder(col_name_mapping=\
    col_name_inverse_stats, tbl_name_mapping=tbl_name_inverse_stats)


default_mapping = lambda a: a   # 默认的映射，不做出改变
