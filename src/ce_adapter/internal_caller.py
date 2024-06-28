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
import psycopg2 as pg
from multiprocessing import Pool
from copy import deepcopy
from base import BaseCaller

# %%

JOB_config = {
    "user": "lianyuan",
    "host": "localhost",
    "password": "",
    "database": "imdbload", 
    "port": "6432"
}

STATS_config = {
    "user": "lianyuan",
    "host": "localhost",
    "password": "",
    "database": "stats", 
    "port": "6432"
}

RELEASE_config = {
    "user": "lianyuan",
    "host": "localhost",
    "password": "",
    "database": "stats", 
    "port": "6432"
}

workload_option = {
    "job": JOB_config,
    "stats": STATS_config,
    "release": RELEASE_config
}

class InternalCaller(BaseCaller):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, workload = "job"):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        self.db_config = workload_option[workload]
        print("self.db_config = {}.".format(self.db_config))

    def get_estimation_in_batch(self, query_list, label_list = []):
        """
        批量的获得基数估计结果

        Args:
            query_list:
            label_list:
        Returns:
            result_list:
        """
        conn = pg.connect(**self.db_config)
        plan_list = []
        def process_sql(query_text):
            if 'COUNT(*)' in query_text:
                query_text = query_text.replace("COUNT(*)", "*")
            elif 'count(*)' in query_text:
                query_text = query_text.replace("count(*)", "*")
            return query_text
            
        with conn.cursor() as cursor:
            for sql_text in query_list: 
                # print("sql_text = EXPLAIN {}".format(sql_text))
                cursor.execute("EXPLAIN (FORMAT JSON) " + process_sql(sql_text))
                plan = cursor.fetchall()
                plan_list.append(plan)

        result_list = [i[0][0][0]['Plan']['Plan Rows'] for i in plan_list]
        return result_list


    def get_estimation_by_query(self, subquery_dict, single_table_dict):
        """
        以query为单位获得基数的信息
        
        Args:
            subquery_dict:
            single_table_dict:
        Returns:
            subquery_res:
            single_table_res:
        """
        subquery_res = {}
        single_table_res = {}

        query_list = subquery_dict.keys()
        card_list = self.get_estimation_in_batch(query_list = query_list)

        for k, v in zip(query_list, card_list):
            subquery_res[k] = v

        query_list = single_table_dict.keys()
        card_list = self.get_estimation_in_batch(query_list = query_list)

        for k, v in zip(query_list, card_list):
            single_table_res[k] = v

        return subquery_res, single_table_res
        
# %%

def get_internal_caller_instance(workload):
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    return InternalCaller(workload=workload)

# %%
