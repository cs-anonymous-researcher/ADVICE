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
# import ce_adapter.remote_caller
import ce_adapter.remote_caller as remote_caller
from base import BaseCaller
from utility import workload_parser
from query import query_construction
from ce_adapter.case_process import stats_name_inverse, stats_name_mapping, default_mapping

workload_option = {
    "job": {
        "ip_address": "101.6.96.160",
        "port": "30005",
        "workload": "job"
    },
    "stats": {
        "ip_address": "101.6.96.160",
        "port": "30005",
        "workload": "stats",
    },
    "music": {
        "query_path": "",
        "label_path": "",
        "result_path": ""
    }
}

# join构造函数的映射
query_func_mapping = {
    "job": {
        "before": default_mapping,
        "after": default_mapping,
        "schema": query_construction.JOB_schema_constructor,
        "join": query_construction.JOB_join_constructor
    },
    "stats": {
        "before": stats_name_inverse,
        "after": stats_name_mapping,
        "schema": query_construction.stats_schema_constructor,
        "join": query_construction.stats_join_constructor
    }
}


def get_Neurocard_caller_instance(workload = "job"):
    """
    {Description}
    
    Args:
        workload:
    Returns:
        res1:
        res2:
    """
    return NeurocardCaller(**workload_option[workload])

class NeurocardCaller(BaseCaller):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, ip_address = "101.6.96.160", port = "30005", workload = "job"):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        self.workload = workload
        self.remote = remote_caller.NeuroCardRemote(ip_address, port)
        
        self.schema_func = query_func_mapping[workload]["schema"]
        self.join_func = query_func_mapping[workload]["join"]
        self.before_func = query_func_mapping[workload]["before"]
        self.after_func = query_func_mapping[workload]["after"]

        self.abbr_mapping = query_construction.abbr_option[workload]

        # 调用初始化
        self.initialization()

    def initialization(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        self.remote.launch_service(workload=self.workload)

    def get_estimation_in_batch(self, query_list, label_list):
        """
        {Description}

        Args:
            query_list: 查询列表
            label_list:
        Returns:
            card_list:
        """
        query_list = self.transform_workload(query_list=query_list)
        return self.remote.get_cardinalities(\
            sql_list = query_list, label_list = label_list)


    def get_estimation(self, sql_text, label):
        """
        {Description}

        Args:
            sql_text: 查询文本
            label:
        Returns:
            card:
        """
        return self.remote.get_cardinality(sql_text, label)
    

    def transform_query(self, query_text):
        """
        将query转换成neurocard能理解的形式
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        query_text = self.before_func(query_text)   # 先转成全部小写的

        # neurocard_template = """{schemas}#{joins}#{filter}#{cardinality}"""
        neurocard_template = """{schemas}|{joins}|{filter}|{cardinality}"""     # 使用新的分隔符

        pesudo_card = 1000
        parser = workload_parser.SQLParser(sql_text=query_text)
        # 提取query_meta
        query_meta = parser.generate_meta()

        neurocard_query = ""

        # 由
        schema_list, filter_list = query_meta        
        
        # 处理schema
        schema_res = self.schema_func(schema_list=schema_list, abbr_mapping=self.abbr_mapping)
        schema_str = ",".join(["{} {}".\
            format(s, self.abbr_mapping[s]) for s in schema_res])
        
        # 处理join
        join_cond_list = self.join_func(schema_list=schema_list, abbr_mapping=self.abbr_mapping)

        # 处理filter
        item_list = []
        for cond_tuple in filter_list:
            table_abbr, col_name, lower_bound, upper_bound = cond_tuple

            if lower_bound is not None:
                item_list.append("{}.{},>=,{}".format(table_abbr, col_name, lower_bound))
            if upper_bound is not None:
                item_list.append("{}.{},<=,{}".format(table_abbr, col_name, upper_bound))

        neurocard_query = neurocard_template.format(schemas = schema_str, \
            joins = ",".join(join_cond_list), filter = ",".join(item_list), cardinality = pesudo_card)
        
        neurocard_query = self.after_func(neurocard_query)  # 再转成大写首字母的
        print("neurocard_query = {}.".format(neurocard_query))
        return neurocard_query

    def transform_workload(self, query_list):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        return [self.transform_query(query_text) for query_text in query_list]
