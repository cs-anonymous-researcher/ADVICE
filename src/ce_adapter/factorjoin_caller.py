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

import ce_adapter.remote_caller as remote_caller
from base import BaseCaller
from utility import workload_parser
from query import query_construction
from ce_adapter.case_process import stats_name_inverse, stats_name_mapping, default_mapping

workload_option = {
    "job": {
        "ip_address": "101.6.96.160",
        "port": "30005",
        "workload": "job",
        "mapping_func": default_mapping
    },
    "stats": {
        "ip_address": "101.6.96.160",
        "port": "30005",
        "workload": "stats",
        "mapping_func": stats_name_mapping
    },
    "music": {
        "query_path": "",
        "label_path": "",
        "result_path": ""
    }
}


def get_FactorJoin_caller_instance(workload = "job"):
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    return FactorJoinCaller(**workload_option[workload])


class FactorJoinCaller(BaseCaller):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, ip_address = "101.6.96.160", port = "30005", \
                 workload = "job", mapping_func = default_mapping):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        self.workload = workload
        self.remote = remote_caller.FactorJoinRemote(ip_address, port)
        self.mapping_func = mapping_func

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
        query_list = self.mapping_func(query_list)          # 完成名字大小写的映射
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
    
