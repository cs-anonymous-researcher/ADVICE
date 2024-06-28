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
from ce_adapter.oceanbase_caller import get_Oceanbase_caller_instance
from ce_adapter.deepdb_caller import DeepDBCaller, get_DeepDB_caller_instance
from ce_adapter.neurocard_caller import NeurocardCaller, get_Neurocard_caller_instance
from ce_adapter.deepdb_server import DeepDBServer, get_DeepDB_server_instance
from ce_adapter.mscn_caller import get_MSCN_caller_instance
from ce_adapter.fcn_caller import get_FCN_caller_instance, get_FCNPool_caller_instance
from ce_adapter.oracle_caller import get_Oracle_caller_instance
from ce_adapter.deepdb_advance import get_DeepDB_advance_instance
from ce_adapter.factorjoin_caller import get_FactorJoin_caller_instance
from ce_adapter.sqlserver_caller import get_SQLServer_caller_instance

from ce_adapter.internal_caller import get_internal_caller_instance

from scipy.stats import gmean, hmean
# %%


# %%
class ResultFetcher(object):
    """
    获取具体的结果

    Members:
        field1:
        field2:
    """

    def __init__(self, call_type = "DeepDB", workload = "job"):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        # print("ResultFetcher initialization: call_type = {}. workload = {}.".\
        #     format(call_type, workload))
            
        self.ce_type = call_type
        if call_type == "DeepDB":
            # self.test_caller = DeepDBCaller()      
            self.test_caller = get_DeepDB_caller_instance(workload = workload) 
        elif call_type == "NeuroCard":
            # self.test_caller = NeurocardCaller()
            self.test_caller = get_Neurocard_caller_instance(workload = workload)
        elif call_type == "DeepDBServer":
            self.test_caller = get_DeepDB_server_instance(workload = workload)
        elif call_type.lower() == "mscn":
            self.test_caller = get_MSCN_caller_instance(workload = workload)
        elif call_type.lower() == "fcn":
            self.test_caller = get_FCN_caller_instance(workload=workload)
        elif call_type.lower() == "fcnpool":
            self.test_caller = get_FCNPool_caller_instance(workload=workload)
        elif call_type.lower() == "oracle":
            self.test_caller = get_Oracle_caller_instance(workload=workload)
        elif call_type.lower() == "oceanbase":
            self.test_caller = get_Oceanbase_caller_instance(workload = workload)
        elif call_type.lower() == "sqlserver":
            self.test_caller = get_SQLServer_caller_instance(workload=workload)
        elif call_type.lower() == "factorjoin":
            self.test_caller = get_FactorJoin_caller_instance(workload=workload)
        elif call_type.lower() == "deepdb_rdc":
            self.test_caller = get_DeepDB_advance_instance(workload=workload, option="rdc")
        elif call_type.lower() == "deepdb_jct":
            self.test_caller = get_DeepDB_advance_instance(workload=workload, option="jct")
        elif call_type.lower() == "internal":
            self.test_caller = get_internal_caller_instance(workload=workload)
        else:
            raise ValueError("Unsupported call_type: {}".format(call_type))


    def initialize_caller(self, ):
        """
        初始化基数调用器
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        self.test_caller.initialization()

    def get_pair_result(self, query_list, label_list):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        pass
    
    def get_one_result(self, query, label, with_origin = False):
        """
        获得单条query的结果
        
        Args:
            query:
            label:
            with_origin:
        Returns:
            q_error:
            pair:
        """
        query_list = [query,]   # 手动构造查询列表
        label_list = [label,]   # 手动构造标签列表
        if with_origin == False:
            q_error_list = self.get_result(query_list, label_list, False)
            print("get_one_result: q_error_list = {}".format(q_error_list))
            return q_error_list[0]
        else:
            q_error_list, pair_list = self.get_result(query_list, label_list, True)
            print("get_one_result: q_error_list = {}. pair_list = {}.".format(q_error_list, pair_list))
            return q_error_list[0], pair_list[0]


    def get_result(self, query_list, label_list, with_origin = False):
        """
        获得结果（估计误差）

        Args:
            query_list:
            label_list:
            with_origin:
        Returns:
            q_error_list:
            pair_list(optional):
        """
        result_list = self.test_caller.get_estimation_in_batch(query_list, label_list)
        if with_origin == False:
            return self.result_summary(label_list, result_list)
        else:
            return self.result_summary(label_list, result_list), list(zip(label_list, result_list))

    def get_one_side_q_error(self, query_list, label_list, mode="over-estimation"):
        """
        获得某一侧的q_error结果
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        estimation_list = self.test_caller.get_estimation_in_batch(query_list, label_list)
        if mode == "over-estimation":
            result_list = [i / (j + 1) for i, j in zip(estimation_list, label_list)]
        elif mode == "under-estimation":
            result_list = [j / (i + 1) for i, j in zip(label_list, estimation_list)]
        else:
            raise ValueError("Unsupported mode = {}".format(mode))

        return result_list

    def pseudo_label(self, query_list):
        """
        获得伪造的标签
        
        Args:
            query_list:
        Returns:
            label_list:
        """
        const_num = 1000
        return [const_num for _ in query_list]


    def get_card_estimation(self, query_list):
        """
        获得基数估计的结果
        
        Args:
            query_list:
        Returns:
            result_list:
        """
        label_list = self.pseudo_label(query_list)
        result_list = self.test_caller.\
            get_estimation_in_batch(query_list, label_list)
        
        # 全部转化成整型
        return [int(i) for i in result_list]

    def result_average(self, value_list, mode = "arithmetic"):
        """
        {Description}
        
        Args:
            value_list:
            mode:
        Returns:
            value: 平均值
        """
        if mode == "harmonic":
            return hmean(value_list)
        elif mode == "geometric":
            return gmean(value_list)
        elif mode == "arithmetic":
            return np.average(value_list)

    def result_summary(self, label_list, result_list):
        """
        {Description}

        Args:
            label_list:
            result_list:
        Returns:
            q_error_list:
        """
        def q_error_func(true_card, estimated_card):
            if true_card > estimated_card:
                return true_card / estimated_card
            else:
                return estimated_card / true_card
                
        q_error_list = [(idx, q_error_func(i, j)) for idx, (i,j) in enumerate(zip(label_list, result_list))]
        return q_error_list
        