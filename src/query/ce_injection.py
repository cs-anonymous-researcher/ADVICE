#!/usr/bin/env python
# -*- coding: utf-8 -*-


# %%

import numpy as np
import pandas as pd
from os.path import join as p_join
from matplotlib import pyplot as plt
import json, os, time, re, math
import pickle

# %% 自己实现的代码
from data_interaction import feedback, mv_management, postgres_connector
from query import query_construction
import importlib
from base import BaseHandler

# %%
class DeepDBHandler(BaseHandler):
    """
    面向DeepDB的估计基数获取

    Members:
        field1:
        field2:
    """

    def __init__(self, workload = "job", mode = "normal"):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        # TODO: 默认用的是JOB，后续得支持多个workload
        if mode == "normal":
            self.result_fetcher = feedback.ResultFetcher(call_type="DeepDB", workload=workload)
        elif mode == "server":
            self.result_fetcher = feedback.ResultFetcher(call_type="DeepDBServer", workload=workload)
        else:
            raise ValueError("Unsupported mode = {}.".format(mode))
        

    def estimate_subqueries(self, meta_list):
        """
        {Description}
        
        Args:
            meta_list:
        Returns:
            cardinality_list:
        """
        cardinality_list = []
        subquery_list = self.construct_subqueries(meta_list)
        cardinality_list = self.get_cardinalities(subquery_list)
        return cardinality_list

    def list_ravel(self, list_batch):
        """
        {Description}
    
        Args:
            list_batch:
        Returns:
            flatten_list:
            idx_list:
        """
        flatten_list, idx_list = [], [0,]
        for sub_list in list_batch:
            flatten_list.extend(sub_list)
            idx_list.append(idx_list[-1] + len(sub_list))

        return flatten_list, idx_list

    def list_unravel(self, flatten_list, idx_list):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        list_batch = []
        for start, end in zip():
            list_batch.append(flatten_list[start: end])
        return list_batch


    def estimate_subqueries_batch(self, meta_list_batch):
        """
        {Description}
        
        Args:
            meta_list_batch:
        Returns:
            cardinality_list_batch:
        """
        cardinality_list_batch = []
        flatten_meta_list, idx_list = self.list_ravel(meta_list_batch)
        flatten_cardinality_list = self.construct_subqueries(flatten_meta_list)
        cardinality_list_batch = self.list_unravel(flatten_cardinality_list, idx_list)

        return cardinality_list_batch

    def construct_subqueries(self, meta_list):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        subquery_list = []
        for query_meta in meta_list:
            subquery_list.append(self.generate_query(query_meta))
        return subquery_list

    def pseudo_dataframe(self,):
        """
        获得一个默认的DataFrame
        
        Args:
            None
        Returns:
            out_df:
        """
        return pd.DataFrame([])

    def get_query_wrapper(self, query_meta):
        single_df = self.pseudo_dataframe()
        if hasattr(self, 'query_wrapper') == False:
            self.query_wrapper = query_construction.\
                SingleWrapper(single_df, query_meta, mode="DeepDB")
        else:
            self.query_wrapper.replace_meta(new_meta = query_meta)
        return self.query_wrapper

    def generate_query(self, query_meta):
        """
        {Description}
        
        Args:
            query_meta:
        Returns:
            res1:
            res2:
        """
        single_df = self.pseudo_dataframe()
        # single_wrapper = query_construction.SingleWrapper(single_df, query_meta, mode="DeepDB")
        single_wrapper = self.get_query_wrapper(query_meta)
        return single_wrapper.generate_current_query()

    def get_cardinalities(self, query_list):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        return self.result_fetcher.get_card_estimation(query_list)

# %%

class NeuroCardHandler(BaseHandler):
    """
    面向NeuroCard的估计基数获取

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
        # TODO: 默认用的是JOB，后续得支持多个workload
        self.result_fetcher = feedback.ResultFetcher(call_type="NeuroCard", workload=workload)
        # # 自动初始化
        # self.result_fetcher.initialize_caller()

    def get_cardinalities(self, query_list):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        return self.result_fetcher.get_card_estimation(query_list)

# %%

class MSCNHandler(BaseHandler):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, workload):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        # TODO: 默认用的是JOB，后续得支持多个workload
        self.result_fetcher = feedback.ResultFetcher(call_type="MSCN", workload=workload)
        # 自动初始化
        self.result_fetcher.initialize_caller()

    def get_cardinalities(self, query_list):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        return self.result_fetcher.get_card_estimation(query_list)

# %%

class FCNHandler(BaseHandler):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, workload):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        # TODO: 默认用的是JOB，后续得支持多个workload
        self.result_fetcher = feedback.ResultFetcher(call_type="FCN", workload=workload)
        # 自动初始化
        self.result_fetcher.initialize_caller()

    def get_cardinalities(self, query_list):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        return self.result_fetcher.get_card_estimation(query_list)


# %%
class FCNPoolHandler(BaseHandler):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, workload):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        # TODO: 默认用的是JOB，后续得支持多个workload
        self.result_fetcher = feedback.ResultFetcher(call_type="FCNPool", workload=workload)
        # 自动初始化
        self.result_fetcher.initialize_caller()

    def get_cardinalities(self, query_list):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        return self.result_fetcher.get_card_estimation(query_list)

# %%
class OracleHandler(BaseHandler):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, workload):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        # TODO: 默认用的是JOB，后续得支持多个workload
        self.result_fetcher = feedback.ResultFetcher(call_type="Oracle", workload=workload)
        # 自动初始化
        self.result_fetcher.initialize_caller()

    def get_cardinalities(self, query_list):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        return self.result_fetcher.get_card_estimation(query_list)

# %%

class SQLServerHandler(BaseHandler):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, workload):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        # TODO: 默认用的是JOB，后续得支持多个workload
        self.result_fetcher = feedback.ResultFetcher(call_type="SQLServer", workload=workload)
        # 自动初始化
        self.result_fetcher.initialize_caller()

    def get_cardinalities(self, query_list):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        return self.result_fetcher.get_card_estimation(query_list)

# %%
class OceanbaseHandler(BaseHandler):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, workload):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        # TODO: 默认用的是JOB，后续得支持多个workload
        self.result_fetcher = feedback.ResultFetcher(call_type="Oceanbase", workload=workload)
        # 自动初始化
        self.result_fetcher.initialize_caller()

    def get_cardinalities(self, query_list):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        return self.result_fetcher.get_card_estimation(query_list)

# %%
class FactorJoinHandler(BaseHandler):
    def __init__(self, workload):
        self.result_fetcher = feedback.ResultFetcher(call_type="FactorJoin", workload=workload)
        # 自动初始化
        self.result_fetcher.initialize_caller()

    def get_cardinalities(self, query_list):
        return self.result_fetcher.get_card_estimation(query_list)

# %%
class DeepDBRDCHandler(BaseHandler):
    def __init__(self, workload):
        self.result_fetcher = feedback.ResultFetcher(call_type="DeepDB_RDC", workload=workload)
        # 自动初始化
        self.result_fetcher.initialize_caller()

    def get_cardinalities(self, query_list):
        return self.result_fetcher.get_card_estimation(query_list)

# %%
class DeepDBJCTHandler(BaseHandler):
    def __init__(self, workload):
        self.result_fetcher = feedback.ResultFetcher(call_type="DeepDB_JCT", workload=workload)
        # 自动初始化
        self.result_fetcher.initialize_caller()

    def get_cardinalities(self, query_list):
        return self.result_fetcher.get_card_estimation(query_list)


# %%
def make_internal_handler_ref():
    """
    {Description}

    Args:
        None
    Returns:
        function
    """
    ctrl_dict = {}
    def local_func(workload):
        if workload in ctrl_dict.keys():
            return ctrl_dict[workload]
        else:
            ctrl_dict[workload] = PGInternalHandler(workload = workload)
            return ctrl_dict[workload]
    return local_func


get_internal_handler = make_internal_handler_ref()


# %%
class PGInternalHandler(BaseHandler):
    """
    PostgreSQL内部基数估计获取，之所以需要这个，是发现PG在没有hint做join的时候，
    用到的基数和每次估一个子查询获得的基数，实际上是不一致的。因此针对PG内部估计器
    的研究，也需要借助pg_plan_hint实现。

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
        db_config = postgres_connector.workload_option[workload]
        self.db_conn = postgres_connector.Connector(**db_config)

    def initialization(self,):
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

    def estimate_subqueries(self, meta_list):
        """
        {Description}
        
        Args:
            meta_list:
        Returns:
            cardinality_list:
        """
        cardinality_list = []
        subquery_list = self.construct_subqueries(meta_list)
        cardinality_list = self.get_cardinalities(subquery_list)
        return cardinality_list
    
    def estimate_subqueries_batch(self, meta_list_batch):
        """
        {Description}
        
        Args:
            meta_list_batch:
        Returns:
            cardinality_list_batch:
        """
        cardinality_list_batch = []
        for meta_list in meta_list_batch:
            cardinality_list_batch.append(self.estimate_subqueries(meta_list))
        return cardinality_list_batch
        
    def construct_subqueries(self, meta_list):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        subquery_list = []
        for query_meta in meta_list:
            subquery_list.append(self.generate_query(query_meta))
        return subquery_list

    def get_query_wrapper(self, query_meta):
        single_df = self.pseudo_dataframe()
        if hasattr(self, 'query_wrapper') == False:
            self.query_wrapper = query_construction.\
                SingleWrapper(single_df, query_meta, mode="Default")
        else:
            self.query_wrapper.replace_meta(new_meta = query_meta)
        return self.query_wrapper

    def pseudo_dataframe(self,):
        """
        获得一个默认的DataFrame
        
        Args:
            None
        Returns:
            out_df:
        """
        return pd.DataFrame([])

    def generate_query(self, query_meta):
        """
        {Description}
        
        Args:
            query_meta:
        Returns:
            res1:
            res2:
        """
        single_wrapper = self.get_query_wrapper(query_meta)
        return single_wrapper.generate_current_query()

    def get_cardinalities(self, query_list):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        card_list = []
        for query_text in query_list:
            curr_card = self.db_conn.get_card_estimation(query_text)
            card_list.append(curr_card)

        return card_list

    def single_table_cardinalities(self, query_meta):
        """
        获得一个查询下所有的单表结果的基数
        
        Args:
            query_meta:
        Returns:
            single_meta_list:
            cardinality_list:
        """
        single_meta_list = mv_management.meta_decompose(in_meta = query_meta)   # 单表组成的meta列表
        cardinality_list = []

        query_list = self.construct_subqueries(single_meta_list)
        cardinality_list = self.get_cardinalities(query_list)

        return single_meta_list, cardinality_list

# %%

class TrueCardinalityHandler(BaseHandler):
    """
    数据库真实基数的获取

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
        db_config = postgres_connector.workload_option[workload]
        self.db_conn = postgres_connector.Connector(**db_config)


    def estimate_subqueries(self, meta_list):
        """
        {Description}
        
        Args:
            meta_list:
        Returns:
            cardinality_list:
        """
        cardinality_list = []
        subquery_list = self.construct_subqueries(meta_list)
        cardinality_list = self.get_cardinalities(subquery_list)
        return cardinality_list

    def pseudo_dataframe(self,):
        """
        获得一个默认的DataFrame
        
        Args:
            None
        Returns:
            out_df:
        """
        return pd.DataFrame([])

    def construct_subqueries(self, meta_list):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        subquery_list = []
        for query_meta in meta_list:
            subquery_list.append(self.generate_query(query_meta))
        return subquery_list

    def construct_subqueries_batch(self, meta_list_batch):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """


    def generate_query(self, query_meta):
        """
        {Description}
        
        Args:
            query_meta:
        Returns:
            res1:
            res2:
        """
        single_df = self.pseudo_dataframe()
        single_wrapper = query_construction.SingleWrapper(\
            single_df, query_meta, mode="Default")
        return single_wrapper.generate_current_query()


    def get_cardinalities(self, query_list):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        return self.db_conn.get_cardinalities(query_list)



# %%
class HardcodeHandler(BaseHandler):
    """
    硬编码的基数，用于直接注入

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
        self.card_dict = {}
        self.abbr_mapping = query_construction.abbr_option[workload]

    def estimate_subqueries(self, meta_list):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        result_list = []
        for meta in meta_list:
            schema_list, filter_list = meta
            schema_sorted = sorted([self.abbr_mapping[s] for s in schema_list])
            result_list.append(self.card_dict[tuple(schema_sorted)])

        print("result_list = {}".format(result_list))
        return result_list

    def set_cardinalities(self, schema_group_list, card_list):
        """
        {Description}

        Args:
            schema_group_list:
            card_list:
        Returns:
            return1:
            return2:
        """
        self.card_dict = {}
        for schema_group, card in zip(schema_group_list, card_list):
            sorted_group = sorted(schema_group) # 排序
            self.card_dict[tuple(sorted_group)] = card

        return self.card_dict

    def set_cardinalities_batch(self, schema_group_list_batch, card_list_batch):
        """
        设置一批的基数，用以处理一批的查询
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        pass

    def estimate_subqueries_batch(self,):
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


# %%
def get_ce_handler_by_name(workload, ce_type):
    if ce_type.lower() == "internal":
        ce_handler = PGInternalHandler(workload=workload)
    elif ce_type.lower() == "deepdb":
        # 默认采用Server模式来实现
        ce_handler = DeepDBHandler(workload=workload, mode="server")
        # self.ce_handler = ce_injection.
    elif ce_type.lower() == "neurocard":
        ce_handler = NeuroCardHandler(workload = workload)
    elif ce_type.lower() == "oracle":
        ce_handler = OracleHandler(workload = workload)
    elif ce_type.lower() == "mscn":
        ce_handler = MSCNHandler(workload = workload)
    elif ce_type.lower() == "fcn":
        ce_handler = FCNHandler(workload = workload)
    elif ce_type.lower() == "factorjoin":
        ce_handler = FactorJoinHandler(workload = workload)
    elif ce_type.lower() == "deepdb_rdc":
        ce_handler = DeepDBRDCHandler(workload = workload)
    elif ce_type.lower() == "deepdb_jct":
        ce_handler = DeepDBJCTHandler(workload = workload)
    elif ce_type.lower() == "sqlserver":
        ce_handler = SQLServerHandler(workload=workload)
    elif ce_type.lower() == "oceanbase":
        ce_handler = OceanbaseHandler(workload=workload)
    elif ce_type.lower() == "fcnpool":
        ce_handler = FCNPoolHandler(workload=workload)
    else:
        raise ValueError("Unsupported ce_type = {}.".format(ce_type))
    
    return ce_handler