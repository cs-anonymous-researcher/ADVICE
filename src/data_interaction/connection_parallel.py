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
from data_interaction import postgres_connector
import threading
from typing import List
# %% 专门用于并发获得Plan Cost的连接池，优化Benefit Estimation的效率

output_list = []    # 共用的output列表
output_lock = threading.Lock()

def query_func(db_conn: postgres_connector.Connector, query_idx_list: List[tuple]):
    """
    {Description}

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    global output_list
    # local_cache = []
    for query_text, query_idx in query_idx_list:
        try:
            result = db_conn.execute_sql_under_single_process(query_text)
            output_list.append((result, query_idx))
        except Exception as e:
            with output_lock:
                print(f"query_text = {query_text}")
                print(e)
            output_list.append(("123", query_idx))
        
        # local_cache.append((result, query_idx))
    # print(f"connection_parallel.query_func: query_idx = {query_idx}. len(output_list) = {len(output_list)}.")
    # with output_lock:
    #     output_list.extend(local_cache)


class ConnectionPool(object):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, workload, conn_num):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        self.workload = workload
        self.conn_num = conn_num
        self.conn_list = self.create_connections(conn_num)
        self.hint_adder = postgres_connector.HintAdder(alias2table = {})

    def construct_card_hint_sql(self, query_text: str, subquery_dict: dict, single_table_dict: dict):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        head = "EXPLAIN (FORMAT JSON)"
        rows_list = [(k, v) for k, v in subquery_dict.items()]
        schema_list = [(k, v) for k, v in single_table_dict.items()]
        hint_str = self.hint_adder.generate_cardinalities_hint_str(rows_list, schema_list)
        hint_sql_template = "{HINT_STRING}\n" + head + " {SQL_STRING}"
        hint_sql_text = hint_sql_template.format(HINT_STRING = hint_str, SQL_STRING = query_text)

        return hint_sql_text

    def construct_complete_hint_sql(self, sql_text, subquery_dict, single_table_dict,
            join_ops, leading_hint, scan_ops):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        head = "EXPLAIN (FORMAT JSON)"
        def dict2list(in_dict):
            return [(k, v) for k, v in in_dict.items()]
        
        # 从字典到列表
        join_ops = dict2list(join_ops)  
        scan_ops = dict2list(scan_ops)

        config_dict = {
            "subquery_dict": subquery_dict,
            "single_table_dict": single_table_dict,
            "join_ops": join_ops,
            "leading_hint": leading_hint,
            "scan_ops": scan_ops
        }

        hint_str = self.hint_adder.generate_complete_hint_str(config_dict = config_dict)
        hint_sql_template = "{HINT_STRING}\n" + head + " {SQL_STRING}"
        hint_sql_text = hint_sql_template.format(HINT_STRING = hint_str, SQL_STRING = sql_text)
        return hint_sql_text
    

    def create_connections(self, conn_num):
        """
        {Description}

        Args:
            conn_num:
            arg2:
        Returns:
            return1:
            return2:
        """
        conn_list = []

        for _ in range(conn_num):
            conn_list.append(postgres_connector.connector_instance(self.workload))

        return conn_list


    def query_assignment(self, query_input):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        if len(query_input) < self.conn_num:
            query_list_batch = [[] for _ in range(len(query_input))]
        else:
            query_list_batch = [[] for _ in range(self.conn_num)]

        for idx, query_text in enumerate(query_input):
            batch_id = idx % len(query_list_batch)
            query_list_batch[batch_id].append((query_text, idx))

        return query_list_batch
    

    def execute_query_parallel(self, query_list, assign_func = None):
        """
        利用threading模块并发执行数据库查询
        
        Args:
            query_list:
            assign_func:
        Returns:
            result_list:
            res2:
        """
        global output_list  
        # 初始化全局变量
        output_list = []

        # 查询分块
        query_list_batch = self.query_assignment(query_list)
        thread_list = []

        # 启动并发任务
        for idx, query_local in enumerate(query_list_batch):
            local_thread = threading.Thread(target=query_func, args=(self.conn_list[idx], query_local))
            local_thread.start()
            thread_list.append(local_thread)

        # 等待任务结束
        for local_thread in thread_list:
            local_thread.join()

        result_list = [None for _ in range(len(query_list))]
        
        # 用做结果测试
        idx_list = [item[1] for item in output_list]

        for item, idx in output_list:
            # print(f"execute_query_parallel: item = {item}. idx = {idx}. len(result_list) = {len(result_list)}.")
            result_list[idx] = item

        for idx, item in enumerate(result_list):
            if item is None:
                raise ValueError(f"execute_query_parallel: result item is None. idx = {idx}. idx_list = {idx_list}. output_list = {output_list}.")

        # 长度匹配确认
        assert len(result_list) == len(output_list) == len(query_list), \
            f"execute_query_parallel: result_list = {len(result_list)}. output_list = {len(output_list)}. query_list = {len(query_list)}"
        return result_list


    def parse_plan_batch(self, result_list):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        plan_list = [item[0][0][0]["Plan"] for item in result_list]
        return plan_list

    def get_plan_parallel(self, query_list):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        result_list = self.execute_query_parallel(query_list)
        plan_list = self.parse_plan_batch(result_list)
        return plan_list

    def get_plan_under_card_parallel(self, query_list, subquery_dict_list, single_table_dict_list):
        """
        {Description}

        Args:
            query_list:
            subquery_dict_list:
            single_table_dict_list:
        Returns:
            return1:
            return2:
        """
        # 
        hint_sql_list = [self.construct_card_hint_sql(query_text, subquery_dict, single_table_dict)
            for query_text, subquery_dict, single_table_dict in zip(query_list, subquery_dict_list, single_table_dict_list)]

        result_list = self.execute_query_parallel(hint_sql_list)
        plan_list = self.parse_plan_batch(result_list)
        return plan_list
    
    def get_plan_under_complete_parallel(self, query_list, subquery_dict_list, 
            single_table_dict_list, join_ops_list, leading_hint_list, scan_ops_list):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        hint_sql_list = [self.construct_complete_hint_sql(query_text, subquery_dict, single_table_dict)
            for query_text, subquery_dict, single_table_dict in zip(query_list, subquery_dict_list, 
            single_table_dict_list, join_ops_list, leading_hint_list, scan_ops_list)]

        result_list = self.execute_query_parallel(hint_sql_list)
        plan_list = self.parse_plan_batch(result_list)
        return plan_list

# %% 全局连接池变量
conn_pool_dict = {}

def get_conn_pool_by_workload(workload) -> ConnectionPool:
    """
    {Description}

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    if workload not in conn_pool_dict:
        curr_conn_pool = ConnectionPool(workload, conn_num=10)
        conn_pool_dict[workload] = curr_conn_pool
        return curr_conn_pool
    else:
        return conn_pool_dict[workload]


# %%
