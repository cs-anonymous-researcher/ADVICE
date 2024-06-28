#!/usr/bin/env python
# -*- coding: utf-8 -*-


# %%

import numpy as np
import pandas as pd
from os.path import join as p_join
from matplotlib import pyplot as plt
import json, os, time, re, math
import pickle
import pdb

# %%

from data_interaction import postgres_connector
from utility.workload_parser import SQLParser

from query import ce_injection, query_construction
from data_interaction.mv_management import meta_subset
from data_interaction.postgres_connector import connector_instance
from utility import utils


# %% 一些从其他代码借鉴过来的函数
PG_HINT_LEADING_TMP = "Leading({JOIN_ORDER})"

join_types = set(["Nested Loop", "Hash Join", "Merge Join", "Index Scan",\
        "Seq Scan", "Bitmap Heap Scan"])

def extract_values(obj, key):
    """Recursively pull values of specified key from nested JSON."""
    arr = []

    def extract(obj, arr, key):
        """Return all matching values in an object."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    extract(v, arr, key)
                elif k == key:
                    # if "Scan" in v:
                        # print(v)
                        # pdb.set_trace()
                    # if "Join" in v:
                        # print(obj)
                        # pdb.set_trace()
                    arr.append(v)

        elif isinstance(obj, list):
            for item in obj:
                extract(item, arr, key)
        return arr

    results = extract(obj, arr, key)
    return results

def extract_aliases(plan, jg=None):
    if "Alias" in plan:
        # print(plan['Alias'])
        assert plan["Node Type"] == "Bitmap Heap Scan" or "Plans" not in plan
        if jg:
            alias = plan["Alias"]
            real_name = jg.nodes[alias]["real_name"]
            # yield f"{real_name} as {alias}"
            yield "{} as {}".format(real_name, alias)
        else:
            # print("123")
            yield plan["Alias"]

    if "Plans" not in plan:
        return

    for subplan in plan["Plans"]:
        yield from extract_aliases(subplan, jg=jg)


def get_pg_join_order(join_graph, explain):
    physical_join_ops = {}
    scan_ops = {}
    def __update_scan(plan):
        node_types = extract_values(plan, "Node Type")
        alias = extract_values(plan, "Alias")[0]
        for nt in node_types:
            if "Scan" in nt:
                scan_type = nt
                break
        scan_ops[alias] = nt

    def __extract_jo(plan):
        if plan["Node Type"] in join_types:
            left = list(extract_aliases(plan["Plans"][0], jg=join_graph))
            right = list(extract_aliases(plan["Plans"][1], jg=join_graph))
            all_froms = left + right
            all_nodes = []
            for from_clause in all_froms:
                # print("from_clause = {}".format(from_clause))
                # from_alias = from_clause[from_clause.find(" as ")+4:]
                from_alias = from_clause
                if "_info" in from_alias:
                    print(from_alias)
                    pdb.set_trace()
                all_nodes.append(from_alias)
            all_nodes.sort()
            all_nodes = " ".join(all_nodes)
            physical_join_ops[all_nodes] = plan["Node Type"]

            if len(left) == 1 and len(right) == 1:
                __update_scan(plan["Plans"][0])
                __update_scan(plan["Plans"][1])
                return left[0] +  " CROSS JOIN " + right[0]

            if len(left) == 1:
                __update_scan(plan["Plans"][0])
                return left[0] + " CROSS JOIN (" + __extract_jo(plan["Plans"][1]) + ")"

            if len(right) == 1:
                __update_scan(plan["Plans"][1])
                return "(" + __extract_jo(plan["Plans"][0]) + ") CROSS JOIN " + right[0]

            return ("(" + __extract_jo(plan["Plans"][0])
                    + ") CROSS JOIN ("
                    + __extract_jo(plan["Plans"][1]) + ")")

        return __extract_jo(plan["Plans"][0])

    try:
        # return __extract_jo(explain[0][0][0]["Plan"]), physical_join_ops, scan_ops
        return __extract_jo(explain), physical_join_ops, scan_ops
    except:
        print(explain)
        pdb.set_trace()

# %%

def get_plan_component_cost(plan_dict, kw = 'Total Cost'):
    """
    获得查询每一部分的代价
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    join_graph, explain = None, plan_dict
    physical_join_cost = {}
    scan_cost = {}
    
    def __update_scan(plan):
        # node_types = extract_values(plan, "Node Type")
        alias = extract_values(plan, "Alias")[0]
        scan_cost[alias] = plan[kw]

    def __extract_jo(plan):
        if plan["Node Type"] in join_types:
            left = list(extract_aliases(plan["Plans"][0], jg=join_graph))
            right = list(extract_aliases(plan["Plans"][1], jg=join_graph))
            all_froms = left + right
            all_nodes = []
            for from_clause in all_froms:
                from_alias = from_clause
                if "_info" in from_alias:
                    print(from_alias)
                    pdb.set_trace()
                all_nodes.append(from_alias)
            all_nodes.sort()
            all_nodes = " ".join(all_nodes)

            physical_join_cost[all_nodes] = plan[kw]

            if len(left) == 1 and len(right) == 1:
                __update_scan(plan["Plans"][0])
                __update_scan(plan["Plans"][1])
                return left[0] +  " CROSS JOIN " + right[0]

            if len(left) == 1:
                __update_scan(plan["Plans"][0])
                return left[0] + " CROSS JOIN (" + __extract_jo(plan["Plans"][1]) + ")"

            if len(right) == 1:
                __update_scan(plan["Plans"][1])
                return "(" + __extract_jo(plan["Plans"][0]) + ") CROSS JOIN " + right[0]

            return ("(" + __extract_jo(plan["Plans"][0])
                    + ") CROSS JOIN ("
                    + __extract_jo(plan["Plans"][1]) + ")")

        return __extract_jo(plan["Plans"][0])

    try:
        # return __extract_jo(explain[0][0][0]["Plan"]), physical_join_ops, scan_ops
        return __extract_jo(explain), physical_join_cost, scan_cost
    except:
        print(explain)
        pdb.set_trace()

# %%



def get_leading_hint(join_graph, explain):
    '''
    Ryan's implementation.
    '''
    def __extract_jo(plan):
        if plan["Node Type"] in join_types:
            left = list(extract_aliases(plan["Plans"][0], jg=join_graph))
            right = list(extract_aliases(plan["Plans"][1], jg=join_graph))
            # print("left = {}. right = {}.".format(left, right))

            if len(left) == 1 and len(right) == 1:
                # left_alias = left[0][left[0].lower().find(" as ")+4:]
                # right_alias = right[0][right[0].lower().find(" as ")+4:]
                left_alias = left[0]
                right_alias = right[0]
                return left_alias +  " " + right_alias

            if len(left) == 1:
                # left_alias = left[0][left[0].lower().find(" as ")+4:]
                left_alias = left[0]
                return left_alias + " (" + __extract_jo(plan["Plans"][1]) + ")"

            if len(right) == 1:
                # right_alias = right[0][right[0].lower().find(" as ")+4:]
                right_alias = right[0]
                return "(" + __extract_jo(plan["Plans"][0]) + ") " + right_alias

            return ("(" + __extract_jo(plan["Plans"][0])
                    + ") ("
                    + __extract_jo(plan["Plans"][1]) + ")")

        return __extract_jo(plan["Plans"][0])

    jo = __extract_jo(explain)
    jo = "(" + jo + ")"
    return PG_HINT_LEADING_TMP.format(JOIN_ORDER = jo)


# %%

def get_physical_plan(plan):
    """
    获得plan的所有物理信息
    
    Args:
        plan: 查询计划字典
    Returns:
        leading: 连接顺序
        join_ops: 连接物理运算符
        scan_ops: 单表扫描运算符
    """
    join_graph = None
    leading = get_leading_hint(join_graph, explain = plan)
    join_component, join_ops, scan_ops = \
        get_pg_join_order(join_graph, explain = plan)

    return leading, join_ops, scan_ops

# %%

def clone_plan_info(dbc, plan):
    """
    拷贝复制查询计划的信息
    
    Args:
        arg1:
        arg2:
    Returns:
        leading:
        join_ops:
        scan_ops:
        card_set:
    """
    join_graph = None
    leading = get_leading_hint(join_graph, explain = plan)
    join_component, join_ops, scan_ops = \
        get_pg_join_order(join_graph, explain = plan)

    card_set = dbc.parse_all_subquery_cardinality(plan)    # Plan执行路径上涉及到的基数
    return leading, join_ops, scan_ops, card_set


# %%

from collections import namedtuple

class Plan(namedtuple("Plan", ["leading", "scan_ops", "join_ops", "card_set"])):
    __slots__ = ()
    def __str__(self):
        str_list = []
        str_list.append("join leading: {}".format(self.leading))
        str_list.append("scan operators: {}".format(self.scan_ops))
        str_list.append("join operators: {}".format(self.join_ops))
        str_list.append("involving cardinalities: {}".format(self.card_set))
        return "\n".join(str_list)



def plan_comparison(plan1: Plan, plan2: Plan):
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    if plan1.leading != plan2.leading:
        print("Join Order Diff")
        return False
    else:
        print("Join Order Same")
        join_ops_diff = {}
        scan_ops_diff = {}
        join_ops1, join_ops2 = plan1.join_ops, plan2.join_ops
        scan_ops1, scan_ops2 = plan1.scan_ops, plan2.scan_ops

        print("join_ops1 = {}. join_ops2 = {}".format(join_ops1, join_ops2))

        for k in join_ops1.keys():
            if join_ops1[k] != join_ops2[k]:
                join_ops_diff[k] = (join_ops1[k], join_ops2[k])

        for k in scan_ops1.keys():
            if scan_ops1[k] != scan_ops2[k]:
                scan_ops_diff[k] = (scan_ops1[k], scan_ops2[k])

        if len(join_ops_diff) != 0:
            print("join_ops diff: {}".format(join_ops_diff))
            return False
        
        if len(scan_ops_diff) != 0:
            print("scan_ops diff: {}".format(scan_ops_diff))
            return False

    print("Two Plans are identical")
    return True


# subquery基数提取函数(其他地方也会用到，单独拿出来)

    
def read_file(file_path = ""):
    """
    {Description}
    
    Args:
        file_path:
    Returns:
        res1:
        res2:
    """
    with open(file_path, "r") as f_in:
        return f_in.read().splitlines()

def diff_file(line_list1, line_list2):
    """
    {Description}
    
    Args:
        file_path:
    Returns:
        res1:
        res2:
    """
    return line_list2[len(line_list1):]

def eval_subqueries(subquery_list, parser):
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    subquery_set = set()
    # join_info_tmpl = "RELOPTINFO (pb pt pa): rows=(\d+) width=(\d+)"
    join_info_tmpl = "RELOPTINFO \(([\w\s]+)\): rows=(\d+) width=(\d+)"

    card_mapping = {}
    prog = re.compile(join_info_tmpl)
    for sub_info in subquery_list:
        # print("sub_info = {}".format(sub_info))
        res = prog.match(sub_info)
        if res is not None:
            alias_list = res.group(1).split(" ")
            alias_tuple = tuple(sorted(alias_list))
            card_mapping[alias_tuple] = int(res.group(2))
            subquery_set.add(alias_tuple)
        else:
            print("res is None")

    subquery_repr_list = list(subquery_set)
    subquery_sql_list = [parser.construct_sub_queries(alias_list)\
        for alias_list in subquery_repr_list]
    subquery_card_list = [card_mapping[table_set] for table_set in subquery_repr_list] # 

    return subquery_sql_list, subquery_repr_list, subquery_card_list

def get_subqueries(sql_text, file_path, parser, db_conn):
    """
    获得子查询，单独的函数

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    previous_file_list = read_file(file_path = file_path)
    db_conn.eval_subqueries(sql_text)
    current_file_list = read_file(file_path = file_path)

    delta_list = diff_file(previous_file_list, current_file_list)
    subquery_sql_list, subquery_repr_list, subquery_card_list = \
        eval_subqueries(delta_list, parser)
    # cardinality_list = self.get_true_cardinalities(subquery_sql_list)
    return subquery_sql_list, subquery_repr_list, subquery_card_list


# %%

from collections import namedtuple

"""
查询计划包含的元素

连接顺序
物理算子（包括扫描算子和连接算子）
"""


class QueryEvaluator(object):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, dbc:postgres_connector.Connector, \
        file_path = "", handler = None):
        """
        构造函数

        Args:
            arg1:
            arg2:
        """
        self.dbc = dbc
        self.file_path = file_path
        if handler is not None:
            self.handler = handler
        else:
            raise ValueError("QueryEvaluator: handler is None")
    
    def get_plan(self, sql_text):
        """
        获取查询的执行计划
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        return self.dbc.get_plan(sql_text)

    def output_result(self,):
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
    
    def process_single_query(self, sql_text):
        """
        端到端的处理单条查询
        
        1. 获得子查询
        2. 获得真实基数
        3. 获得估计器的基数
        4. 比较结果

        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """

        self.parser = SQLParser(sql_text)       # SQL解析处理
        # 获得子查询
        sql_list, repr_list, estimated_card_list = self.get_subqueries(sql_text)

        # 获得真实基数以及真实基数下查询计划相关信息
        cardinality_list = self.get_true_cardinalities(sql_list)    # 真实基数的列表
        true_card_plan = self.true_cardinality_plan(sql_text, cardinality_list, repr_list)
        true_leading, true_join_ops, true_scan_ops, true_card_set = self.clone_plan_info(true_card_plan)
        # print("true_card_plan = {}".format(true_card_plan))

        # 获得基数估计器计划的信息
        ce_card_list = self.handler.process_subqueries(repr_list, cardinality_list)                                        # 获得基数估计器给出的结果
        ce_plan = self.ce_cardinality_plan(sql_text, ce_card_list, repr_list)
        ce_leading, ce_join_ops, ce_scan_ops, ce_card_set = self.clone_plan_info(ce_plan)

        true_cardinalities = list(zip(repr_list, cardinality_list))
        ce_cost_under_true = self.plan_cost_estimation(sql_text, cardinalities = true_cardinalities,\
            join_ops = ce_join_ops, scan_ops = ce_scan_ops, leading = ce_leading)
        true_cost_under_true = self.plan_cost_estimation(sql_text, cardinalities = true_cardinalities, \
            join_ops = true_join_ops, scan_ops = true_scan_ops, leading = true_leading)

        result_dict = {
            "ce_cost_under_true": ce_cost_under_true,
            "true_cost_under_true": true_cost_under_true,
            "ce_leading": ce_leading,
            "ce_join_ops": ce_join_ops,
            "ce_scan_ops": ce_scan_ops,
            "true_leading": true_leading,
            "true_join_ops": true_join_ops,
            "true_scan_ops": true_scan_ops,
            "true_cardinalities": list(zip(repr_list, cardinality_list)),
            "ce_cardinalities": list(zip(repr_list, ce_card_list))
        }
        # return sql_list, repr_list, cardinality_list
        return result_dict

    def plan_variation_under_injection(self, sql_text):
        """
        注入基数后查询计划的变化
        
        Args:
            sql_text:
        Returns:
            result_dict:
        """
        self.parser = SQLParser(sql_text)       # SQL解析处理

        # 获得默认计划的信息
        ce_plan_dict = self.get_plan(sql_text) 
        ce_leading, ce_join_ops, ce_scan_ops, ce_card_set = self.clone_plan_info(ce_plan_dict)
        ce_plan = Plan(leading = ce_leading, \
            scan_ops = ce_scan_ops, join_ops = ce_join_ops, card_set = ce_card_set)

        sql_list, repr_list, estimated_card_list = self.get_subqueries(sql_text)

        print("sql_list:")
        print("\n".join(["{}. {}".format(idx, content) for idx, content in enumerate(sql_list)]))
        print("repr_list:")
        print("\n".join(["{}. {}".format(idx, content) for idx, content in enumerate(repr_list)]))

        cardinality_list = self.get_true_cardinalities(sql_list)    # 真实基数的列表
        true_card_plan_dict = self.true_cardinality_plan(sql_text, cardinality_list, repr_list)

        # print("true_card_plan = {}".format(true_card_plan_dict))
        # 真实基数下查询计划相关信息
        true_leading, true_join_ops, true_scan_ops, true_card_set = self.clone_plan_info(true_card_plan_dict)
        true_card_plan = Plan(leading = true_leading, \
            scan_ops = true_scan_ops, join_ops = true_join_ops, card_set = true_card_set)

        result_dict = {
            "ce_plan": ce_plan,
            "true_plan": true_card_plan,
            "ce_cardinalities": list(zip(repr_list, estimated_card_list)),
            "true_cardinalities": list(zip(repr_list, cardinality_list)), 
        }
        return result_dict



    def plan_cost_estimation(self, sql_text, cardinalities, \
            join_ops, scan_ops, leading):
        """
        获得固定查询计划下的cost
        
        Args:
            sql_text:
            cardinalities:
            join_ops:
            scan_ops:
            leading:
        Returns:
            res1:
            res2:
        """
        def dict2list(in_dict):
            out_list = []
            for k, v in in_dict.items():
                out_list.append((k, v))
            return out_list
        
        # 从字典到列表
        join_ops = dict2list(join_ops)  
        scan_ops = dict2list(scan_ops)

        print("join_ops = {}".format(join_ops))
        print("scan_ops = {}".format(scan_ops))

        res_list = self.dbc.pghint_modified_sql(sql_text = sql_text, \
            cardinalities=cardinalities, join_ops=join_ops, scan_ops=scan_ops,\
            leading_hint = leading)

        return res_list[0][0][0]['Plan']['Total Cost']

    def true_cardinality_plan(self, sql_text, cardinality_list, subquery_repr_list):
        """
        获得真实基数对应的查询计划
        
        Args:
            sql_text:
            cardinality_list:
            subquery_repr_list:
        Returns:
            plan_dict:
        """
        cardinalities = zip(subquery_repr_list, cardinality_list) # 构造真实基数元组
        plan = self.dbc.pghint_modified_sql(sql_text = sql_text, \
            cardinalities = cardinalities)
        # print("plan = {}".format(plan))
        return plan[0][0][0]['Plan']

    def ce_cardinality_plan(self, sql_text, cardinality_list, subquery_repr_list):
        """
        注入估计器基数下的查询计划
        
        Args:
            sql_text:
            cardinality_list:
            subquery_repr_list:
        Returns:
            plan_dict:
        """
        return self.true_cardinality_plan(sql_text, cardinality_list, subquery_repr_list)

    def get_subqueries(self, sql_text):
        """
        获得子查询

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        previous_file_list = self.read_file(file_path=self.file_path)
        self.dbc.eval_subqueries(sql_text)
        current_file_list = self.read_file(file_path = self.file_path)

        # print("len(previous) = {}. len(current) = {}.".\
        #     format(len(previous_file_list), len(current_file_list)))

        delta_list = self.diff_file(previous_file_list, current_file_list)
        subquery_sql_list, subquery_repr_list, subquery_card_list = \
            self.eval_subqueries(delta_list)
        # cardinality_list = self.get_true_cardinalities(subquery_sql_list)
        return subquery_sql_list, subquery_repr_list, subquery_card_list

    def get_true_cardinalities(self, sql_list):
        """
        获得真实的基数

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        result_list = self.dbc.get_cardinalities(sql_list)
        return result_list
    
    def read_file(self, file_path = ""):
        """
        {Description}
        
        Args:
            file_path:
        Returns:
            res1:
            res2:
        """
        with open(file_path, "r") as f_in:
            return f_in.read().splitlines()
    
    def clear_file(self, file_path = ""):
        """
        清理文件
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        with open(file_path, "w+") as f_out:
            f_out.write("")


    def clone_plan_info(self, plan):
        return clone_plan_info(self.dbc, plan)

    def diff_file(self, line_list1, line_list2):
        """
        {Description}
        
        Args:
            file_path:
        Returns:
            res1:
            res2:
        """
        return line_list2[len(line_list1):]

    def eval_subqueries(self, subquery_list):
        """
        处理所有的子查询
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        subquery_set = set()
        join_info_tmpl = "RELOPTINFO \(([\w\s]+)\): rows=(\d+) width=(\d+)"

        card_mapping = {}
        prog = re.compile(join_info_tmpl)
        for sub_info in subquery_list:
            # print("sub_info = {}".format(sub_info))
            res = prog.match(sub_info)
            if res is not None:
                alias_list = res.group(1).split(" ")
                alias_tuple = tuple(sorted(alias_list))
                card_mapping[alias_tuple] = int(res.group(2))
                subquery_set.add(alias_tuple)
            else:
                print("res is None")

        subquery_repr_list = list(subquery_set)
        subquery_sql_list = [self.parser.construct_sub_queries(alias_list)\
            for alias_list in subquery_repr_list]
        subquery_card_list = [card_mapping[table_set] for table_set in subquery_repr_list] # 

        return subquery_sql_list, subquery_repr_list, subquery_card_list

# %%
def make_query_controller_ref():
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
            ctrl_dict[workload] = QueryController(workload = workload)
            return ctrl_dict[workload]
    return local_func

get_query_controller = make_query_controller_ref()

# %%
import psycopg2 as pg

class QueryController(object):
    """
    查询的控制类，通过注入不同的基数估计，由此得到不一样的结果

    Members:
        field1:
        field2:
    """

    def __init__(self, db_conn:postgres_connector.Connector = None, \
        file_path = "/home/lianyuan/Database/PostgreSQL/File/basic.txt", workload = "job"):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        if db_conn is None:
            self.db_conn = connector_instance(workload)
        else:
            self.db_conn = db_conn
        self.file_path = file_path
        self.workload = workload
        self.clear_file(file_path=file_path)        # 清理文件

        self.query_text, self.query_meta = "", ((), ())

    def set_query_instance(self, query_text, query_meta):
        """
        设置当前处理的查询实例
        
        Args:
            query_text:
            query_meta:
        Returns:
            res1:
            res2:
        """
        self.query_text = query_text
        self.query_meta = query_meta

    def set_query_batch(self, query_text_list, query_meta_list):
        """
        批量的设置查询
        
        Args:
            query_text_list:
            query_meta_list:
        Returns:
            res1:
            res2:
        """
        self.query_text_list = query_text_list
        self.query_meta_list = query_meta_list
    

    def get_subquery_estimation_batch(self, ce_handler):
        """
        {Description}
        
        Args:
            ce_handler:
        Returns:
            subquery_repr_list_batch:
            subquery_est_list_batch:
        """
        query_text_list, query_meta_list = self.query_text_list, self.query_meta_list
        subquery_meta_list_batch, subquery_repr_list_batch, subquery_card_list_batch = [], [], []

        for query_text in query_text_list:
            subquery_meta_list, subquery_repr_list, subquery_card_list \
                = self.get_subqueries(query_text)
            subquery_meta_list_batch.append(subquery_meta_list)
            subquery_repr_list_batch.append(subquery_repr_list)
            subquery_card_list_batch.append(subquery_card_list)

        subquery_est_list_batch = ce_handler.estimate_subqueries_batch(subquery_meta_list_batch)

        return subquery_repr_list_batch, subquery_est_list_batch


    def get_subquery_estimation(self, ce_handler):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        query_text, query_meta = self.query_text, self.query_meta
        subquery_meta_list, subquery_repr_list, subquery_card_list \
            = self.get_subqueries(query_text)

        subquery_est_list = ce_handler.estimate_subqueries(subquery_meta_list)
        return subquery_repr_list, subquery_est_list


    def get_plan_by_external_info(self, subquery_dict = None, single_table_dict = None, \
        join_leading = None, scan_ops = None, join_ops = None):
        """
        根据外部信息获得指定的查询计划
        
        Args:
            subquery_dict:
            single_table_dict:
            join_leading:
            scan_ops:
            join_ops:
        Returns:
            plan_dict:
        """
        plan_res = self.db_conn.execute_sql_under_complete_hint(sql_text = self.query_text, subquery_dict = subquery_dict, \
            single_table_dict = single_table_dict, join_ops = join_ops, leading_hint = join_leading, scan_ops = scan_ops)[0][0][0]['Plan']
        return plan_res

    def get_plan_by_external_card(self, subquery_dict, single_table_dict):
        """
        通过外部的基数
        
        Args:
            subquery_dict:
            single_table_dict:
        Returns:
            plan_dict:
            res2:
        """
        hint_query_text = self.db_conn.inject_cardinalities_sql(sql_text = self.query_text, \
            subquery_dict = subquery_dict, single_table_dict = single_table_dict)

        plan_dict = self.db_conn.execute_sql_under_single_process(\
            hint_query_text)[0][0][0]['Plan']
        
        return plan_dict


    def get_all_single_relations(self,):
        """
        获得所有单表查询的关系
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        single_table_repr_list = []
        alias_mapping = query_construction.abbr_option[self.workload]

        for schema in self.query_meta[0]:
            single_table_repr_list.append(alias_mapping[schema])

        return single_table_repr_list
        
    # @utils.timing_decorator
    def get_all_sub_relations(self, ):
        """
        获得所有多表子查询的关系
        
        Args:
            None
        Returns:
            subquery_repr_list:
        """
        query_text = self.query_text
        _, subquery_repr_list, _ \
            = self.get_subqueries(query_text)
        return subquery_repr_list
        

    def get_specific_plan(self, ce_handler):
        """
        根据基数估计器的结果获得具体的查询
    
        Args:
            ce_handler:
        Returns:
            res_dict:
        """
        query_text, query_meta = self.query_text, self.query_meta
        subquery_meta_list, subquery_repr_list, subquery_card_list \
            = self.get_subqueries(query_text)

        subquery_est_list = ce_handler.estimate_subqueries(subquery_meta_list)
        ce_card_plan = self.ce_cardinality_plan(\
            query_text, subquery_est_list, subquery_repr_list)

        ce_leading, ce_join_ops, ce_scan_ops, ce_card_set = \
            self.clone_plan_info(ce_card_plan)

        res_dict = {
            "ce_leading": ce_leading,
            "ce_join_ops": ce_join_ops,
            "ce_scan_ops": ce_scan_ops,
        }

        return res_dict


    def get_specific_plan_batch(self, ce_handler):
        """
        根据基数估计器的结果获得具体的查询
    
        Args:
            ce_handler:
        Returns:
            res_dict:
        """
        query_text_list = self.query_text_list
        subquery_repr_list_batch, subquery_est_list_batch = \
            self.get_subquery_estimation_batch(ce_handler)
        res_dict_list = []

        for query_text, subquery_repr_list, subquery_est_list in \
            zip(query_text_list, subquery_repr_list_batch, subquery_est_list_batch):
            ce_card_plan = self.ce_cardinality_plan(\
                query_text, subquery_est_list, subquery_repr_list)

            ce_leading, ce_join_ops, ce_scan_ops, ce_card_set = \
                self.clone_plan_info(ce_card_plan)

            res_dict = {
                "ce_leading": ce_leading,
                "ce_join_ops": ce_join_ops,
                "ce_scan_ops": ce_scan_ops,
            }
            res_dict_list.append(res_dict)

        return res_dict_list


    def clone_plan_info(self, plan_dict):
        """
        {Description}
        
        Args:
            plan_dict:
        Returns:
            plan_info:
        """
        return clone_plan_info(self.db_conn, plan_dict)

    def get_specific_multi_plans(self, ce_handler_list):
        """
        根据一系列基数估计器的结果获得一批查询计划
        
        Args:
            ce_handler_list:
        Returns:
            plan_list:
        """
        

    def ce_cardinality_plan(self, sql_text, cardinality_list, subquery_repr_list):
        """
        注入估计器基数下的查询计划
        
        Args:
            sql_text:
            cardinality_list:
            subquery_repr_list:
        Returns:
            plan_dict:
        """
        cardinalities = list(zip(subquery_repr_list, cardinality_list)) # 构造真实基数元组
        print("cardinalities = {}.".format(cardinalities))
        plan = self.db_conn.pghint_modified_sql(sql_text = sql_text, \
            cardinalities = cardinalities)
        # print("plan = {}".format(plan))
        return plan[0][0][0]['Plan']

    def get_subqueries(self, sql_text):
        """
        {Description}
        
        Args:
            sql_text:
        Returns:
            subquery_meta_list:
            subquery_repr_list:
            subquery_card_list:
        """
        previous_file_list = self.read_file(file_path=self.file_path)
        try:
            self.db_conn.eval_subqueries(sql_text)
        except pg.errors.SyntaxError as e:
            print(f"get_subqueries: meet SyntaxError. sql_text = {sql_text}")
            raise e
        
        current_file_list = self.read_file(file_path = self.file_path)

        delta_list = self.diff_file(previous_file_list, current_file_list)
        subquery_meta_list, subquery_repr_list, subquery_card_list = \
            self.eval_subqueries(delta_list)
        return subquery_meta_list, subquery_repr_list, subquery_card_list

    def construct_subquery_meta(self, subquery_repr_list):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        subquery_meta_list = []
        for schema_subset in subquery_repr_list:
            # print("schema_subset = {}".format(schema_subset))
            subquery_meta = meta_subset(self.query_meta, schema_subset, \
                abbr_mapping=self.workload)
            # print("schema_subset = {}. subquery_meta = {}".\
            #     format(schema_subset, subquery_meta))
            subquery_meta_list.append(subquery_meta)

        return subquery_meta_list

    def eval_subqueries(self, subquery_list):
        """
        处理所有的子查询
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        subquery_set = set()
        join_info_tmpl = "RELOPTINFO \(([\w\s]+)\): rows=(\d+) width=(\d+)"

        card_mapping = {}
        prog = re.compile(join_info_tmpl)
        for sub_info in subquery_list:
            # print("sub_info = {}".format(sub_info))
            res = prog.match(sub_info)
            if res is not None:
                alias_list = res.group(1).split(" ")
                alias_tuple = tuple(sorted(alias_list))
                card_mapping[alias_tuple] = int(res.group(2))
                subquery_set.add(alias_tuple)
            else:
                print("res is None")

        subquery_repr_list = list(subquery_set)
        subquery_meta_list = self.construct_subquery_meta(subquery_repr_list)               # 查询元信息
        subquery_card_list = [card_mapping[table_set] for table_set in subquery_repr_list]  # 

        return subquery_meta_list, subquery_repr_list, subquery_card_list


    def read_file(self, file_path = ""):
        """
        {Description}
        
        Args:
            file_path:
        Returns:
            result:
        """
        with open(file_path, "r") as f_in:
            return f_in.read().splitlines()

    def clear_file(self, file_path = ""):
        """
        清理文件
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """

        if os.path.isfile(file_path) == True:
            # TODO: 将文件的历史内容打包

            # 清理当前文件
            with open(file_path, "w+") as f_out:
                f_out.write("")
        else:
            Warning("clear_file: path does not exist!")

    def diff_file(self, line_list1, line_list2):
        """
        {Description}
        
        Args:
            line_list1:
            line_list2:
        Returns:
            sub_list:
        """
        return line_list2[len(line_list1):]


    