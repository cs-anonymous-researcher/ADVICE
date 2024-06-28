#!/usr/bin/env python
# -*- coding: utf-8 -*-


# %%

import numpy as np
import pandas as pd
from os.path import join as p_join
from matplotlib import pyplot as plt
import json, os, time, re, math
import pickle
from plan import join_analysis

# %%
from data_interaction import postgres_connector
from query.query_exploration import get_physical_plan
# %%

def physical_comparison(physical_plan1, physical_plan2):
    """
    比较两个物理计划是否等价

    Args:
        physical_plan1: orders = leading, join_ops, scan_ops
        physical_plan2:
    Returns:
        flag: True说明两个查询计划相等，False说明查询计划相等
        error_dict:
    """
    
    def leading_cmp(leading1, leading2):
        error_dict = {}
        if leading1 != leading2:
            error_dict['leading_diff'] = (leading1, leading2)
            return False, error_dict
        else:
            return True, error_dict

    def join_ops_cmp(join1, join2):
        error_dict = {}
        flag = True
        if set(join1.keys()) != set(join2.keys()):
            error_dict['key_diff'] = (join1.keys(), join2.keys())
            flag = False
        else:
            for k in join1.keys():
                if join1[k] != join2[k]:
                    flag = False
                    error_dict[k] = (join1[k], join2[k])
        return flag, error_dict

    def scan_ops_cmp(scan1, scan2):
        error_dict = {}
        flag = True
        if set(scan1.keys()) != set(scan2.keys()):
            error_dict['key_diff'] = (scan1.keys(), scan2.keys())
            flag = False
        else:
            for k in scan1.keys():
                if scan1[k] != scan2[k]:
                    flag = False
                    error_dict[k] = (scan1[k], scan2[k])
        
        return flag, error_dict
    
    def parse_components(physical_plan):
        if isinstance(physical_plan, (tuple, list)):
            leading, join_ops, scan_ops = physical_plan
        elif isinstance(physical_plan, dict):
            leading, join_ops, scan_ops = \
                physical_plan['leading'], physical_plan['join_ops'], physical_plan['scan_ops']
        elif isinstance(physical_plan, PhysicalPlan):
            leading, join_ops, scan_ops = \
                physical_plan.leading, physical_plan.join_ops, physical_plan.scan_ops
        else:
            raise TypeError("Unsupported physical_plan type: {}".format(type(physical_plan)))

        return leading, join_ops, scan_ops

    flag = True
    # error_dict = {}
    leading1, join_ops1, scan_ops1 = parse_components(physical_plan1)
    leading2, join_ops2, scan_ops2 = parse_components(physical_plan2)

    flag1, error_dict1 = leading_cmp(leading1, leading2)
    flag2, error_dict2 = join_ops_cmp(join_ops1, join_ops2)
    flag3, error_dict3 = scan_ops_cmp(scan_ops1, scan_ops2)
    
    error_dict = {
        "leading": error_dict1,
        "scan_ops": error_dict2,
        "join_ops": error_dict3
    }
    flag = flag1 and flag2 and flag3
    return flag, error_dict
        

# %%


class PhysicalPlan(object):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, query_text, plan_dict, db_conn = None):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        leading, join_ops, scan_ops = get_physical_plan(plan = plan_dict)

        self.leading = leading
        self.join_ops = join_ops
        self.scan_ops = scan_ops
        self.plan_dict = plan_dict  # 保存查询计划
        self.plan_cost = plan_dict["Total Cost"]

        self.db_conn, self.query_text = db_conn, query_text

        # 2024-03-31: 增加join order list信息
    
    def get_join_order_info(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            is_bushy:
            join_order_list:
        """
        jo_analyzer = join_analysis.JoinOrderAnalyzer(\
                join_order_str=self.leading)
        
        is_bushy = jo_analyzer.is_bushy()
        # join_order_list = jo_analyzer.get_leading_tables()
        join_order_list = jo_analyzer.get_leading_order()
        
        return is_bushy, join_order_list

    def get_physical_spec(self,):
        """
        获得物理的具体信息
        
        Args:
            None
        Returns:
            leading:
            scan_ops:
            join_ops:
        """
        return self.leading, self.scan_ops, self.join_ops

    def set_database_connector(self, db_conn: postgres_connector.Connector):
        """
        {Description}

        Args:
            db_conn:
        Returns:
            None
        """
        self.db_conn = db_conn

    def get_specific_hint_query(self, subquery_dict, single_table_dict):
        """
        获得具体的提示查询
        
        Args:
            subquery_dict:
            single_table_dict:
        Returns:
            hint_query:
        """
        # 2024-03-10: 获得特定card_dict下的hint查询
        if self.db_conn is None:
            raise AttributeError("get_specific_plan: self.db_conn is None.")
        
        hint_query = self.db_conn.get_query_under_complete_hint(sql_text = self.query_text, \
            subquery_dict = subquery_dict, single_table_dict = single_table_dict, join_ops = self.join_ops, \
            leading_hint = self.leading, scan_ops = self.scan_ops)

        return hint_query


    def get_specific_plan(self, subquery_dict, single_table_dict):
        """
        在特定基数的条件下获得查询计划
    
        Args:
            subquery_dict:
            single_table_dict:
        Returns:
            plan_dict:
        """
        if self.db_conn is None:
            raise AttributeError("get_specific_plan: self.db_conn is None.")
        
        result = self.db_conn.execute_sql_under_complete_hint(sql_text = self.query_text, \
            subquery_dict = subquery_dict, single_table_dict = single_table_dict, join_ops = self.join_ops, \
            leading_hint = self.leading, scan_ops = self.scan_ops)

        res_plan = result[0][0][0]['Plan']      # 查询计划

        # 自我验证物理算子的apply是否符合期望
        curr_leading, curr_join_ops, curr_scan_ops = get_physical_plan(plan = res_plan)

        flag, error_dict = physical_comparison((self.leading, self.join_ops, self.scan_ops), \
            (curr_leading, curr_join_ops, curr_scan_ops))  # 调用比较物理计划的函数

        if flag == False:
            # 检查hint是否正确的被使用了
            print("query_text = {}".format(self.query_text))
            print("error_dict = {}".format(error_dict))
            print("hint_query = {}".format(self.db_conn.get_query_under_complete_hint(\
                sql_text = self.query_text, subquery_dict = subquery_dict, single_table_dict = single_table_dict, \
                join_ops = self.join_ops, leading_hint = self.leading, scan_ops = self.scan_ops)))
            # raise ValueError("get_specific_plan: hint apply error!")
            print("get_specific_plan: hint apply error!")
            
        return res_plan

    def get_plan_cost(self, subquery_dict, single_table_dict):
        """
        获得具体基数下的查询计划代价

        Args:
            subquery_dict:
            single_table_dict:
        Returns:
            cost:
        """
        plan_dict = self.get_specific_plan(subquery_dict, single_table_dict)
        return plan_dict['Total Cost']
    
    def plan_verification(self, plan_input):
        """
        验证两个plan是否物理计划相同
        
        Args:
            plan_input: 另一个外部的plan_dict
        Returns:
            flag: 两个物理计划是否相等
            error_dict: 不想等的相关信息
        """
        ext_leading, ext_join_ops, ext_scan_ops = get_physical_plan(plan = plan_input)

        # print(ext_leading, ext_join_ops, ext_scan_ops)
        flag, error_dict = physical_comparison((self.leading, self.join_ops, self.scan_ops), \
            (ext_leading, ext_join_ops, ext_scan_ops))  # 调用比较物理计划的函数
        return flag, error_dict
        
    def show_plan(self,):
        """
        展示当前的查询计划
    
        Args:
            None
        Returns:
            scan_ops:
            join_ops:
            leading:
        """
        print("Scan Operations: {}".format(self.scan_ops))
        print("Join Operations: {}".format(self.join_ops))
        print("Join Order: {}".format(self.leading))

        return self.scan_ops, self.join_ops, self.leading

    def show_join_order(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        print("Join Order: {}".format(self.leading))
        return None

    # def plan_comparison(self,):

# %%
