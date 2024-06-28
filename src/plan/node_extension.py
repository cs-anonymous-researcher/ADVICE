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
from query import query_exploration
from utility.workload_parser import SQLParser
from workload import physical_plan_info     # 物理计划的相关信息
from data_interaction import data_management
from grid_manipulation.grid_analysis import split_segment   # 
from utility import workload_spec
from utility.utils import tuple_in

from plan.node_query import QueryInstance
from plan.node_utils import complement_true_card, complement_true_card_with_hint, \
    complement_estimation_card, plan_evaluation_under_cardinality, plan_evaluation_under_cardinality_parallel
from plan import node_utils, join_analysis

# %% 新增的import

from asynchronous import construct_input, process_output
import multiprocessing
from utility import utils
from typing import List
from data_interaction import connection_parallel

# %%

class ExtensionInstance(object):
    """
    在QueryInstance上添加一个新的表以及Column条件，针对真实基数进行分析，
    判断是否可以在估计基数和真实基数下生成两个Plan，并且Cost之间的差足够大

    Members:
        field1:
        field2:
    """

    def __init__(self, query_text:str, query_meta:tuple, query_ctrl:query_exploration.QueryController, \
        subquery_estimation: dict, single_table_estimation: dict, subquery_true: dict, \
        single_table_true: dict, external_info:dict):
        """
        {Description}

        Args:
            query_text:
            query_meta:
            query_ctrl:
            subquery_estimation:
            single_table_estimation:
            subquery_true:
            single_table_true:
            external_info:
        """
        self.workload = query_ctrl.workload
        self.query_text = query_text
        # 创建SQL的解析器，附带workload参数
        self.query_parser = SQLParser(sql_text = query_text, workload=self.workload) 
        self.query_meta = query_meta

        # self.query_ctrl = deepcopy(query_ctrl)  # 复制一个本地的QueryController，存在的问题是代价较大
        # 生成新的实例，但是仍然重用db_conn
        self.query_ctrl = query_exploration.QueryController(query_ctrl.\
            db_conn, query_ctrl.file_path, query_ctrl.workload)

        self.set_query_instance()

        self.subquery_estimation = subquery_estimation
        self.single_table_estimation = single_table_estimation
        self.subquery_true = subquery_true
        self.single_table_true = single_table_true
        self.external_info = external_info

        # print(f"ExtensionInstance.__init__: estimaiton_subquery = {utils.dict_str(subquery_estimation)}.")
        # print(f"ExtensionInstance.__init__: true_subquery = {utils.dict_str(subquery_true)}.")
        # print(f"ExtensionInstance.__init__: estimaiton_single_table = {utils.dict_str(single_table_estimation)}.")
        # print(f"ExtensionInstance.__init__: true_single_table = {utils.dict_str(single_table_true)}.")

        self.get_cardinalities = self.query_ctrl.db_conn.get_cardinalities
        self.get_cardinalities_with_hint = self.query_ctrl.db_conn.get_cardinalities_with_hint

        self.subquery_diff, self.single_table_diff = {}, {}
        
        # 自动验证输入基数的合法性
        flag, needless_items, missing_items = self.cardinality_items_validation()
        if flag == False:
            print("ExtensionInstance: Warning! needless_items = {}. missing_items = {}".\
                format(needless_items, missing_items))
            
        # 保存真实基数验证的实例
        self.verification_instance = {}

    def invalid_evaluation(self, card_dict):
        """
        判断是否会出现None的结果(主要是由超时的Query衍生而来)
    
        Args:
            card_dict:
            arg2:
        Returns:
            flag:
            return2:
        """
        flag = True
        for k, v in card_dict.items():
            if v is None:
                return False
        return flag


    def set_query_instance(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        self.query_ctrl.set_query_instance(query_text=self.\
            query_text, query_meta=self.query_meta)

    def cardinality_items_validation(self,):
        """
        验证基数项的正确性，即subquery_estimation和single_table_estimation
        包含且仅包含构造hint_sql所需的项
    
        Args:
            arg1:
            arg2:
        Returns:
            flag: 结果是否正确
            needless_items: 多余的数据项
            missing_items: 缺少的数据项
        """
        needless_items, missing_items = set(), set()
        flag = True

        all_single_table_items = self.query_ctrl.get_all_single_relations()
        all_subquery_items = self.query_ctrl.get_all_sub_relations()

        subquery_estimation_keys = self.subquery_estimation.keys()
        single_table_estimation_keys = self.single_table_estimation.keys()

        subquery_union = set(all_subquery_items).union(subquery_estimation_keys)
        single_table_union = set(all_single_table_items).union(single_table_estimation_keys)
        
        for item in subquery_union:
            flag1 = item in all_subquery_items
            flag2 = item in subquery_estimation_keys
            if flag1 and flag2:
                pass
            elif flag1 and (not flag2):
                missing_items.add(item)
                flag = False
            elif (not flag1) and flag2:
                needless_items.add(item)
                flag = False
            else:
                raise ValueError("cardinality_items_validation: subquery_union = {}. \
                    all_subquery_items = {}. subquery_estimation_keys = {}".\
                    format(subquery_union, all_subquery_items, subquery_estimation_keys))

        for item in single_table_union:
            flag1 = item in all_single_table_items
            flag2 = item in single_table_estimation_keys
            if flag1 and flag2:
                pass
            elif flag1 and (not flag2):
                missing_items.add(item)
                flag = False
            elif (not flag1) and flag2:
                needless_items.add(item)
                flag = False
            else:
                raise ValueError("cardinality_items_validation: flag1 = {}. flag2 = {}.".\
                                 format(flag1, flag2))


        return flag, needless_items, missing_items


    def infer_missing_cardinalities(self,):
        """
        {Description}
    
        Args:
            None
        Returns:
            return1:
            return2:
        """
        pass


    def load_external_missing_card(self, subquery_dict, single_table_dict):
        """
        {Description}
        
        Args:
            subquery_dict:
            single_table_dict:
        Returns:
            plan1:
            plan2:
            cost1:
            cost2:
        """
        subquery_full = node_utils.dict_complement(self.subquery_true, subquery_dict)
        single_table_full = node_utils.dict_complement(self.single_table_true, single_table_dict)

        flag, cost1, cost2 = self.two_plan_verification(subquery_dict1=subquery_full, \
            single_table_dict1=single_table_full, subquery_dict2=self.subquery_estimation, \
            single_table_dict2=self.single_table_estimation, keyword1="mixed", keyword2="estimation")
        
        return flag, cost1, cost2
        

    def get_external_missing_card(self,):
        """
        {Description}
        
        Args:
            None
        Returns:
            subquery_dict:
            single_table_dict:
        """
        subquery_out, single_table_out = node_utils.parse_missing_card(\
            query_parser=self.query_parser, subquery_ref=self.subquery_estimation, single_table_ref=\
            self.single_table_estimation, subquery_missing=self.subquery_true, \
            single_table_missing=self.single_table_true)

        return subquery_out, single_table_out

    def complement_missing_mixed_card(self, option_dict = {}):
        """
        补全缺失的真实基数，生成混合基数，考虑采用多种策略完成
        1. 最近邻等比例补全
        2. 基于数学模型的补全

        Args:
            option_dict: 选项字典
        Returns:
            subquery_mixed:
            single_table_mixed:
        """

        subquery_mixed = deepcopy(self.subquery_true)           # 已有的真实基数部分，直接复制过来
        single_table_mixed = deepcopy(self.single_table_true)   # 

        def find_nearest_subquery(alias_in):
            """
            找到距离最近的子查询
            
            Args:
                alias_in:
            Returns:
                alias_out:
            """
            alias_ref = ()
            curr_alias, curr_delta = None, 1000
            print("self.subquery_true.keys = {}.".format(self.subquery_true.keys()))

            for k in self.subquery_true.keys():
                # print("alias_in = {}. k = {}.".format(alias_in, k))
                # 判断tuple之间的关系
                if tuple_in(alias_in, k) == True and len(alias_in) - len(k) < curr_delta:
                    curr_alias = k
                    curr_delta = len(alias_in) - len(k)
            alias_ref = curr_alias
            print("alias_in = {}. alias_ref = {}.".format(alias_in, alias_ref))
            return alias_ref
        
        def find_nearest_single_table(alias_in):
            """
            {Description}
            
            Args:
                arg1:
                arg2:
            Returns:
                res1:
                res2:
            """
            alias_ref = ()
            curr_alias, curr_delta = None, 1000
            print("self.single_table_true.keys = {}.".format(self.single_table_true.keys()))

            for k in self.single_table_true.keys():
                # 判断tuple之间的关系
                if tuple_in(alias_in, (k,)) == True and len(alias_in) - 1 < curr_delta:
                    curr_alias = (k,)
                    curr_delta = len(alias_in) - 1
            alias_ref = curr_alias
            print("alias_in = {}. alias_ref = {}.".format(alias_in, alias_ref))
            return alias_ref
        
        def card_infer(card1_true, card1_est, card2_est):
            """
            真实基数估计，目前用的办法肯定不科学，后续考虑优化
            
            Args:
                card1_true:
                card1_est:
                card2_est:
            Returns:
                car2_true:
            """
            return card1_true * card2_est / card1_est

        # 单表的基数暂时直接沿用estimation的结果
        for alias in self.single_table_estimation.keys():
            if alias not in single_table_mixed.keys():
                single_table_mixed[alias] = int(self.single_table_estimation[alias])

        # 多表子查询的基数补全
        for alias_tuple in self.subquery_estimation.keys():
            if alias_tuple not in subquery_mixed.keys():
                # 找最近的子查询基数
                alias_ref = find_nearest_subquery(alias_in = alias_tuple)   

                if alias_ref is not None:
                    card1_true, card1_est, card2_est = self.subquery_true[alias_ref], \
                        self.subquery_estimation[alias_ref], self.subquery_estimation[alias_tuple]
                    card2_est = card_infer(card1_true, card1_est, card2_est)
                    subquery_mixed[alias_tuple] = int(card2_est)    # 强制转换成整数
                    continue

                # 若子查询找不到，就尝试找最近的单表基数
                alias_ref = find_nearest_single_table(alias_in = alias_tuple)
                if alias_ref is not None:
                    card1_true, card1_est, card2_est = self.single_table_true[alias_ref[0]], \
                        self.single_table_estimation[alias_ref[0]], self.subquery_estimation[alias_tuple]
                    card2_est = card_infer(card1_true, card1_est, card2_est)
                    subquery_mixed[alias_tuple] = int(card2_est)    # 强制转换成整数
                    continue

                # 如果都找不到的话，直接raise ValueError
                raise ValueError("card_infer: subquery/single_table match all fail!")

        return subquery_mixed, single_table_mixed


    def complement_true_card(self, time_limit = -1, with_card_dict = False):
        """
        完成真实基数的补全
        
        Args:
            time_limit: 
            arg2:
        Returns:
            subquery_true: 真实子查询基数的字典
            single_table_true: 真实单表基数的字典
        """
        # print("complement_true_card: time_limit = {}.".format(time_limit))

        if with_card_dict == False:
            subquery_true, single_table_true = complement_true_card(query_parser=self.query_parser, 
                subquery_true=self.subquery_true, single_table_true=self.single_table_true,
                subquery_estimation=self.subquery_estimation, single_table_estimation=self.single_table_estimation,
                get_cardinalities_func=self.get_cardinalities, time_limit=time_limit)
        else:
            subquery_true, single_table_true = complement_true_card_with_hint(
                self.query_parser, self.subquery_true, self.single_table_true, 
                self.subquery_estimation, self.single_table_estimation, 
                self.get_cardinalities_with_hint, time_limit
            )

        self.subquery_true = subquery_true
        self.single_table_true = single_table_true
        return subquery_true, single_table_true

    def cardinality_format(self, subquery_dict: dict, single_table_dict: dict):
        """
        {Description}
    
        Args:
            subquery_dict:
            single_table_dict:
        Returns:
            result_str:
        """
        result_str = ""
        result_list = []

        # 考虑自动换行
        key_list = sorted(subquery_dict.keys())
        for k in key_list:
            result_list.append("{}: {}.".format(k, subquery_dict[k]))

        result_list.append("\n")
        key_list = sorted(single_table_dict.keys())
        for k in key_list:
            result_list.append("{}: {}.".format(k, single_table_dict[k]))

        result_str = "".join(result_list)
        return result_str

    


    def two_plan_comparison(self, subquery_dict1, single_table_dict1, subquery_dict2, \
                            single_table_dict2, keyword1, keyword2, debug = False):
        """
        {Description}
        
        Args:
            subquery_dict1:
            single_table_dict1:
            subquery_dict2:
            single_table_dict2:
            keyword1: 
            keyword2:
            debug:
        Returns:
            flag: 两个查询计划是否相同
            cost1: 第一个查询计划的代价
            cost2: 第二个查询计划的代价
            plan1: 第一个查询计划对象
            plan2: 第二个查询计划对象
        """

        # 获得两个不同基数下的plan
        plan1 = plan_evaluation_under_cardinality(workload = self.workload, query_ctrl = self.query_ctrl, \
            query_meta = self.query_meta, subquery_dict = subquery_dict1, single_table_dict = single_table_dict1)
        plan2 = plan_evaluation_under_cardinality(workload = self.workload, query_ctrl = self.query_ctrl, \
            query_meta = self.query_meta, subquery_dict = subquery_dict2, single_table_dict = single_table_dict2)

        # 设置数据库连接
        plan1.set_database_connector(db_conn=self.query_ctrl.db_conn)
        plan2.set_database_connector(db_conn=self.query_ctrl.db_conn)
        
        # 显示两个查询计划的细节
        if debug == True:
            # 展示基数估计之间的区别

            print("Display {} cardinalities".format(keyword1))
            card_str1 = self.cardinality_format(subquery_dict=subquery_dict1, single_table_dict=single_table_dict1)
            print(card_str1)
            print("Display {} cardinalities".format(keyword2))
            card_str2 = self.cardinality_format(subquery_dict=subquery_dict2, single_table_dict=single_table_dict2)
            print(card_str2)
            # 展示两个计划
            print("Display plan under {} cardinalities".format(keyword1))
            plan1.show_plan()
            print("Display plan under {} cardinalities".format(keyword2))
            plan2.show_plan()

        # 比较两个查询计划
        flag, cost1, cost2 = self.plan_comparison(plan1, plan2, \
            subquery_dict=subquery_dict1, single_table_dict=single_table_dict1)
        
        # 执行完保存结果
        if keyword1 == "true":
            self.verification_instance = {
                "query_meta": self.query_meta, "query_text": self.query_text,
                "card_dict": utils.pack_card_info(subquery_dict1, \
                    single_table_dict1, subquery_dict2, single_table_dict2),
                "true_plan": plan1, "est_plan": plan2,
                "true_cost": cost1, "est_cost": cost2,
                "p_error": cost2 / cost1,
                "valid": True
            }
        else:
            self.verification_instance = {
                "query_meta": self.query_meta, "query_text": self.query_text,
                "card_dict": utils.pack_card_info(subquery_dict2, \
                    single_table_dict2, subquery_dict1, single_table_dict1),
                "true_plan": plan2, "est_plan": plan1,
                "true_cost": cost2, "est_cost": cost1,
                "p_error": cost1 / cost2,
                "valid": True
            }
        
        if flag == True:
            if cost1 > cost2:
                raise ValueError("cost1 = {}. cost2 = {}.".format(cost1, cost2))
            return flag, cost1, cost2, plan1, plan2
        else:
            return flag, cost1, cost2, plan1, plan2


    def multi_plan_comparison(self, subquery_dict_list1: list, single_table_dict_list1: list, 
            subquery_dict2: dict, single_table_dict2: dict, keyword1: str, keyword2: str, debug = False):
        """
        比较一个查询计划和一批查询计划的结果
        
        Args:
            subquery_dict_list1:
            single_table_dict_list1:
            subquery_dict2:
            single_table_dict2:
            keyword1: 
            keyword2:
            debug:
        Returns:
            cmp_result_list: 比较结果列表，每一项内容如下
                flag 两个查询计划是否相同
                cost1: 第一个查询计划的代价
                cost2: 第二个查询计划的代价
                plan1: 第一个查询计划对象
                plan2: 第二个查询计划对象
        """

        # 获得两个不同基数下的plan
        # plan1 = plan_evaluation_under_cardinality(workload = self.workload, query_ctrl = self.query_ctrl, \
        #     query_meta = self.query_meta, subquery_dict = subquery_dict1, single_table_dict = single_table_dict1)
        # plan2 = plan_evaluation_under_cardinality(workload = self.workload, query_ctrl = self.query_ctrl, \
        #     query_meta = self.query_meta, subquery_dict = subquery_dict2, single_table_dict = single_table_dict2)

        # 把第二个结果添到list中
        subquery_dict_list1.append(subquery_dict2)
        single_table_dict_list1.append(single_table_dict2)
        assert len(subquery_dict_list1) == len(single_table_dict_list1)

        plan_num = len(subquery_dict_list1)
        query_list = [self.query_text for _ in range(plan_num)]
        meta_list = [self.query_meta for _ in range(plan_num)]

        total_plan_list = plan_evaluation_under_cardinality_parallel(self.workload, \
            query_list, meta_list, subquery_dict_list1, single_table_dict_list1)
        plan_list1, plan2 = total_plan_list[:-1], total_plan_list[-1]

        output_list = self.plan_comparison_parallel(total_plan_list[:-1], total_plan_list[-1], 
            subquery_dict_list1[:-1], single_table_dict_list1[:-1])

        cmp_result_list = []
        try:
            for (flag, cost1, cost2), plan1 in zip(output_list, plan_list1):
                cmp_result_list.append((flag, cost1, cost2, plan1, plan2))
        except TypeError as e:
            print(f"multi_plan_comparison: meet TypeError. output_list = {output_list}.")
            raise e

        return cmp_result_list
        # 设置数据库连接
        # plan1.set_database_connector(db_conn=self.query_ctrl.db_conn)
        # plan2.set_database_connector(db_conn=self.query_ctrl.db_conn)
        
        # # 显示两个查询计划的细节
        # if debug == True:
        #     # 展示基数估计之间的区别

        #     print("Display {} cardinalities".format(keyword1))
        #     card_str1 = self.cardinality_format(subquery_dict=subquery_dict1, single_table_dict=single_table_dict1)
        #     print(card_str1)
        #     print("Display {} cardinalities".format(keyword2))
        #     card_str2 = self.cardinality_format(subquery_dict=subquery_dict2, single_table_dict=single_table_dict2)
        #     print(card_str2)
        #     # 展示两个计划
        #     print("Display plan under {} cardinalities".format(keyword1))
        #     plan1.show_plan()
        #     print("Display plan under {} cardinalities".format(keyword2))
        #     plan2.show_plan()

        # 比较两个查询计划
        # flag, cost1, cost2 = self.plan_comparison(plan1, plan2, \
        #     subquery_dict=subquery_dict1, single_table_dict=single_table_dict1)
        
        # if flag == True:
        #     if cost1 > cost2:
        #         raise ValueError("cost1 = {:.2f}. cost2 = {:.2f}.".format(cost1, cost2))
        #     return flag, cost1, cost2, plan1, plan2
        # else:
        #     return flag, cost1, cost2, plan1, plan2


    @utils.timing_decorator
    def two_plan_verification_under_constraint(self,subquery_dict1, single_table_dict1, subquery_dict2, \
        single_table_dict2, keyword1, keyword2, target_table, return_plan = False):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            plan_flag: 两个查询计划是否相同
            table_flag: 真实计划下最后一张表和目标的table是否匹配
            cost1: 第一个查询计划的代价
            cost2: 第二个查询计划的代价
            plan1(optional): 第一个查询计划
            plan2(optional): 第二个查询计划
        """

        debug = False
        plan_flag, cost1, cost2, plan1, plan2 = self.two_plan_comparison(subquery_dict1, \
            single_table_dict1, subquery_dict2, single_table_dict2, keyword1, keyword2, debug)
        
        if keyword1 == "true" or "mixed":
            jo_str = plan1.leading[len("Leading"):]
            local_analyzer = join_analysis.JoinOrderAnalyzer(join_order_str=jo_str)
        elif keyword2 == "true" or "mixed":
            jo_str = plan2.leading[len("Leading"):]
            local_analyzer = join_analysis.JoinOrderAnalyzer(join_order_str=jo_str)
        else:
            raise ValueError(f"two_plan_verification_under_constraint: kw1 = {keyword1}. kw2 = {keyword2}.")

        target_alias = self.query_parser.inverse_alias[target_table]
        if local_analyzer.top_table_eval(target_alias) == True:
            # 理想的结果
            table_flag = True
        else:
            # 失败的结果
            table_flag = False

        alias_order = local_analyzer.get_leading_order()
        alias_inverse = workload_spec.get_alias_inverse(self.workload)
        table_order = [alias_inverse[alias] for alias in alias_order]

        # if table_flag == False:
        #     # 打印失败的结果
        #     print(f"two_plan_verification_under_constraint: plan_flag = {plan_flag}. table_flag = {table_flag}. cost1 = {cost1:.2f}. "\
        #         f"cost2 = {cost2:.2f}. target_alias = {target_alias}. jo_str = {jo_str}. leading = {alias_order}.")

        # local_analyzer.is_bushy()
        # 执行完保存结果
        if keyword1 == "true":
            self.verification_instance = {
                "query_meta": self.query_meta, "query_text": self.query_text,
                "card_dict": utils.pack_card_info(subquery_dict1, \
                    single_table_dict1, subquery_dict2, single_table_dict2),
                "true_plan": plan1, "est_plan": plan2,
                "true_cost": cost1, "est_cost": cost2,
                "p_error": cost2 / cost1,
                "valid": table_flag,
                # 2024-03-12: 添加新的成员
                "is_bushy": local_analyzer.is_bushy(),
                "join_order": table_order
            }
        else:
            self.verification_instance = {
                "query_meta": self.query_meta, "query_text": self.query_text,
                "card_dict": utils.pack_card_info(subquery_dict2, \
                    single_table_dict2, subquery_dict1, single_table_dict1),
                "true_plan": plan2, "est_plan": plan1,
                "true_cost": cost2, "est_cost": cost1,
                "p_error": cost1 / cost2,
                "valid": table_flag,
                # 2024-03-12: 添加新的成员
                "is_bushy": local_analyzer.is_bushy(),
                "join_order": table_order
            }

        if return_plan:
            return plan_flag, table_flag, cost1, cost2, plan1, plan2
        else:
            return plan_flag, table_flag, cost1, cost2
        

    @utils.timing_decorator
    def multi_plan_verification_under_constraint(self, subquery_dict_list1, single_table_dict_list1, 
        subquery_dict2, single_table_dict2, keyword1, keyword2, target_table, return_plan = False):
        """
        {Description}
    
        Args:
            subquery_dict_list1: 
            single_table_dict_list1:
            subquery_dict2: 
            single_table_dict2: 
            keyword1: 
            keyword2: 
            target_table: 
            return_plan: 
        Returns:
            out_list: (plan_flag, table_flag, cost1, cost2, plan1, plan2) / (plan_flag, table_flag, cost1, cost2)
        """
        def local_proc_func(item):
            # 本地的处理函数
            plan_flag, cost1, cost2, plan1, plan2 = item

            if keyword1 == "true" or "mixed":
                jo_str = plan1.leading[len("Leading"):]
                local_analyzer = join_analysis.JoinOrderAnalyzer(join_order_str=jo_str)
            elif keyword2 == "true" or "mixed":
                jo_str = plan2.leading[len("Leading"):]
                local_analyzer = join_analysis.JoinOrderAnalyzer(join_order_str=jo_str)
            else:
                raise ValueError(f"two_plan_verification_under_constraint: kw1 = {keyword1}. kw2 = {keyword2}.")

            target_alias = self.query_parser.inverse_alias[target_table]
            if local_analyzer.top_table_eval(target_alias) == True:
                # 理想的结果
                table_flag = True
            else:
                # 失败的结果
                table_flag = False

            # return item
            if return_plan == True:
                return plan_flag, table_flag, cost1, cost2, plan1, plan2
            else:
                return plan_flag, table_flag, cost1, cost2

        # 返回的每一个item为(plan_flag, cost1, cost2, plan1, plan2)
        cmp_result_list = self.multi_plan_comparison(subquery_dict_list1, \
            single_table_dict_list1, subquery_dict2, single_table_dict2, keyword1, keyword2)

        out_list = [local_proc_func(item) for item in cmp_result_list]
        return out_list
    

    # @utils.timing_decorator
    def two_plan_verification(self, subquery_dict1, single_table_dict1, subquery_dict2, \
                              single_table_dict2, keyword1, keyword2, debug = False):
        """
        查询计划比较，最终cost的比较参考的是subquery_dict1和single_table_dict1
        
        Args:
            subquery_dict1:
            single_table_dict1:
            subquery_dict2:
            single_table_dict2:
            keyword1: 
            keyword2:
            debug:
        Returns:
            flag: 两个查询计划是否相同
            cost1: 第一个查询计划的代价
            cost2: 第二个查询计划的代价
        """
        flag, cost1, cost2, _, _ = self.two_plan_comparison(subquery_dict1, \
            single_table_dict1, subquery_dict2, single_table_dict2, keyword1, keyword2, debug)
        return flag, cost1, cost2

    def top_query_estimation(self,):
        """
        顶层查询的误差估计
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        pair_list = [(i, len(i)) for i in self.subquery_estimation.keys()]
        pair_list.sort(key=lambda a: a[1], reverse=True)

        top_query_key = pair_list[0][0]
        top_query_estimation = self.subquery_estimation[top_query_key]
        top_query_true = self.subquery_true[top_query_key]

        return top_query_estimation, top_query_true

    def mixed_card_plan_verification(self, budget = 0, debug = False):
        """
        混合基数下的查询计划验证
    
        Args:
            budget: 可用的查询预算
            debug: 是否打印细节
        Returns:
            flag: 是否产生了查询计划扰动
            cost1: mixed_plan在mixed_card下的cost
            cost2: estimation_plan在mixed_card下的cost
            这里满足cost1 <= cost2
        """
        # 补全基数信息，这里考虑使用budget
        subquery_mixed, single_table_mixed = self.complement_missing_mixed_card()   

        # 获得两个不同基数下的plan
        plan1 = plan_evaluation_under_cardinality(workload = self.workload, query_ctrl = self.query_ctrl, \
            query_meta = self.query_meta, subquery_dict = subquery_mixed, single_table_dict = single_table_mixed)
        plan2 = plan_evaluation_under_cardinality(workload = self.workload, query_ctrl = self.query_ctrl, \
            query_meta = self.query_meta, subquery_dict = self.subquery_estimation, single_table_dict = self.single_table_estimation)

        # 设置数据库连接
        plan1.set_database_connector(db_conn=self.query_ctrl.db_conn)
        plan2.set_database_connector(db_conn=self.query_ctrl.db_conn)
        
        # 显示两个查询计划的细节
        if debug == True:
            # 展示基数估计之间的区别

            print("Display mixed cardinalities")
            mixed_card_str = self.cardinality_format(subquery_dict=subquery_mixed, single_table_dict=single_table_mixed)
            print(mixed_card_str)
            print("Display estimation cardinalities")
            estimation_card_str = self.cardinality_format(subquery_dict=self.subquery_estimation, \
                                                          single_table_dict=self.single_table_estimation)
            print(estimation_card_str)
            # 展示两个计划
            print("Display plan under mixed cardinalities")
            plan1.show_plan()
            print("Display plan under estimation cardinalities")
            plan2.show_plan()

        # 比较两个查询计划
        flag, cost1, cost2 = self.plan_comparison(plan1, plan2, \
            subquery_dict=subquery_mixed, single_table_dict=single_table_mixed)
        
        if flag == True:
            if cost1 > cost2:
                raise ValueError("cost1 = {}. cost2 = {}.".format(cost1, cost2))
            return flag, cost1, cost2
        else:
            return flag, cost1, cost2

    def mixed_card_quality_evaluation(self, debug = True):
        """
        评测混合基数的质量，和真实基数的查询计划对比
    
        Args:
            debug:
            arg2:
        Returns:
            flag:
            cost1:
            cost2:
            这里满足cost1 <= cost2
        """
        self.subquery_true, self.single_table_true = self.complement_true_card()
        subquery_mixed, single_table_mixed = self.complement_missing_mixed_card()

        res = self.two_plan_verification(subquery_dict1=self.subquery_true, single_table_dict1=self.single_table_true,
                                         subquery_dict2=subquery_mixed, single_table_dict2=single_table_mixed,
                                         keyword1="true", keyword2="mixed", debug=debug)
        return res
    
    def true_card_plan_verification(self, time_limit = -1, debug = True, with_card_dict = False):
        """
        真实基数下的查询计划验证
        
        Args:
            time_limit:
            debug:
            with_card_dict:
        Returns:
            flag:
            cost1:
            cost2:
            这里满足cost1 <= cost2
        """
        # print("true_card_plan_verification: time_limit = {}.".format(time_limit))
        self.subquery_true, self.single_table_true = self.complement_true_card(\
            time_limit=time_limit, with_card_dict = with_card_dict)

        # 
        flag = self.invalid_evaluation(card_dict=self.subquery_true)
        if flag == True:
            # 比较两个查询计划
            res = self.two_plan_verification(subquery_dict1=self.subquery_true, single_table_dict1=self.single_table_true,
                                            subquery_dict2=self.subquery_estimation, single_table_dict2=self.single_table_estimation,
                                            keyword1="true", keyword2="estimation", debug=debug)
            # 返回结果
            return res
        else:
            #
            return True, -1, -1
        

    def true_card_plan_async_test(self,):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        subquery_diff_dict, single_table_diff_dict = node_utils.parse_missing_card(\
            query_parser=self.query_parser, subquery_ref=self.subquery_estimation, \
            single_table_ref=self.single_table_estimation, subquery_missing=self.subquery_true, \
            single_table_missing=self.single_table_true, out_mode="query")

        query_list = []
        for k, v in subquery_diff_dict.items():
            query_list.append(v)

        for k, v in single_table_diff_dict.items():
            query_list.append(v)

        return query_list


    def true_card_plan_async_explore(self,):
        """
        {Description}
        
        Args:
            None
        Returns:
            subquery_res: 
            single_table_res:
        """
        subquery_diff_dict, single_table_diff_dict = node_utils.parse_missing_card(\
            query_parser=self.query_parser, subquery_ref=self.subquery_estimation, \
            single_table_ref=self.single_table_estimation, subquery_missing=self.subquery_true, \
            single_table_missing=self.single_table_true, out_mode="both")
        
        print("subquery_diff_dict = {}.".format(subquery_diff_dict))
        print("single_table_diff_dict = {}.".format(single_table_diff_dict))
        
        # TODO: 当前的实现，所有query均对应一个进程，之后考虑优化

        subquery_res, single_table_res = {}, {}
        out_path = construct_input.output_path_from_meta(query_meta=self.query_meta)    
        self.async_out = out_path
        for idx, (k, v) in enumerate(subquery_diff_dict.items()):
            curr_query, curr_meta = v
            # curr_conn = db_conn_list[idx]
            curr_proc = multiprocessing.Process(target=construct_input.\
                    proc_function, args=(self.workload, curr_query, k, out_path))
            curr_proc.start()
            # 查询进程的相关信息，之后考虑把附带card_hint的cost也放进来
            # 并采用greedy-scheduling的办法优化连接数
            
            info_dict = {
                "query": curr_query,
                "meta": curr_meta
            }
            subquery_res[k] = info_dict

        for idx, (k, v) in enumerate(single_table_diff_dict.items()):
            curr_query, curr_meta = v

            curr_proc = multiprocessing.Process(target=construct_input.\
                proc_function, args=(self.workload, curr_query, k, out_path))
            curr_proc.start()
            # curr_proc.join()
            info_dict = {
                "query": curr_query,
                "meta": curr_meta
            }
            single_table_res[k] = info_dict

        return subquery_res, single_table_res
    
    def get_subset_true_card(self, query_key):
        """
        给定query_key的情况下，获得真实基数的子集
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        if isinstance(query_key, str):
            # single_table的情况，直接跳过
            return {}
        elif isinstance(query_key, tuple):
            # subquery的情况
            result_dict = {
                "subquery": {},
                "single_table": {}
            }
            for k, v in self.subquery_true.items():
                if len(set(k).union(query_key)) == len(set(query_key)):
                    # k完全包含在query_key中的情况
                    result_dict['subquery'][k] = v

            for k, v in self.single_table_true.items():
                if k in query_key:
                    result_dict['single_table'][k] = v

            # 按理说是空的才对
            # print("get_subset_true_card: query_key = {}. result_dict = {}.".format(query_key, result_dict))
            # print("get_subset_true_card: subquery_true = {}.".format(self.subquery_true))
            # print("get_subset_true_card: single_table_true = {}".format(self.single_table_true))

            return result_dict

    def get_extension_signature(self):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        return construct_input.get_query_signature(query_meta=self.query_meta)


    def true_card_plan_async_under_constaint(self, proc_num, timeout = None, with_card_dict = False):
        """
        在具有限制条件下的异步真实基数探索
        
        Args:
            proc_num: 分配的进程数目
            arg2:
        Returns:
            subquery_res: 
            single_table_res:
        """
        # print("true_card_plan_async_under_constaint: len(subquery_estimation) = {}. len(single_table_estimation) = {}. \
        #       len(subquery_true) = {}. len(single_table_true) = {}.".format(len(self.subquery_estimation), \
        #        len(self.single_table_estimation), len(self.subquery_true), len(self.single_table_true)))
        
        subquery_diff_dict, single_table_diff_dict = node_utils.parse_missing_card(\
            query_parser=self.query_parser, subquery_ref=self.subquery_estimation, \
            single_table_ref=self.single_table_estimation, subquery_missing=self.subquery_true, \
            single_table_missing=self.single_table_true, out_mode="both")
        
        self.subquery_diff, self.single_table_diff = subquery_diff_dict, single_table_diff_dict

        # print("true_card_plan_async_under_constaint: len(subquery_diff_dict) = {}. len(single_table_diff_dict) = {}. ".\
        #       format(len(subquery_diff_dict), len(single_table_diff_dict)))
        
        subquery_res, single_table_res = {}, {}
        out_path = construct_input.output_path_from_meta(query_meta=self.query_meta)
        self.async_out = out_path   # 设计输出路径
        
        query_list, key_list, meta_list = [], [], []
        for k, v in list(subquery_diff_dict.items()) + list(single_table_diff_dict.items()):
            curr_query, curr_meta = v
            query_list.append(curr_query)
            key_list.append(k)
            meta_list.append(curr_meta)
        
        card_dict_list = [self.get_subset_true_card(key) for key in key_list]
        _, cost_list = construct_input.query_cost_estimation(workload = self.workload,
                        query_list=query_list, card_dict_list=card_dict_list)
        
        # print("true_card_plan_async_under_constaint: len(query_list) = {}. len(key_list) = {}. len(meta_list) = {}. len(cost_list) = {}".\
        #       format(len(query_list), len(key_list), len(meta_list), len(cost_list)))

        index_assign_list, _, _ = construct_input.multi_query_scheduling(query_list=query_list, \
                        cost_list=cost_list, proc_num=proc_num)
        
        node_signature = self.get_extension_signature()
        # utils.trace("launch_async_exploration: node_signature = {}. out_path = {}. proc_num = {}.".\
        #             format(node_signature, out_path, len(index_assign_list)))

        for index_list in index_assign_list:
            # print("true_card_plan_async_under_constaint: index_list = {}.".format(index_list))
            query_local, meta_local, key_local, cost_local = [utils.list_index(val_list, \
                index_list) for val_list in [query_list, meta_list, key_list, cost_list]]
            
            # workload: str, query_list: Any, key_list: Any, out_path: Any
            if timeout is None:
                if with_card_dict == False:
                    # without基数的情况
                    curr_proc = multiprocessing.Process(target=construct_input.\
                        proc_multi_queries, args=(self.workload, query_local, key_local, out_path))
                else:
                    # with基数的情况
                    card_dict_local = [self.get_subset_true_card(key) for key in key_local]
                    curr_proc = multiprocessing.Process(target=construct_input.proc_multi_queries_under_card_dict, \
                        args=(self.workload, query_local, key_local, card_dict_local, out_path))
            else:
                if with_card_dict == False:
                    curr_proc = multiprocessing.Process(target=construct_input.proc_multi_queries_under_timeout, 
                        args=(self.workload, query_local, key_local, out_path, timeout))
                else:
                    # 2024-03-27: 新增的选择，有card_dict同时有timeout
                    # print(f"true_card_plan_async_under_constraint: with_card_dict = {with_card_dict}. "\
                    #       f"timeout = {timeout}. query_local = {query_local}.")
                    card_dict_local = [self.get_subset_true_card(key) for key in key_local]
                    curr_proc = multiprocessing.Process(target=construct_input.proc_multi_queries_under_card_dict, \
                        args=(self.workload, query_local, key_local, card_dict_local, out_path, timeout))


            # print("true_card_plan_async_under_constaint: construct process. out_path = {}.".format(out_path))
            curr_proc.start()

            # 20231224: 添加额外的cost信息
            for query, meta, key, cost in zip(query_local, meta_local, key_local, cost_local):
                if isinstance(key, tuple):
                    info_dict = {
                        "query": query,
                        "meta": meta,
                        "cost": cost
                    }
                    subquery_res[key] = info_dict
                elif isinstance(key, str):
                    info_dict = {
                        "query": query,
                        "meta": meta,
                        "cost": cost
                    }
                    single_table_res[key] = info_dict
                else:
                    raise TypeError(f"true_card_plan_async_under_constaint: type(key) = {type(key)}.")

        return subquery_res, single_table_res

    # @utils.timing_decorator
    def load_external_card(self, subquery_dict, single_table_dict):
        """
        加载外部的card_dict到extension中
    
        Args:
            arg1:
            arg2:
        Returns:
            flag:
            subquery_true: 
            single_table_true:
        """
        node_utils.dict_complement(self.subquery_true, subquery_dict)
        node_utils.dict_complement(self.single_table_true, single_table_dict)

        # print("load_external_card: query_meta = {}.".format(self.query_meta))
        # print("load_external_card: subquery_true = {}. single_table_true = {}. subquery_estimation = {}. single_table_estimation = {}. subquery_dict = {}. single_table_dict = {}.".\
        #     format(self.subquery_true.keys(), self.single_table_true.keys(), self.subquery_estimation.keys(), self.single_table_estimation.keys(), subquery_dict.keys(), single_table_dict.keys()))

        if len(self.subquery_true) > len(self.subquery_estimation) or \
            len(self.single_table_true) > len(self.single_table_estimation):
            raise ValueError("sub_true = {}. sub_est = {}. single_true = {}. single_est = {}".\
                format(len(self.subquery_true), len(self.subquery_estimation), \
                       len(self.single_table_true), len(self.single_table_estimation)))

        flag = len(self.subquery_true) == len(self.subquery_estimation) and \
               len(self.single_table_true) == len(self.single_table_estimation)
        
        # print("load_external_card: flag = {}. signature = {}. len(subquery_true) = {}. len(single_table_true) = {}. len(subquery_estimation) = {}. len(single_table_estimation) = {}. len(subquery_dict) = {}. len(single_table_dict) = {}.".\
        #     format(flag, self.get_extension_signature(), len(self.subquery_true), len(self.single_table_true), len(self.subquery_estimation), len(self.single_table_estimation), len(subquery_dict), len(single_table_dict)))
        
        return flag, self.subquery_true, self.single_table_true

    def true_card_plan_async_integration(self, subquery_res: dict, single_table_res: dict):
        """
        真实Plan验证结果的集成
        
        Args:
            None
        Returns:
            is_complete:
            result:
        """
        subquery_diff_card, single_table_diff_card = \
            process_output.get_async_cardinalities(out_path=self.async_out)

        missing_subquery, missing_single_table = [], []
        for k, _ in subquery_res.items():
            if k not in subquery_diff_card.keys():
                missing_subquery.append(k)

        for k, _ in single_table_res.items():
            if k not in single_table_diff_card.keys():
                missing_single_table.append(k)

        if len(missing_single_table + missing_subquery) == 0:
            is_complete = True
            node_utils.dict_complement(self.subquery_true, subquery_diff_card)
            node_utils.dict_complement(self.single_table_true, single_table_diff_card)
            flag, cost1, cost2, plan1, plan2 = self.two_plan_comparison(subquery_dict1 = self.subquery_true,
                single_table_dict1 = self.single_table_true, subquery_dict2 = self.subquery_estimation,
                single_table_dict2 = self.single_table_estimation, keyword1 = "true", keyword2 = "estimation",
                debug = False)
            return is_complete, (flag, cost1, cost2, plan1, plan2)
        else:
            is_complete = False
            return is_complete, (missing_subquery, missing_single_table)

    def true_card_plan_comparison(self, time_limit = -1, debug = True):
        """
        真实基数下的查询计划验证
        
        Args:
            time_limit:
            debug:
        Returns:
            flag:
            cost1:
            cost2:
            这里满足cost1 <= cost2
        """
        # print("true_card_plan_verification: time_limit = {}.".format(time_limit))
        self.subquery_true, self.single_table_true = self.complement_true_card(time_limit=time_limit)

        # 
        flag = self.invalid_evaluation(card_dict=self.subquery_true)
        if flag == True:
            # 比较两个查询计划
            res = self.two_plan_comparison(subquery_dict1=self.subquery_true, single_table_dict1=self.single_table_true,
                                            subquery_dict2=self.subquery_estimation, single_table_dict2=self.single_table_estimation,
                                            keyword1="true", keyword2="estimation", debug=debug)
            # 返回结果
            return res
        else:
            #
            return True, -1, -1, None, None

    def plan_comparison(self, plan1:physical_plan_info.PhysicalPlan, plan2:physical_plan_info.PhysicalPlan, \
                        subquery_dict: dict, single_table_dict: dict):
        """
        比较估计基数的计划和混合基数的计划，以及混合基数下两者的差值

        Args:
            plan1: 混合/真实基数下的查询计划
            plan2: 估计基数下的查询计划
            subquery_dict: 混合/真实子查询基数
            single_table_dict: 混合/真实单表基数
        Returns:
            flag: True代表查询计划等价，False代表查询计划不等价
            cost_difference:
        """

        flag, error_dict = physical_plan_info.physical_comparison((plan1.leading, \
            plan1.join_ops, plan1.scan_ops), (plan2.leading, plan2.join_ops, plan2.scan_ops))  # 调用比较物理计划的函数
        if flag == True:
            # print("两个查询计划等价，探索失败")
            # return flag, 0, 0
            cost = plan1.get_plan_cost(subquery_dict, single_table_dict)
            return flag, cost, cost
        else:
            cost1 = plan1.get_plan_cost(subquery_dict, single_table_dict)
            cost2 = plan2.get_plan_cost(subquery_dict, single_table_dict)
            # print("两个查询计划不等价，探索成功. cost1 = {}. cost2 = {}.".format(cost1, cost2))
            return flag, cost1, cost2

    def plan_comparison_parallel(self, plan_list1: List[physical_plan_info.PhysicalPlan], \
            plan2: physical_plan_info.PhysicalPlan, subquery_dict_list: list, single_table_dict_list: list):
        """
        将一个查询计划和多个查询计划去比较
    
        Args:
            plan_list1:
            plan2:
            subquery_dict_list:
            single_table_dict_list:
        Returns:
            output_list: 每一项包含的元素为(flag, cost1, cost2)
            return2:
        """
        # 确保输入的列表长度相同
        assert len(plan_list1) == len(subquery_dict_list) == len(single_table_dict_list), \
            f"plan_comparison_parallel: plan_list1 = {len(plan_list1)}. subquery_dict_list = {len(subquery_dict_list)}. single_table_dict_list = {len(single_table_dict_list)}."

        def wrap_func(plan1: physical_plan_info.PhysicalPlan, plan2: physical_plan_info.PhysicalPlan):
            flag, error_dict = physical_plan_info.physical_comparison((plan1.leading, \
                plan1.join_ops, plan1.scan_ops), (plan2.leading, plan2.join_ops, plan2.scan_ops))  # 调用比较物理计划的函数
            return flag
        
        output_list = []
        flag_list = [wrap_func(plan1, plan2) for plan1 in plan_list1]   # 

        candidate_query_list = []
        db_conn = self.query_ctrl.db_conn
        for idx, flag in enumerate(flag_list):
            plan1 = plan_list1[idx]
            subquery_dict = subquery_dict_list[idx]
            single_table_dict = single_table_dict_list[idx]
            if flag == True:
                plan1.set_database_connector(db_conn)   # 设置
                candidate_query_list.append(plan1.get_specific_hint_query(subquery_dict, single_table_dict))
            else:
                plan1.set_database_connector(db_conn)
                plan2.set_database_connector(db_conn)
                candidate_query_list.append(plan1.get_specific_hint_query(subquery_dict, single_table_dict))
                candidate_query_list.append(plan2.get_specific_hint_query(subquery_dict, single_table_dict))

        conn_pool: connection_parallel.ConnectionPool = connection_parallel.get_conn_pool_by_workload(self.workload)

        candidate_plan_list = conn_pool.get_plan_parallel(candidate_query_list)
        # candidate_cost_list = [plan[0][0][0]['Plan']['Total Cost'] for plan in candidate_plan_list]
        candidate_cost_list = [plan['Total Cost'] for plan in candidate_plan_list]

        curr_idx = 0

        for flag in flag_list:
            if flag == True:
                cost1, cost2 = candidate_cost_list[curr_idx], candidate_cost_list[curr_idx]
                output_list.append((flag, cost1, cost2))
                curr_idx += 1
            else:
                cost1, cost2 = candidate_cost_list[curr_idx], candidate_cost_list[curr_idx + 1]
                output_list.append((flag, cost1, cost2))
                curr_idx += 2

        # 2024-03-10: 确认curr_idx最终所处的位置正确
        assert curr_idx == len(candidate_cost_list)
        return output_list
    

    def cardinality_tuning(self, step, ratio):
        """
        针对真实基数估计的调整，以此分析最需要测试的真实基数
    
        Args:
            step:
            ratio:
        Returns:
            return1:
            return2:
        """
        pass
    
    def cardinality_exploration(self, budget):
        """
        针对真实基数的探索
        
        Args:
            budget: 探索的总预算
            arg2:
        Returns:
            cardinality_dict:
            res2:
        """
        pass

    def construct_query_instance(self, query_instance_ref: QueryInstance):
        """
        参考创建query_instance_ref，新的query_instance
        
        Args:
            query_instance_ref:
            arg2:
        Returns:
            res1:
            res2:
        """
        # 创建查询实例
        res_instance = QueryInstance(workload=self.workload, query_meta=self.query_meta, 
                                     data_manager_ref=query_instance_ref.data_manager, ce_handler=query_instance_ref.ce_handler,
                                     query_controller=self.query_ctrl, grid_preprocessor=query_instance_ref.grid_preprocessor,
                                     bins_builder=query_instance_ref.bins_builder)
        
        # 添加基数
        res_instance.add_estimation_card(self.subquery_estimation, mode="subquery")
        res_instance.add_estimation_card(self.single_table_estimation, mode="single_table")
        res_instance.add_true_card(self.subquery_true, mode="subquery")
        res_instance.add_true_card(self.single_table_true, mode="single_table")

        return res_instance
    
# %%

def get_extension_instance(workload, query_text, query_meta, subquery_dict, \
                           single_table_dict, query_ctrl:query_exploration.QueryController = None):
    """
    {Description}

    Args:
        arg1:
        arg2:
    Returns:
        extension_instance:
        return2:
    """
    if query_ctrl is None:
        query_ctrl = query_exploration.QueryController(workload=workload)

    extension_instance = ExtensionInstance(query_text=query_text, query_meta=query_meta, query_ctrl=query_ctrl, 
                                            subquery_estimation=subquery_dict, single_table_estimation=single_table_dict, 
                                            subquery_true={}, single_table_true={}, external_info={})
    # extension_instance.complement_estimation_card()

    return extension_instance

# %%

def get_extension_from_query_instance(query_instance: QueryInstance, new_query = None, new_meta = None):
    """
    {Description}
    
    Args:
        query_instance:
    Returns:
        query_extension:
    """
    if new_query is None:
        new_query = query_instance.query_text

    if new_meta is None:
        new_meta = query_instance.query_meta

    qi = query_instance
    extension_instance = ExtensionInstance(query_text=new_query, query_meta=new_meta, query_ctrl=qi.query_ctrl,
                                           subquery_estimation=qi.estimation_card_dict, single_table_estimation=qi.estimation_single_table,
                                           subquery_true=qi.true_card_dict, single_table_true=qi.true_single_table, external_info={})

    return extension_instance

