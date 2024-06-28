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
import graphviz
from plan import node_extension, node_query
from plan.node_query import QueryInstance
from query import query_construction
from workload import physical_plan_info
from utility.workload_parser import SQLParser
from plan.node_extension import ExtensionInstance
from plan.join_analysis import JoinOrderAnalyzer
from utility import utils

# %%

class TemplateAnalyzer(object):
    """
    针对Template的结果分析类

    Members:
        field1:
        field2:
    """

    def __init__(self, template_plan):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        self.template_plan = template_plan
        self.generate_random_query = self.template_plan.generate_random_query
        self.get_true_cardinality = self.template_plan.get_true_cardinality
        self.get_cardinalities = self.template_plan.ce_handler.get_cardinalities

    def valid(self, query_meta, label, info_dict):
        """
        判断query是否符合要求
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        flag = True
        min_card = info_dict.get("min_card", -1000)
        max_card = info_dict.get("max_card", 1e9)

        flag = flag and (min_card <= label <=max_card)
        return flag
    
    def true_card_verification(self, sample_num = 5):
        """
        手动验证true_card结果是否正确，主要是基数估计的结果过于离谱了。。。
        目前看来有可能是对的捏
    
        Args:
            sample_num:
            arg2:
        Returns:
            pair_list:
            return2:
        """
        query_list, label_list = [], []
        for _ in range(sample_num):
            test_query, test_meta = self.generate_random_query()    
            label = self.get_true_cardinality(in_meta=test_meta, query_text=test_query, mode="subquery")

            query_list.append(test_query)
            label_list.append(label)

        return list(zip(query_list, label_list))

    
    def random_query_evaluation(self, sample_num, extra_info = {}):
        """
        随机生成一批查询，评测效果，返回统计结果

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        sample_limit = sample_num * 5
        query_list, meta_list, label_list, estimation_list = \
            [], [], [], []
        
        for _ in range(sample_limit):
            test_query, test_meta = self.generate_random_query()    
            label = self.get_true_cardinality(in_meta=test_meta, query_text=test_query, mode="subquery")

            if self.valid(query_meta=test_meta, label=label, info_dict=extra_info):
                query_list.append(test_query)
                meta_list.append(test_meta)
                label_list.append(label)

            if len(label_list) >= sample_num:
                break

        estimation_list = self.get_cardinalities(query_list=query_list)
        return self.result_analysis(label_list=label_list, estimation_list=estimation_list)
    
    def result_analysis(self, label_list, estimation_list):
        """
        通过分位数的方式分析结果
    
        Args:
            label_list:
            estimation_list:
        Returns:
            return1:
            return2:
        """
        one_side_q_error_list = [i / j for (i, j) in zip(label_list, estimation_list)]
        quantile_list = np.quantile(one_side_q_error_list, q=[0.1 * i for i in range(10)])
        return quantile_list

    def iterative_query_evaluation(self, iter_num):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        pass



# %%

class InstanceAnalyzer(object):
    """
    针对QueryInstance的分析器，疏解一部分QueryInstance的功能(主要是拓展meta相关的)，
    并且增加新的分析内容

    Members:
        field1:
        field2:
    """

    def __init__(self, query_instance: QueryInstance, split_budget = 100):
        """
        {Description}

        Args:
            query_instance: 查询实例
            arg2:
        """
        self.instance = query_instance
        self.split_budget = split_budget

    def root_tuning(self, table_subset, action):
        """
        针对根节点谓词的调优
    
        Args:
            table_subset:
            action: 调优针对的动作
        Returns:
            return1:
            return2:
        """
        result_dict = {}
        for t in table_subset:
            result_dict[t] = self.multi_plans_tuning_under_multi_meta(\
                action_table=action, target_table=t, adjust_num=10, 
                split_budget=self.split_budget)
        return result_dict


    def create_extension_instance(self, ext_query_meta, ext_query_text):
        """
        创建extension的实例
    
        Args:
            ext_query_meta: 外部的query_meta
            ext_query_text: 外部的query_text，包含了一部分foreign_mapping的信息，所以需要导入进去
        Returns:
            extension_res:
            return2:
        """
        instance = self.instance
        # query_meta的合法性检测
        extra_meta = instance.get_extra_schema_meta(ext_query_meta)
        if len(extra_meta[0]) != 1:
            # 额外的table数目不为1
            raise ValueError("create_extension_instance: ext_query_meta = {}. self.query_meta = {}.".\
                    format(ext_query_meta, instance.query_meta))

        # 更新全局主外键映射
        curr_parser = SQLParser(sql_text=ext_query_text, workload=instance.workload)
        curr_parser.update_global_foreign_mapping()

        # 将估计的基数根据query_meta的实例进行补全
        subquery_dict, single_table_dict = instance.get_query_cardinalities(query_meta = ext_query_meta)    

        # 创建Extension实例
        extension_res = ExtensionInstance(query_text=ext_query_text, query_meta = ext_query_meta, \
            query_ctrl = instance.query_ctrl, subquery_estimation = subquery_dict, \
            single_table_estimation = single_table_dict, subquery_true = instance.true_card_dict, \
            single_table_true = instance.true_single_table, external_info = {})

        # 返回结果
        return extension_res


    def cardinalities_tuning(self, object_table, subquery_dict, single_table_dict, tuning_num, \
                             column_list, column_values, return_card = False, mode = "global"):
        """
        针对当前新增表的基数进行调优，需要全部转换成整数
        
        Args:
            object_table: 目标表
            subquery_dict: 子查询基数的字典
            single_table_dict: 单表基数的字典
            tuning_num: 调优的测试数目
            column_list: 使用column的列表
            column_values: 目标表column上的具体取值
            mode: 调优的模式，"global"是在整体范围内调优，"local"是在局部范围内调优
        Returns:
            predicate_candidates:
        """
        predicate_candidates = []
        table_alias = self.instance.get_alias(table_name = object_table)

        print("InstanceAnalyzer.cardinalities_tuning: subquery_dict = {}.".format(subquery_dict))
        print("InstanceAnalyzer.cardinalities_tuning: single_table_dict = {}.".format(single_table_dict))

        def adjust_by_factor(factor):
            subquery_adjust = {}
            single_table_adjust = {}
            for k, v in subquery_dict.items():
                if table_alias in k:
                    subquery_adjust[k] = int(v * factor)
                else:
                    subquery_adjust[k] = v
            for k, v in single_table_dict.items():
                if k == table_alias:
                    single_table_adjust[k] = int(v * factor)
                else:
                    single_table_adjust[k] = v
            return subquery_adjust, single_table_adjust

        # 生成一个factor_list，考虑等差生成/等比生成两种思路
        # factor_list = [1/16, 1/8, 1/4, 1/2, 2, 4, 8, 16]    # 手动先写几个
        
        if mode == "global":
            # 根据column的值进行调优
            value_list, min_value, max_value = self.instance.get_columns_info(schema=object_table, column_list=column_list)
            print("cardinalities_tuning: value_list = {}. min_value = {}. max_value = {}.".format(value_list, min_value, max_value))
            curr_card = self.instance.get_single_table_card(schema=object_table)
            tuning_list = np.linspace(min_value, max_value, tuning_num, endpoint=True)
            factor_list = [i / curr_card for i in tuning_list]
            print("cardinalities_tuning: factor_list = {}.".format(factor_list))
            
        elif mode == "local":
            pass


        # 生成一批subquery_adjust和single_table_adject
        for factor in factor_list:
            predicate_candidates.append(adjust_by_factor(factor))

        if return_card == True:
            return predicate_candidates, tuning_list
        else:
            return predicate_candidates


    def multi_plans_evaluation_under_multi_table(self, table_num = 1, column_num = 1, \
        meta_num = 3, adjust_num = 3, split_budget = 100):
        """
        考虑多个table下对于plans的评估
    
        Args:
            table_num:
            column_num:
            meta_num:
            adjust_num:
            split_budget:
        Returns:
            meta_dict:
            query_dict:
            result_dict:
        """
        instance = self.instance
        table_list = instance.fetch_candidate_tables()
        # 考虑所有表的情况
        if table_num == "all":  
            table_num = len(table_list)
        else:
            table_num = min(table_num, len(table_list))
            
        table_selected = table_list[:table_num]
        meta_dict, query_dict, result_dict = {}, {}, {}
        
        for table in table_selected:
            meta_local, query_local, result_local = instance.multi_plans_evaluation_under_multi_meta(\
                new_table = table, column_num = column_num, meta_num = meta_num, \
                adjust_num = adjust_num, split_budget=split_budget)
            meta_dict[table] = meta_local
            query_dict[table] = query_local
            result_dict[table] = result_local

        return meta_dict, query_dict, result_dict


    def multi_plans_tuning_under_multi_meta(self, action_table, target_table, column_num = 1, \
        meta_num = 3, adjust_num = 3, split_budget = 100):
        """
        针对多个查询计划的调优
    
        Args:
            action_table: 动作添加的表
            target_table: 目标的表
        Returns:
            return1:
            return2:
        """
        instance = self.instance

        # 生成column的相关值
        column_list = instance.predicate_selection(table_name = action_table, \
            column_num = column_num)
        
        # print("column_list = {}.".format(column_list))
        bins_dict, reverse_dict = instance.construct_bins_dict(column_list = column_list, \
            split_budget = split_budget)

        # 一批随机的column值
        column_values_batch = instance.random_on_columns_batch(bins_dict = bins_dict, num = meta_num)
        curr_query_meta = instance.query_meta

        # 生成一批meta信息
        meta_list = [instance.add_new_table_meta(curr_query_meta, action_table, column_range_dict = \
            column_values) for column_values in column_values_batch]

        # 这里还是需要把当前的query_text信息也放进去，因为foreign_mapping也是不断变化的
        # 而query_text会保留下这部分的信息
        query_list = [query_construction.construct_origin_query(meta_info, workload = instance.workload) \
            for meta_info in meta_list]

        # print("meta_list = {}.".format(meta_list))
        result_batch = []
        for meta_info in meta_list:
            curr_res = self.multi_plans_tuning_under_meta(query_meta \
                = meta_info, extend_table = action_table, target_table = target_table, adjust_num=adjust_num)
            result_batch.append(curr_res)

        # return zip(meta_list, result_batch)
        return meta_list, query_list, result_batch
    

    @utils.timing_decorator
    def multi_plans_evaluation_under_multi_meta(self, new_table, column_num = 1, \
        meta_num = 3, split_budget = 100):
        """
        从属于同一个table下的多个meta信息
        
        Args:
            new_table:
            column_num:
            meta_num:
            adjust_num:
            split_budget:
        Returns:
            meta_list:
            query_list:
            result_batch:
        """
        instance = self.instance

        # 生成column的相关值
        column_list = instance.predicate_selection(table_name = new_table, \
            column_num = column_num)
        
        bins_dict, reverse_dict = instance.construct_bins_dict(column_list = column_list, \
            split_budget = split_budget)

        # 一批随机的column值
        column_values_batch = instance.random_on_columns_batch(bins_dict = bins_dict, num = meta_num)
        curr_query_meta = instance.query_meta

        # 生成一批meta信息
        meta_list = [instance.add_new_table_meta(curr_query_meta, new_table, column_range_dict = \
            column_values) for column_values in column_values_batch]

        # 这里还是需要把当前的query_text信息也放进去，因为foreign_mapping也是不断变化的
        # 而query_text会保留下这部分的信息
        query_list = [query_construction.construct_origin_query(meta_info, workload = instance.workload) \
            for meta_info in meta_list]

        # print("meta_list = {}.".format(meta_list))
        result_batch = []
        for meta_info in meta_list:
            curr_res = self.multi_plans_evaluation_under_meta(query_meta \
                = meta_info, new_table = new_table)
            result_batch.append(curr_res)

        # return zip(meta_list, result_batch)
        return meta_list, query_list, result_batch


    def multi_plans_tuning_under_meta(self, query_meta, extend_table, \
            target_table, adjust_num = 10, mode = "under-estimation"):
        """
        选择best cardinality
    
        Args:
            query_meta: 查询元信息
            extend_table: 选择拓展的table
            target_table: 目标调优的table
            adjust_num: 调整的数目
        Returns:
            current_range:
            best_range:
        """
        print("multi_plans_tuning_under_meta: query_meta = {}.".format(query_meta))
        print("multi_plans_tuning_under_meta: extend_table = {}. target_table = {}.".format(extend_table, target_table))
        instance = self.instance
        # 设置查询实例
        query_text = query_construction.construct_origin_query(\
            query_meta, workload = instance.workload)
        # print("plan_evaluation: query_text = {}.".format(query_text))
        instance.query_ctrl.set_query_instance(query_text = query_text, query_meta = query_meta)

        # 获得估计的基数
        subquery_dict, single_table_dict = instance.get_query_cardinalities(query_meta = query_meta)    

        # 评价物理计划
        flag, plan_target, plan_actual = instance.plan_verification(query_meta=query_meta, \
            new_table=target_table, subquery_dict=subquery_dict, single_table_dict=single_table_dict)
        origin_result = flag, plan_target, plan_actual

        # 调整new_table基数，按比例调整，暂时先不考虑
        adjust_card_list, card_list = self.cardinalities_tuning(object_table=target_table, \
            subquery_dict=subquery_dict, single_table_dict=single_table_dict, tuning_num=adjust_num, \
            column_list=instance.get_columns_from_schema(schema=target_table), column_values=[], return_card=True)

        print("multi_plans_tuning_under_meta: card_list = {}.".format(card_list))
        best_result, best_idx = None, -1
        
        for idx, (subquery_local, single_table_local) in enumerate(adjust_card_list):

            flag, plan_target, plan_actual, indicator_pair = self.instance.extend_estimation(query_meta = query_meta, \
                    new_table = extend_table, subquery_dict = subquery_local, single_table_dict = single_table_local)
            
            print("multi_plans_tuning_under_meta: idx = {}. flag = {}. indicator_pair = {}.".format(idx, flag, indicator_pair))
            if flag == True:
                cost1, cost2 = indicator_pair
                if best_result is None or best_result < cost2 - cost1:
                    best_result = cost2 - cost1
                    best_idx = idx
            elif flag == False:
                cost1, cost2 = indicator_pair
                if best_result is None or best_result < cost2 - cost1:
                    best_result = cost2 - cost1
                    best_idx = idx
        # 
        return best_result, card_list[best_idx]


    @utils.timing_decorator
    def multi_plans_evaluation_under_meta(self, query_meta, new_table):
        """
        考虑当前query_meta下多个查询计划的评估
    
        Args:
            query_meta:
            adjust_num:
        Returns:
            origin_result:
            adjust_result_list:
        """
        instance = self.instance
        # 设置查询实例
        query_text = query_construction.construct_origin_query(\
            query_meta, workload = instance.workload)
        # print("plan_evaluation: query_text = {}.".format(query_text))
        instance.query_ctrl.set_query_instance(query_text = query_text, query_meta = query_meta)

        # 获得估计的基数
        subquery_dict, single_table_dict = instance.get_query_cardinalities(query_meta = query_meta)    
        
        # 评价物理计划
        flag, plan_target, plan_actual = instance.plan_verification(query_meta=query_meta, \
            new_table=new_table, subquery_dict=subquery_dict, single_table_dict=single_table_dict)
        origin_result = flag, plan_target, plan_actual

        return origin_result, None


    
    def get_all_available_actions(self, table_subset):
        """
        获得所有的合法动作
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        # 获取所有当前可以join的表
        table_list = self.instance.fetch_candidate_tables(table_subset=table_subset)    
        return table_list
    
    @utils.timing_decorator
    def init_all_actions(self, table_subset, forward_steps = 1, mode = "under-estimation"):
        """
        初始化所有的动作，并且计算获得预期的收益
        
        Args:
            table_subset: 所有数据表的子集
            forward_steps: 向前拓展的步骤
        Returns:
            action_list: 动作列表 
            value_list: 对应的收益列表
        """
        instance = self.instance
        action_list, value_list = [], []
        table_list = instance.fetch_candidate_tables(table_subset=table_subset)    # 获取所有当前可以join的表
        # print("init_all_actions: existing_tables = {}. table_list = {}. table_subset = {}."\
        #       .format(instance.query_meta[0], table_list, table_subset))

        # 结果保存字典
        result_dict = {}
        for table in table_list:
            # 同一table的情况下评测多个meta信息，随机生成5个meta
            meta_list, query_list, result_batch = self.multi_plans_evaluation_under_multi_meta(\
                new_table=table, meta_num=10, split_budget=self.split_budget)
            # print("init_all_actions: table = {}.".format(table))
            # print("meta_list = {}.".format(meta_list))
            # print("query_list = {}.".format(query_list))
            # result_dict[table] = meta_list, query_list, result_batch
            # value_list = []
            benefit, candidate_list = self.result_filter(meta_list, query_list, result_batch, mode=mode)
            result_dict[table] = candidate_list

            # 添加benefit和action，如果结果不优直接暂时把action禁用了
            if benefit > 1e-2:
                value_list.append(benefit)
                action_list.append(table)

        # raise ValueError("init_all_actions test")
        self.action_result_dict = result_dict

        return action_list, value_list
    
    @utils.timing_decorator
    def result_filter(self, query_list, meta_list, result_batch, card_dict_list, \
                      candidate_num = 3, mode = "under-estimation"):
        """
        针对结果的过滤
    
        Args:
            query_list:
            meta_list:
            result_batch:
            candidate_num: 
        Returns:
            benefit: 预估收益
            candidate_list: 候选列表
            card_dict_candidate:
        """
        assert mode in ("under-estimation", "over-estimation")
        assert len(query_list) == len(meta_list) == len(result_batch)

        if len(query_list) == 0:
            # 如果没有结果的话，直接返回空集
            return 0.0, [], []

        benefit, candidate_list, card_dict_candidate = 0, [], []

        debug = False
        for query, meta, result, card_dict in zip(query_list, meta_list, result_batch, card_dict_list):
            if debug:
                # 用于debug
                plan1: physical_plan_info.PhysicalPlan = result[0][1] 
                plan2: physical_plan_info.PhysicalPlan = result[0][2]
                
                # 打印查询计划以进行比较
                plan1.show_plan(), plan2.show_plan()

                # 只打印join_order
                plan1.show_join_order(), plan2.show_join_order()

            if mode == "under-estimation":
                if result[0][0] == True:
                    # 如果查询计划符合预期，加入到结果集中
                    candidate_list.append((query, meta))
                    card_dict_candidate.append(card_dict)
            else:
                # 只有join order不相同的时候才会被加入到结果
                if result[0][0] == False:
                    candidate_list.append((query, meta))
                    card_dict_candidate.append(card_dict)
            # print("result = {}.".format(result))

        if len(candidate_list) >= candidate_num:
            # 结果太多了就进行截断
            benefit = 1.0
            candidate_list = candidate_list[:candidate_num]
        else:
            # 结果小的话随机添加结果
            benefit = len(candidate_list) / candidate_num   # 自适应调整benefit

            # 这一步感觉不理解。。。
            # for query, meta, result in zip(query_list, meta_list, result_batch):
            #     if mode == "over-estimation":
            #         if result[0][0] == False:
            #             candidate_list.append((query, meta))
            #     else:
            #         if result[0][0] == True:
            #             candidate_list.append((query, meta))

            #     if len(candidate_list) == candidate_num:
            #         break
        
        return benefit, candidate_list, card_dict_candidate
    

    def exploit(self, table_name, config = {}):
        """
        选取若干个表进行拓展，最后选一个收益最大的meta信息创建新的query_instance
        
        Args:
            table_name: 表的名字
            config: 探索的配置
        Returns:
            new_query_instance:
            benefit:
            actual_cost: 
            expected_cost: 
            actual_card: 
            expected_card:
        """
        # instance = self.instance
        candidate_list = self.action_result_dict[table_name]
        new_query_instance = None

        extension_list = []
        clip_factor = config.get('clip_factor', 10.0)     # 截断变量
        # alpha = config['alpha']
        alpha = config.get('alpha', 0.5)
        # 
        time_limit = config.get('timeout', -1)

        def benefit_calculate(cost_true, cost_estimation, card_true, card_estimation):
            """
            计算节点拓展的收益情况
            
            Args:
                arg1:
                arg2:
            Returns:
                res1:
                res2:
            """
            benefit1, benefit2 = 0.0, 0.0   # [0, 1]之间的值
            # 计算plan的增益
            if cost_true * clip_factor <= cost_estimation:
                benefit1 = 1.0
            else:
                benefit1 = (cost_estimation - cost_true) / (cost_true * (clip_factor - 1))

            # 计算cardinality带来的增益
            if config['mode'] == "over-estimation":
                if card_estimation <= card_true:
                    benefit2 = 0.0
                elif card_true * clip_factor <= card_estimation:
                    benefit2 = 1.0
                else:
                    benefit2 = (card_estimation - card_true) / (cost_true * (clip_factor - 1))

            elif config['mode'] == "under-estimation":
                if card_estimation >= card_true:
                    benefit2 = 0.0
                else:
                    benefit2 = (card_true - card_estimation) / card_true

            return alpha * benefit1 + (1 - alpha) * benefit2

        for query_meta, query_text in candidate_list:
            # 处理配一个候选的配置，记录最优的结果
            # print("InstanceAnalyzer.exploit: query_meta = {}. query_text = {}.".format(query_meta, query_text))
            local_extension = self.create_extension_instance(ext_query_meta=query_meta, ext_query_text=query_text)
            flag, cost1, cost2 = local_extension.true_card_plan_verification(time_limit=time_limit)     # P-Error相关素材，cost1 <= cost2

            # 针对flag的处理，考虑超时的情况
            if flag == False:
                # 两个查询计划不等的情况
                top_query_estimation, top_query_true = local_extension.top_query_estimation()               # Q-Error相关素材
                benefit = benefit_calculate(cost1, cost2, top_query_true, top_query_estimation)
            elif flag == True:
                # 两个查询计划相等的情况
                if cost1 == -1 and cost2 == -1:
                    top_query_estimation, top_query_true = 1, 1     # 伪造一个基数
                    benefit = 0.0
                else:
                    top_query_estimation, top_query_true = local_extension.top_query_estimation()               # Q-Error相关素材
                    benefit = benefit_calculate(cost1, cost2, top_query_true, top_query_estimation)

            # 在原有的基础上添加额外的内容
            # extension_list.append((local_extension, benefit))
            extension_list.append((local_extension, benefit, cost1, cost2, top_query_true, top_query_estimation))

        extension_list.sort(key=lambda a:a[1], reverse=True)
        selected_extension:node_extension.ExtensionInstance = extension_list[0][0]
        benefit = extension_list[0][1]      # 收益情况
        actual_cost, expected_cost = extension_list[0][3], extension_list[0][2]
        actual_card, expected_card = extension_list[0][5], extension_list[0][4]
        # 修改入参
        new_query_instance = selected_extension.construct_query_instance(query_instance_ref=self.instance)  
        
        return new_query_instance, benefit, actual_cost, expected_cost, actual_card, expected_card




# %%

def external_func_wrapper(workload):
    def external_join_order_func(query_list, meta_list):
        """
        处理一批查询以获得结果
        
        Args:
            arg1:
            arg2:
        Returns:
            result_list:
            res2:
        """

        result_list = []
        jo_str_list = node_query.get_query_join_order_batch(workload=workload, meta_list=meta_list)
        for jo_str in jo_str_list:
            jo_analyzer = JoinOrderAnalyzer(join_order_str=jo_str)
            flag = jo_analyzer.is_bushy()
            level_dict = jo_analyzer.level_dict
            result_list.append((flag, level_dict))
        
        return result_list
    return external_join_order_func
    
