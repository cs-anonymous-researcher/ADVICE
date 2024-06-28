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
import graphviz
from collections import defaultdict

from utility import query_sample, generator, utils, common_config
from plan import node_query, plan_analysis

# %%

lambda_ucb = math.sqrt(2) # 计算UCB的时候用到的lambda值
inf = 10e8  # 定义无穷大


class ExplorationNode(object):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, info_dict):
        """
        {Description}

        Args:
            info_dict: 节点信息的字典，需要包含以下的内容
            {
                "query_instance": 查询的实例
                ""
            }
            arg2:
        """
        self.id = "undefined"
        self.parent = None

        # 查询的实例
        self.query_instance: node_query.QueryInstance = info_dict['query_instance']
        # 针对该实例的分析器
        self.query_analyzer = plan_analysis.InstanceAnalyzer(query_instance=\
            self.query_instance, split_budget=common_config.get_split_budget())

        self.selected_tables = info_dict['selected_tables']
        
        self.children_dict = {}           # 子节点的字典
        self.info_dict = info_dict        # 当前状态
        self.benefit = info_dict.get("benefit", 0.0)    # 节点收益情况

        self.visit_number = 0             # 节点的已访问次数
        self.exploit_number = 0           # 该节点执行BO的次数
        
        self.action_set = set()
        self.init_value_dict = {}

        max_depth = info_dict.get("max_depth", 1000)
        self.max_depth = max_depth
        self.timeout = info_dict.get("timeout", -1)

        schema_list:list = self.query_instance.query_meta[0]
        
        if len(schema_list) >= max_depth or len(schema_list) >= len(self.selected_tables):
            self.is_terminate = True
        else:
            self.is_terminate = False       # 该节点是否已经探索完毕

        self.is_action_init = False         # action是否有被初始化

    def set_benefit_properties(self, actual_cost, expected_cost, actual_card, expected_card):
        """
        设置节点相关的属性
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        self.actual_cost = actual_cost
        self.expected_cost = expected_cost
        self.actual_card = actual_card
        self.expected_card = expected_card


    # def get_properties(self,):
    #     """
    #     获得节点相关的属性
        
    #     Args:
    #         arg1:
    #         arg2:
    #     Returns:
    #         res1:
    #         res2:
    #     """
    #     try:
    #         return self.actual_cost, self.expected_cost, self.actual_card, self.expected_card
    #     except AttributeError:
    #         # 有些属性不具备，直接返回None
    #         return None

    def root_tuning(self, table_subset):
        """
        针对根节点的调优
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        action_list = self.query_analyzer.get_all_available_actions(table_subset=self.selected_tables)
        print("root_tuning: action_list = {}.".format(action_list))
        
        result_dict = {}
        for action in action_list:
            result_local = self.query_analyzer.root_tuning(\
                table_subset=table_subset, action=action)
            result_dict[action] = result_local

        return result_dict


    def query_repr(self,):
        """
        查询的表达式，只考虑schema的部分，忽略filter的部分
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        return self.query_instance.query_meta[0]

    @utils.timing_decorator
    def initialize_actions(self, ):
        """
        初始化所有的动作
        
        Args:
            None
        Returns:
            action: 初始的动作
            value: 动作对应的价值
        """
        # 直接判断状态已中止的情况
        if self.is_terminate == True:
            return "terminate", -1
        
        action_list, value_list = self.query_analyzer.\
            init_all_actions(table_subset=self.selected_tables)
        
        # 不存在action的情况
        if len(action_list) == 0:
            self.is_terminate = True
            return "terminate", -1
        
        # 存在action的情况
        pair_list = list(zip(action_list, value_list))
        pair_list.sort(key=lambda a: a[1], reverse=True)
        # self.pair_list = pair_list

        for action, value in pair_list:
            self.init_value_dict[action] = value 

        self.action_set = set(action_list)
        print("initialize_actions: action_list = {}. value_list = {}. init_value_dict = {}".\
              format(action_list, value_list, self.init_value_dict))
        
        self.is_action_init = True
        return pair_list[0]

    def expand_actions(self, ):
        """
        TODO: 可能需要增加更多的动作
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """


    def take_action(self, action):
        """
        执行一个动作

        Args:
            action: 采取的动作
            arg2:
        Returns:
            new_node: 一个新的节点
            return2:
        """
        print("take_action: action = {}.".format(action))
        # new_query_instance, benefit = self.query_instance.exploit(table_name=action)
        # 返回额外的内容
        new_query_instance, benefit, actual_cost, expected_cost, actual_card, expected_card = \
            self.query_analyzer.exploit(table_name=action, config={
            "mode": "under-estimation",
            "timeout": self.timeout
        })


        info_dict = {
            "query_instance": new_query_instance,       #
            "selected_tables": self.selected_tables,    # 
            # "benefit": self.init_value_dict[action],    # 初始收益 
            "benefit": benefit,                         # 修改benefit的来源
            "max_depth": self.max_depth,                 # 最大深度
            "timeout": self.timeout
        }

        new_node = ExplorationNode(info_dict = info_dict)

        # 设置节点的收益相关信息
        new_node.set_benefit_properties(actual_cost, expected_cost, actual_card, expected_card)
        return new_node
    

    def add_child(self, child_node, action):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        self.children_dict[action] = child_node


    def check_existence(self, action) -> bool:
        """
        判断节点是否在树上存在
    
        Args:
            action:
            arg2:
        Returns:
            flag:
            return2:
        """
        flag = action in self.children_dict.keys()
        return flag

    def UCB_score(self, action):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        if action not in self.children_dict.keys():
            # 没有子节点被探索过
            # return inf
            return self.init_value_dict[action]
        else:
            # 直接访问子节点获取信息
            benefit, visit_number = self.get_child_features(action=action)

            # Q_s_a = self.visit_child_by_action(action).benefit / self.visit_child_by_action(action).visit_number    # 需要做归一化
            # N_s = self.visit_number
            # n_s_a = self.visit_child_by_action(action).visit_number
            Q_s_a = benefit / visit_number    # 需要做归一化
            N_s = self.visit_number
            n_s_a = visit_number
            
            return Q_s_a + \
                lambda_ucb * np.sqrt(np.log(N_s) / n_s_a)
        
        
    def highest_UCB_action(self, action_list = None):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        # print("选择UCB最高的Action:")
        best_score, best_action = -1.0, None

        if action_list is None:
            # action_list = self.children_dict.keys()
            action_list = list(self.action_set)


        for action in action_list:
            current_ucb = self.UCB_score(action)
            # print("action = {}. score = {}.".format(action, current_ucb))
            if current_ucb > best_score:
                best_score = current_ucb
                best_action = action

        # print("highest_UCB_action: action_list = {}. best_score = {:.2f}. best_action = {}".\
        #       format(action_list, best_score, best_action))
        
        if best_action is None:
            raise ValueError("best_action is None")

        return best_action
    


    def visit_child_by_action(self, action):
        """
        通过action访问到对应的子节点
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        # if action in self.children_dict.keys():
        return self.children_dict[action]
    
    def get_child_features(self, action):
        """
        获得子节点的相关特征
    
        Args:
            action:
            arg2:
        Returns:
            benefit:
            visit_number:
        """
        benefit, visit_number = 0.0, 1 
        if action in self.children_dict.keys():
            # 节点的
            benefit = self.visit_child_by_action(action=action).benefit
            visit_number = self.visit_child_by_action(action).visit_number
        else:
            visit_number = 1
            benefit = self.init_value_dict[action]

        return benefit, visit_number

    def get_properties(self, config = ["query"]):
        """
        获得当前节点的一些性质，以dictionary的形式返回
        支持的config: ["query", "meta", "p_error", "q_error", "cardinalities"]
        
        Args:
            config:
            arg2:
        Returns:
            result:
            res2:
        """
        result = {}
        if "query" in config:
            # 获得查询文本
            result['query'] = self.query_instance.query_text
        if "meta" in config:
            # 获得查询
            result['meta'] = self.query_instance.query_meta
        if "p_error" in config:
            try:
                actual_cost, expected_cost, p_error = \
                    self.actual_cost, self.expected_cost, self.actual_cost / (2.0 + self.expected_cost)
                print("ExplorationNode.get_properties: actual_cost = {}. expected_cost = {}. p_error = {}.".\
                      format(actual_cost, expected_cost, p_error))
                result['p_error'] = p_error, actual_cost, expected_cost
            except AttributeError:
                # 没有设置属性的意外情况
                result['p_error'] = 1.0, 0.0, 0.0
        if "q_error" in config:
            try:
                actual_card, expected_card, q_error = self.actual_card, \
                    self.expected_card, max(self.actual_card/(1+self.expected_card), self.expected_card/(2.0 + self.actual_card))
                print("ExplorationNode.get_properties: actual_card = {}. expected_card = {}. q_error = {}.".\
                      format(actual_card, expected_card, q_error))
                
                result['q_error'] = q_error, actual_card, expected_card
            except AttributeError:
                #
                result['q_error'] = 1.0, 0.0, 0.0
        if "cardinalities" in config:
            # true_subquery, true_single_table = {}, {}
            # estimated_subquery, estimated_single_table = {}, {}
            true_subquery, true_single_table = \
                self.query_instance.true_card_dict, self.query_instance.true_single_table
            estimated_subquery, estimated_single_table = \
                self.query_instance.estimation_card_dict, self.query_instance.estimation_single_table
            
            result["cardinalities"] = (true_subquery, true_single_table, \
                                       estimated_subquery, estimated_single_table)

        return result 
# %%


class ExplorationTree(object):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, external_info: dict, create_root = True):
        """
        {Description}

        Args:
            external_info: 外部的信息
        """
        # print("external_info = {}.".format(external_info))
        self.is_terminate = False       # 表示搜索过程是否结束了
        self.external_info = external_info
        # self.external_info['max_depth'] = max_depth # 设置最大深度
        self.max_depth = external_info.get("max_depth", 1000)

        if create_root:
            self.root = self.create_root()  # 创建根节点
        else:
            self.root = None

    def create_root(self,):
        """
        创建根节点
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        # 创建根节点
        root_node = ExplorationNode(info_dict=self.external_info)
        # 
        action, value = root_node.initialize_actions()
        if action == "terminate":
            self.is_terminate = True

        return root_node
    
    def visualization(self, out_path = '../doctest-output/{}.gv'.format("test")):
        """
        对当前的结果进行可视化，观察探索过程是否符合预期
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        
        global_id = 0
        dot = graphviz.Digraph(comment="monte carlo tree")
        dot.attr("node", shape="box")

        def add_to_graph(node:ExplorationNode):
            nonlocal global_id
            global_id += 1
            dot.node(str(global_id), node.format_str())
            current_id = str(global_id)

            for k, v in node.children_dict.items():
                child_id = add_to_graph(v)
                dot.edge(current_id, child_id)
            return current_id

        add_to_graph(self.root)
        dot.render(out_path, view=False)
        return dot


    def select_action(self, node: ExplorationNode):
        """
        选择当前节点的动作，这里指添加的表

        Args:
            node:
        Returns:
            action:
        """
        # 记录action_count的调用次数，
        try:
            self.action_count += 1
        except AttributeError:
            self.action_count = 1   # 创建变量

        if node.is_terminate == True:
            # 如果发现是terminate的话，直接raise error
            print("ExplorationTree.select_action: self.action_count = {}.".format(self.action_count))
            # raise ValueError("ExplorationTree.select_action: node({}) is terminate.".\
            #                  format(node.query_repr()))
            Warning("ExplorationTree.select_action: node({}) is terminate.".\
                             format(node.query_repr()))
            return "error"
        
        # 判断action
        if node.is_action_init == False:
            # 初始化所有的动作，并且返回价值最高的动作
            action, _ = node.initialize_actions()
            # if action == "terminate":
        else:
            # 选择一个动作
            action = node.highest_UCB_action()

        return action


    def exploit_node(self,):
        """
        针对当前的node进行挖掘探索

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        pass
    
    def expand_node(self, node:ExplorationNode, action):
        """
        尝试在当前的节点上扩展新的节点，如果节点存在的话
    
        Args:
            node:
            action:
        Returns:
            return1:
            return2:
        """
        if node.check_existence(action) == False:
            new_node = node.take_action(action=action)
            new_node.parent = node
            node.add_child(new_node, action)
            return new_node, "leaf"     # 表示到达叶节点了
        else:
            next_node: ExplorationNode = node.visit_child_by_action(action)
            if next_node.is_terminate == False:
                return next_node, "nonleaf"
            else:
                return next_node, "leaf"
    

    def update(self, node: ExplorationNode, benefit):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        
        current_node = node
        while current_node is not None:
            current_node.visit_number += 1
            current_node.benefit += benefit
            current_node = current_node.parent

        return benefit

    def main_search_process(self, max_step, return_option = "best"):
        """
        主搜索过程
    
        Args:
            max_step: 最大搜索的步数
            return_option: 
        Returns:
            return1:
            return2:
        """
        
        current_step = 0

        while True:
            is_terminate = self.run_episode()  # 进行一次探索
            current_step += 1
            if is_terminate == True:
                print("整棵树已经探索完毕")
                break

            if current_step >= max_step:
                print("搜索到达最大的步数")
                break

        if return_option == "best":
            # 只返回最好的结果
            optimal_config = self.best_configuration()
            return optimal_config
        elif return_option == "all":
            # 返回所有节点，with_meta这个参数
            all_config = self.all_configuration(with_meta=False)
            return all_config
        elif return_option == "full":
            # 返回所有节点并附带完整的信息
            full_config = self.full_configuration()
            return full_config
        elif return_option == "none":
            # 不返回任何结果
            return None
        

    def run_episode(self,):
        """
        执行一次探索过程

        Args:
            None
        Returns:
            flag: 整一棵树的探索是否已经结束
            res2:
        """
        # print("执行sample_configuration")
        node, benefit = self.sample_configuration(self.root) # 从根节点开始执行一次采样训练

        if node is not None:
            # print("origin benefit = {}".format(benefit))
            # print("执行update")
            # print("transformed benefit = {}.".format(transformed_benefit))

            transformed_benefit = benefit                           # 获得收益
            self.update(node, transformed_benefit)                  # 更新节点信息
            return False
        else:
            return True
        

    def calculate_benefit(self, node: ExplorationNode):
        """
        计算节点的收益情况
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        return node.benefit


    def sample_configuration(self, node):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        def visit(node, curr_type):
            if curr_type == "leaf":
                # 到达叶节点
                benefit = self.calculate_benefit(node)
                return node, benefit
            elif curr_type == "nonleaf":
                action = self.select_action(node)
                if action == "error":
                    # 代表出现了预期之外的行为
                    return None, 0.0
                elif action != "terminate":
                    next_node, next_mode = self.expand_node(node, action)
                    return visit(next_node, next_mode)
                else:
                    # terminate action代表这也是一个叶节点
                    return node, self.calculate_benefit(node)
            return node
        
        return visit(node, "nonleaf")
        
    def best_configuration(self,):
        """
        获得当前最优的配置
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        def map_func(node:ExplorationNode):
            return node, node.benefit

        def reduce_func(result_list:list):
            result_list.sort(key = lambda a: a[1], reverse = True)
            return result_list[0]

        return self.traversal(map_func, reduce_func)
    
    def all_configuration(self, with_meta = False):
        """
        获得所有的结果
        
        Args:
            with_meta: 是否包含元信息
            arg2:
        Returns:
            res1:
            res2:
        """
        def map_func(node:ExplorationNode):
            # 获得节点的具体信息，这里需要在ExplorationNode添加新的方法
            result = node.get_properties(config=["query", "meta", "p_error", ])
            print("all_configuration: result = {}.".format(result))

            query_text = result["query"]
            query_meta = result["meta"]
            actual_cost, expected_cost, p_error = result["p_error"]
            if with_meta == False:
                return query_text, actual_cost, expected_cost, p_error
            else:
                return query_text, query_meta, actual_cost, expected_cost, p_error
            
        def reduce_func(output_list:list):
            # 返回所有结果(感觉需要把结果拆一下)
            query_list = [item[0] for item in output_list]
            if with_meta == False:
                result_list = [item[1:] for item in output_list]
                return query_list, result_list
            else:
                meta_list = [item[1] for item in output_list]
                result_list = [item[2:] for item in output_list]
                return query_list, meta_list, result_list

        return self.traversal(map_func=map_func, reduce_func=reduce_func)
    

    def full_configuration(self):
        """
        获得所有完整的结果，包括了query/meta/result/card_dict
        
        Args:
            None
        Returns:
            query:
            meta:
            result:
            card_dict
        """
        def map_func(node:ExplorationNode):
            # 获得节点的具体信息，这里需要在ExplorationNode添加新的方法
            result = node.get_properties(config=["query", "meta", "p_error", "cardinalities"])
            print("all_configuration: result = {}.".format(result))

            query_text = result["query"]
            query_meta = result["meta"]
            error_res = result["p_error"]
            card_dict = result["cardinalities"]

            return query_text, query_meta, error_res, card_dict
            
        def reduce_func(output_list:list):
            # 返回所有结果(感觉需要把结果拆一下)
            query_list = [item[0] for item in output_list]
            meta_list = [item[1] for item in output_list]
            result_list = [item[2] for item in output_list]
            card_dict_list = [item[3] for item in output_list]
            return query_list, meta_list, result_list, card_dict_list

        return self.traversal(map_func=map_func, reduce_func=reduce_func)


    def traversal(self, map_func, reduce_func):
        """
        树上的遍历操作
    
        Args:
            map_func: 节点处理函数
            reduce_func: 结果聚合函数
        Returns:
            return1:
            return2:
        """
        visit_list = []
        # cnt = 0
        def visit(node):
            # print(str(node.state))
            # nonlocal cnt
            # cnt += 1
            # if cnt > 5:
            #     return
            # print(len(node.children_dict))
            visit_list.append(map_func(node))
            for child in node.children_dict.values():
                visit(child)

        visit(self.root)
        return reduce_func(visit_list)


    def root_tuning(self,):
        """
        针对根节点的信息调优
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        table_subset = self.root.query_instance.query_meta[0]

        def map_func(node:ExplorationNode):
            print("call map_function")
            result = node.root_tuning(table_subset=table_subset)
            return result
        
        def reduce_func(result_list:list):
            print("call reduce_function")
            print("result_list = {}.".format(result_list))

            return result_list

        result = self.traversal(map_func=map_func, reduce_func=reduce_func)
        return result
    