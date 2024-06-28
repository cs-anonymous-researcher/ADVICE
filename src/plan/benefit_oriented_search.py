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
from plan import stateful_search
from typing import Any

# %%
from estimation import estimation_interface
from estimation.state_estimation import StateManager
from plan import benefit_oriented_analysis, node_extension, node_query
from utility import common_config, utils

# %%

class BenefitOrientedTree(stateful_search.StatefulTree):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, workload: Any, external_info: Any, max_step: Any, template_id: int = -1, mode: Any | None = None, 
        init_strategy: str = "random", state_manager_ref: StateManager = None, exploration_info: Any = {}):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        super().__init__(workload, external_info, max_step, template_id, 
            mode, init_strategy, state_manager_ref, exploration_info)

    def fetch_all_node_info(self,):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            node_info_list:
            return2:
        """
        def map_func(node: BenefitOrientedNode):
            info_dict = {
                "state": node.node_state,
                "benefit": node.benefit
            }
            return info_dict

        def reduce_func(result_list):
            return result_list

        node_info_list = self.traversal(map_func, reduce_func)
        return node_info_list


    def calculate_exploration_benefit(self,):
        """
        计算当前树下探索新节点带来的收益

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        node_info_list = self.fetch_all_node_info()
        value_list = [item['benefit'] for item in node_info_list if item['state'] == "candidate"]

        if len(value_list) >= 1:
            return np.max(value_list)
        else:
            return -1.0

    def calculate_historical_benefit(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        node_info_list = self.fetch_all_node_info()
        # 选取所有结束状态的节点
        value_list = [item['benefit'] for item in node_info_list if item['state'] == "finish"]

        if len(value_list) >= 1:
            return np.average(value_list)
        else:
            return 0.0
        
    def select_candidate_node(self, mode = "benefit"):
        """
        {Description}
    
        Args:
            mode:
            arg2:
        Returns:
            selected_node:
            return2:
        """
        def map_func(node: BenefitOrientedNode):
            return node.node_state, node.benefit, node
        
        def reduce_func(result_list):
            return [item for item in result_list if item[0] == "candidate"]
        
        node_list = self.traversal(map_func, reduce_func)

        assert len(node_list) > 0, f"select_candidate_node: len(node_list) = {len(node_list)}. "\
            f"root_id = {self.root_id}. tree_state = {self.get_tree_state()}."
        
        if mode == "benefit":
            # 根据benefit的大小去选
            return max(node_list, key=lambda a: a[1])[2]
        elif mode == "random":
            # 随机选
            selected_idx = np.random.randint(len(node_list))
            return node_list[selected_idx][2]


    def get_tree_state(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        node_info_list = self.fetch_all_node_info()
        state_list = [item['state'] for item in node_info_list]

        # node的相关状态：candidate、calculating、finish、terminate
        candidate_cnt, finish_cnt, calculating_cnt, terminate_cnt = 0, 0, 0, 0

        for s in state_list:
            if s == "candidate":
                candidate_cnt += 1
            elif s == "finish":
                finish_cnt += 1
            elif s == "calculating" or s == "reserved":
                # 可以认为reserved状态和calculating等价
                calculating_cnt += 1
            elif s == "terminate":
                terminate_cnt += 1
            else:
                raise ValueError(f"get_tree_state: invalid state = {s}")

        # tree的相关状态：available、blocked、archived、failed
        if len(state_list) == 0:
            return "failed"
        elif candidate_cnt == 0:
            if calculating_cnt > 0:
                return "blocked"
            else:
                return "archived"
        else:
            return "available"
    
    def update_all_node_ids(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        def map_func(node: BenefitOrientedNode):
            # 更新node_id
            if node.node_id == -1:
                node.node_id = self.get_next_node_id()
                self.node_dict[node.signature] = node
            # 更新root_id
            if node.root_id == -1:
                node.root_id = self.root_id
            return node.node_id
        
        def reduce_func(result_list):
            return result_list
        
        result_list = self.traversal(map_func, reduce_func)
        assert -1 not in result_list
        return True
    
    def enumerate_all_actions(self,):
        """
        把所有的candidate node返回
    
        Args:
            arg1:
            arg2:
        Returns:
            result_list: 包含(estimate_benefit, new_node)的list
            return2:
        """
        # 构造
        def map_func(node: BenefitOrientedNode):
            return node.node_state, node.benefit, node

        def reduce_func(result_list):
            return [item[1:] for item in result_list if item[0] == "candidate"]

        result_list = self.traversal(map_func, reduce_func)
        return result_list

    def has_explored_all_nodes(self,):
        """
        根据tree_state判断是否是所有状态都探索完了
    
        Args:
            arg1:
            arg2:
        Returns:
            flag: True表示探索完了，False表示没探索完
            return2:
        """
        tree_state = self.get_tree_state()
        return tree_state == "archived" or tree_state == "failed"

    @utils.timing_decorator
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
        print(f"BenefirOrientedTree.create_root: creation start. template_id = {self.template_id}. root_id = {self.root_id}.")
        node_id = self.get_next_node_id()
        root_schema = tuple(sorted(self.external_info['query_instance'].query_meta[0]))

        root_node = BenefitOrientedNode(self.external_info, self.tree_id, node_id, self.root_id, 
            self.template_id, self.mode, None, self.init_strategy, schema_order=root_schema, 
            exploration_info=self.exploration_info, state_manager_ref=self.state_manager)
        root_node.set_node_state("finish")

        flag, candidate_num = root_node.initialize_actions()
        print(f"BenefirOrientedTree.create_root: creation end. template_id = {self.template_id}. root_id = {self.root_id}.")
        # 

        if self.state_manager is not None:
            analyze_case_list = root_node.get_analyze_cases()
            if self.mode == "under-estimation":
                # 只有under-estimation才能添加结果
                self.state_manager.add_instance_list(analyze_case_list)     # 添加尝试拓展的实例

            # root_case = root_node.get_current_case()
            # self.state_manager.add_new_instance(root_case)

        root_node.is_complete = True    # 
        root_node.selectable = flag            
        self.is_terminate = flag

        return root_node

# %%
available_node_state_list = ["candidate", "reserved", "calculating", "finish", "terminate"]

class BenefitOrientedNode(stateful_search.StatefulNode):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, info_dict: Any, tree_id: Any, node_id: int = -1, root_id: int = -1, template_id: int = -1, 
            mode: Any | None = None, extend_table: Any | None = None, init_strategy: str = "random", action_config: Any = {}, 
            schema_order: Any = (), exploration_info: Any = {}, state_manager_ref: StateManager = None):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        super().__init__(info_dict, tree_id, node_id, root_id, template_id, mode, extend_table, 
            init_strategy, action_config, schema_order, exploration_info, state_manager_ref)
        # 
        self.query_analyzer = benefit_oriented_analysis.BenefitOrientedAnalyzer(self.query_instance, 
            mode, exploration_info, common_config.get_split_budget(), state_manager_ref.state_dict.keys())
        self.node_state = "candidate"


    def set_node_state(self, new_state):
        """
        {Description}
    
        Args:
            new_state:
            arg2:
        Returns:
            return1:
            return2:
        """
        assert new_state in available_node_state_list, \
            f"set_node_state: invalid new_state = {new_state}."
        self.node_state = new_state

    def initialize_actions(self,):
        """
        初始化所有action，等价于candidate enumeration

        Args:
            arg1:
            arg2:
        Returns:
            flag: 表示
            candidate_num: 生成candidate node的数目
        """
        # assert self.node_state == "finish", f"BenefitOrientedNode.initialize_actions: node_state = {self.node_state}."
        # 为了兼容之前的代码，做出必要的妥协
        if self.node_state != "finish":
            return "fail", 0

        self.query_analyzer.set_node_info(self.template_id, self.root_id, self.node_id)
        # 直接判断状态已中止的情况
        if self.is_terminate == True:
            return "terminate", -1
        
        node_info_list = self.query_analyzer.init_all_actions(self.selected_tables, self.init_strategy)

        # 不存在新node的情况
        if len(node_info_list) == 0:
            self.is_terminate = True
            return "terminate", -1
        
        # candidate enumeration
        for table_name, benefit, query_instance, extension_instance in node_info_list:
            # 验证query_instance和extension_instance的类型
            assert isinstance(query_instance, node_query.QueryInstance), \
                f"BenefitOrientedNode.initialize_actions: query_instance = {query_instance}."
            assert isinstance(extension_instance, node_extension.ExtensionInstance), \
                f"BenefitOrientedNode.initialize_actions: extension_instance = {extension_instance}."
            # take_action的相关逻辑
            info_dict = {
                "query_instance": query_instance,           #
                "selected_tables": self.selected_tables,    # 
                "benefit": benefit,                         # 修改benefit的来源
                "max_depth": self.max_depth,                # 最大深度
                "timeout": self.timeout            
            }

            try:
                curr_path_list = self.exploration_info['path_list']
                curr_path_dict = utils.prefix_aggregation(self.schema_order, curr_path_list)
                new_path_list = curr_path_dict[table_name]

                # 新的探索信息
                new_exploration_info = {
                    "path_list": new_path_list,
                    # "state_manager": self.state_manager   # 转移到__init___中导入结果
                }

                if "ref_index_dict" in self.exploration_info.keys():
                    new_exploration_info["ref_index_dict"] = {k: self.exploration_info['ref_index_dict'][k] for k in new_path_list}
                else:
                    print(f"BenefitOrientedNode.initilize_actions: exploration_info = {self.exploration_info.keys()}")
            except KeyError as e:
                new_exploration_info = {}

            new_order = self.schema_order + (table_name,)
            # 由于Node无法访问到Tree，因此node_id一开始是不知道的，但是问题不大，因此node_id只要用来打印信息，
            # 不涉及重要功能。额外在Tree类上实现了一个update_all_node_ids的方法
            node_id = -1

            new_node = BenefitOrientedNode(info_dict, self.tree_id, node_id, self.root_id, 
                self.template_id, self.mode, table_name, self.init_strategy, self.action_config, 
                new_order, new_exploration_info, self.state_manager)
            
            # new_node.set_benefit_properties()   # 设置收益的属性
            new_node.extension_ref = extension_instance

            # expand_node的相关逻辑
            new_node.parent = self
            self.add_child(new_node, table_name)

        # 表示初始化成功
        return "success", len(node_info_list)

    def launch_calculation(self, proc_num, timeout, with_card_dict):
        """
        启动P-Error计算

        Args:
            proc_num:
            timeout:
            with_card_dict:
        Returns:
            benefit: 
            subquery_res:
            single_table_res: 
        """
        # 
        assert self.node_state in ("candidate", "reserved"), f"launch_calculation: node_id = {self.node_id}. "\
            f"root_id = {self.root_id}. template_id = {self.template_id}. node_state = {self.node_state}."
        assert self.extension_ref is not None, "launch_calculation: extension_ref is None. "\
            f"node_id = {self.node_id}. root_id = {self.root_id}. template_id = {self.template_id}. "\
            f"extend_table = {self.extend_table}."
        expected_cost, actual_cost = estimation_interface.estimate_plan_benefit(
            self.extension_ref, self.mode, self.extend_table, card_est_spec = "graph_corr_based", 
            plan_sample_num=common_config.benefit_config['plan_sample_num'], restrict_order = False)
        print(f"launch_calculation: expected_cost = {expected_cost:.2f}. actual_cost = {actual_cost:.2f}.")
        self.set_benefit_properties(actual_cost, expected_cost)

        #
        subquery_res, single_table_res = self.extension_ref.\
            true_card_plan_async_under_constaint(proc_num, timeout, with_card_dict)

        assert len(subquery_res) > 0 and len(single_table_res) > 0, \
            f"launch_calculation: subquery_res = {subquery_res}. single_table_res = {single_table_res}."
        
        # return (expected_cost, actual_cost), self.benefit, subquery_res, single_table_res
        # 设置node状态
        self.set_node_state("calculating")
        return self.benefit, subquery_res, single_table_res

    def update_node_state(self, node_signature, external_info):
        """
        在父类的函数上添加额外的操作
    
        Args:
            node_signature:
            external_info:
        Returns:
            flag:
            curr_benefit:
            cost_true:
            cost_estimation:
        """
        flag, curr_benefit, cost_true, cost_estimation = \
            super().update_node_state(node_signature, external_info)

        # 额外维护一下node_state
        if flag == True:
            self.set_node_state("finish")
            _, candidate_num = self.initialize_actions()

        return flag, curr_benefit, cost_true, cost_estimation

    def terminate_calculation(self,):
        """
        中止P-Error计算
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        self.set_node_state("terminate")


# %%
