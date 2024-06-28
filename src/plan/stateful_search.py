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
from collections import defaultdict

# %%
from plan import advance_search, stateful_analysis, node_query
from typing import Any
from utility import utils
from utility.utils import trace
from estimation import state_estimation
from utility import common_config

# %%

class StatefulNode(advance_search.AdvanceNode):
    """
    {Description}

    Members:
        field1:
        field2:
    """
    def __init__(self, info_dict: Any, tree_id: Any, node_id: int = -1, root_id: int = -1, template_id: int = -1, \
        mode: Any | None = None, extend_table: Any | None = None, init_strategy: str = "random", action_config: Any = {}, \
        schema_order = (), exploration_info = {}, state_manager_ref: state_estimation.StateManager = None):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        if state_manager_ref is not None:
            # 非空时强制写入state_manager
            exploration_info['state_manager'] = state_manager_ref
        super(StatefulNode, self).__init__(info_dict, tree_id, node_id, \
            root_id, template_id, mode, extend_table, init_strategy, action_config)

        # 使用新的analyzer
        self.query_analyzer = stateful_analysis.StatefulAnalyzer(self.query_instance, mode, \
            exploration_dict=exploration_info, split_budget=common_config.get_split_budget(), \
            all_distinct_states = state_manager_ref.state_dict.keys())
        
        self.schema_order = schema_order    # 表示schema连接的顺序

        # 探索信息
        self.state_manager = state_manager_ref
        self.exploration_info = exploration_info


    def is_end(self,):
        """
        根据path_list判断是否为终止节点，True代表没有后续节点
        False代表有后续节点
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        if "path_list" not in self.exploration_info:
            # print("StatefulNode.is_end: path_list is None.")
            # 没有path_list，不需要参与调用
            return False
        
        path_list = self.exploration_info['path_list']
        func = utils.construct_alias_list
        workload = self.query_instance.workload
        print(f"StatefulNode.is_end: schema_order = {func(self.schema_order, workload)}. "\
              f"path_list = {[func(path, workload) for path in path_list]}.")
        for path in path_list:
            if len(path) > len(self.schema_order):
                return False
        return True

    def set_schema_order(self, schema_order):
        """
        设置schema的序列
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        self.schema_order = tuple(schema_order)

    def get_analyze_cases(self,):
        """
        获得探索过程中分析的所有实例，即具有估计基数，但是没有真实基数的结果
    
        Args:
            arg1:
            arg2:
        Returns:
            analyze_case_list:
        """
        # 只包含估计基数的探索
        analyze_case_list = []
        record_list = self.query_analyzer.get_analyze_records()

        for meta, query, card_dict, target_table, flag in record_list:
            new_card_dict = {
                "true": {
                    "subquery": {},
                    "single_table": {}
                },
                "estimation": card_dict
            }
            schema_order = self.schema_order + (target_table,)
            case_dict = {
                "query_meta": meta,
                "card_dict": new_card_dict,
                "valid": flag,
                "schema_order": schema_order
            }
            analyze_case_list.append(case_dict)

        return analyze_case_list
    
    def get_current_case(self,):
        """
        获得当前的case
    
        Args:
            arg1:
            arg2:
        Returns:
            current_case:
        """
        # 获得当前实例的case
        query_text, query_meta, card_dict = node_query.construct_instance_element(self.query_instance)
        res_dict = self.extension_ref.verification_instance
        p_error, estimation_cost, true_cost = \
            res_dict['p_error'], res_dict['est_cost'], res_dict['true_cost']
        
        true_plan, est_plan = res_dict['true_plan'], res_dict['est_plan']
        # current_case = query_text, query_meta, card_dict, (p_error, estimation_cost, true_cost)
        
        # 关于是否valid，over-estimation和under-estimation得分开讨论 
        valid = res_dict['valid']

        current_case = {
            "query_meta": query_meta,
            "card_dict": card_dict, "valid": valid,
            "true_plan": true_plan, "est_plan": est_plan,
            "true_cost": true_cost, "est_cost": estimation_cost,
            "p_error": p_error, "schema_order": self.schema_order   # 这里的schema_order指的是系统需要控制的order
        }

        if valid == False:
            # 检测是否出现valid = False的情况
            print(f"get_current_case: mode = {self.mode}. valid = False. schema_order = {self.schema_order}. true_plan = {true_plan.leading}")
            # 更新schema_order的字段，如果是bushy状态直接把字段设成None
            if res_dict['is_bushy'] == False:
                current_case["schema_order"] = res_dict["join_order"]
            else:
                current_case["schema_order"] = None
                
        return current_case
    

    def initialize_actions(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        self.query_analyzer.set_node_info(self.template_id, self.root_id, self.node_id)
        return super().initialize_actions()

    def try_init(self, cost_estimation, cost_true):
        """
        尝试
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        top_key = self.query_instance.top_key
        top_card = self.query_instance.true_card_dict[tuple(top_key)]

        # 2024-03-23: 顶层基数为0时，便不再拓展新的表/条件
        if top_card <= 0:
            print(f"StatefulNode.try_init: top_key = {top_key}. top_card = {top_card}.")
            return
        
        indicator = cost_estimation / cost_true > 0.98
        # print(f"StatefulNode.try_init: template_id = {self.template_id}. root_id = {self.root_id}. "\
        #       f"node_id = {self.node_id}. indicator = {indicator}. cost_est = {cost_estimation:.2f}. cost_true = {cost_true:.2f}.")
        
        if indicator == True:
            self.initialize_actions()   # 针对当前节点初始化动作
        else:
            # print(f"load_external_true_card: cost_true = {cost_true:.2f}. cost_estimation = {cost_estimation:.2f}.")
            pass
    

    def take_action(self, action, sync):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        # print("take_action: action = {}.".format(action))
        if sync == False:
            # 返回额外的内容
            estimate_benefit, new_extension_instance, new_query_instance, est_expected_cost, \
                est_actual_cost = self.query_analyzer.exploit_async(table_name=action, config=self.action_config)
            # print(f"take_action: node_repr = {self.query_repr()}. action = {action}. "\
            #     f"estimated_benefit = {estimate_benefit: .2f}. estimated_p_error = {est_actual_cost / est_expected_cost: .2f}.")
        else:
            new_query_instance, benefit, actual_cost, expected_cost, actual_card, expected_card = \
                self.query_analyzer.exploit(table_name=action, config=self.action_config)
            estimate_benefit = benefit
            est_expected_cost, est_actual_cost = expected_cost, actual_cost
            # print(f"take_action: node_repr = {self.query_repr()}. action = {action}. "\
            #       f"actual_benefit = {benefit: .2f}. actual_p_error = {actual_cost / expected_cost: .2f}.")
            # print(f"take_action: node_repr = {self.query_repr()}. actual_cost = {actual_cost:.2f}. expected_cost = {expected_cost:.2f}.")

        info_dict = {
            "query_instance": new_query_instance,       #
            "selected_tables": self.selected_tables,    # 
            # "benefit": self.init_value_dict[action],  # 初始收益
            "benefit": estimate_benefit,                # 修改benefit的来源
            "max_depth": self.max_depth,                # 最大深度
            "timeout": self.timeout
        }

        try:
            curr_path_list = self.exploration_info['path_list']
            curr_path_dict = utils.prefix_aggregation(self.schema_order, curr_path_list)
            new_path_list = curr_path_dict[action]

            # 新的探索信息
            new_exploration_info = {
                "path_list": new_path_list,
                # "state_manager": self.state_manager   # 转移到__init___中导入结果
            }

            if "ref_index_dict" in self.exploration_info.keys():
                new_exploration_info["ref_index_dict"] = {k: self.exploration_info['ref_index_dict'][k] for k in new_path_list}
            else:
                print(f"StatefulNode.take_action: exploration_info = {self.exploration_info.keys()}")
        except KeyError as e:
            new_exploration_info = {}

        new_order = self.schema_order + (action,)
        new_node = StatefulNode(info_dict = info_dict, tree_id=self.tree_id, mode=self.mode, 
            extend_table=action, init_strategy=self.init_strategy, action_config=self.action_config, 
            schema_order=new_order, exploration_info = new_exploration_info, state_manager_ref = self.state_manager)

        # 设置节点的收益相关信息
        new_node.set_benefit_properties(est_actual_cost, est_expected_cost)
        if sync == False:
            new_node.extension_ref = new_extension_instance
        else:
            new_node.initialize_actions()
            new_node.is_complete = True
            new_node.selectable = True

        return new_node
    
# %%

class StatefulTree(advance_search.AdvanceTree):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, workload, external_info, max_step, template_id = -1, mode = None, init_strategy = "random", 
            state_manager_ref: state_estimation.StateManager = None, exploration_info = {}):
        """
        {Description}

        Args:
            workload:
            external_info: 
            max_step: 
            template_id: 
            mode:
            init_strategy: 
            state_manager_ref:
            exploration_info: 来自外部的探索信息，具体格式为 {
                "path_list": [路径列表，表示所有需要探索的路径]
            }
        """
        # 新添加的变量
        self.workload = workload
        self.state_manager = state_manager_ref
        self.exploration_info = deepcopy(exploration_info)

        # print(f"StatefulTree.__init__: template_id = {template_id}. max_step = {max_step}. external_info = {external_info}.")
        # print(f"StatefulTree.__init__: exploration_info = {exploration_info}")
        super(StatefulTree, self).__init__(external_info, \
            max_step, template_id, mode, init_strategy)


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
        print(f"StatefulTree.create_root: creation start. template_id = {self.template_id}. root_id = {self.root_id}.")
        node_id = self.get_next_node_id()
        root_schema = tuple(sorted(self.external_info['query_instance'].query_meta[0]))

        root_node = StatefulNode(self.external_info, self.tree_id, \
            node_id, self.root_id, self.template_id, self.mode, None, \
            self.init_strategy, schema_order=root_schema, \
            exploration_info=self.exploration_info, state_manager_ref=self.state_manager)
        
        action, value = root_node.initialize_actions()
        print(f"StatefulTree.create_root: creation end. template_id = {self.template_id}. root_id = {self.root_id}.")
        # root_node.set_schema_order(root_schema)

        if self.state_manager is not None:
            analyze_case_list = root_node.get_analyze_cases()
            if self.mode == "under-estimation":
                # 只有under-estimation才能添加结果
                self.state_manager.add_instance_list(analyze_case_list)     # 添加尝试拓展的实例

            # root_case = root_node.get_current_case()
            # self.state_manager.add_new_instance(root_case)

        root_node.is_complete = True    # 
        if action != "terminate":
            root_node.selectable = True
        else:
            root_node.selectable = False
            
        if action == "terminate":
            self.is_terminate = True
        return root_node


    def enumerate_all_actions(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            result_list: 包含node_signature, estimate_benefit, new_node成员
            return2:
        """
        result_list = []
        max_expl_time = 20

        while True:
            node_signature, estimate_benefit, new_node = self.one_step_search(mode="uct", sync=False)
            if node_signature == "":
                # print(f"enumerate_all_actions: no more nodes now. tmpl_id = {self.template_id}. tree_id = {self.tree_id}. node_signature = {node_signature}.")
                break

            result_list.append((node_signature, estimate_benefit, new_node, self))

            max_expl_time -= 1
            if max_expl_time <= 0:
                # print(f"enumerate_all_actions: meet max_expl_time. tmpl_id = {self.template_id}. tree_id = {self.tree_id}. node_signature = {node_signature}.")
                break

        return result_list

    def expand_node(self, node: StatefulNode, action, sync):
        """
        拓展一个新节点
    
        Args:
            node:
            action:
            sync:
        Returns:
            return1:
            return2:
        """
        if node.check_existence(action) == False:
            new_node = node.take_action(action=action, sync=sync)
            new_node.parent = node
            node.add_child(new_node, action)

            # 同步选项下，更新state_manager
            if sync == True and self.state_manager is not None:
                analyze_case_list = new_node.get_analyze_cases()
                current_case = new_node.get_current_case()
                self.state_manager.add_new_instance(**current_case)         # 添加当前实例

                if self.mode == "under-estimation":
                    self.state_manager.add_instance_list(analyze_case_list)     # 添加尝试拓展的实例

            node_id = self.get_next_node_id()

            # 设置node_id和template_id
            new_node.set_node_id(node_id)
            new_node.set_root_id(self.root_id)
            new_node.set_template_id(self.template_id)

            # trace("new_node: tree_id = {}. node_id = {}. new_table = {}.".\
            #       format(self.tree_signature, new_node.node_id, action))
            # trace("new_edge: tree_id = {}. src_id = {}. dst_id = {}.".\
            #       format(self.tree_signature, node.node_id, new_node.node_id))
            
            if sync == True:
                # 同步模式下，将新生成的case加入到manager中
                pass

            return new_node, "leaf"     # 表示到达叶节点了
        else:
            next_node: StatefulNode = node.visit_child_by_action(action)
            if next_node.is_terminate == False:
                return next_node, "nonleaf"
            else:
                return next_node, "leaf"

    def update_node_state(self, node_signature, external_info):
        """
        更新当前节点状态
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        selected_node: StatefulNode = self.node_dict[node_signature]

        subquery_local, single_table_local = external_info['subquery'], external_info['single_table']
        if utils.card_dict_valid_check(subquery_local, single_table_local) == False:
            # 2024-03-27: 基数非法的情况，直接退出
            print(f"update_node_state: card_dict invalid. template_id = {self.template_id}. root_id = {self.root_id}. "\
                  f"node_id = {selected_node.node_id}. subquery_values = {subquery_local.values()}.")
            # 将节点改成完成但不可被选择的状态
            selected_node.is_complete = True
            selected_node.selectable = False
            selected_node.p_error = 0.0
            # 表示探索完成，但是benefit为负
            return True, -1, 0, 0
        
        flag, curr_benefit, cost_true, cost_estimation = selected_node.load_external_true_card(
            subquery_dict=external_info['subquery'], single_table_dict=external_info['single_table'], 
            sample_config=self.sample_config)
        
        # 保存新生成的结果
        if flag == True and self.state_manager is not None:
            analyze_case_list = selected_node.get_analyze_cases()
            current_case = selected_node.get_current_case()
            self.state_manager.add_new_instance(**current_case)         # 添加当前实例
            if self.mode == "under-estimation":
                # 如果是under-estimation，则添加estimation相关case
                self.state_manager.add_instance_list(analyze_case_list)     # 
        if flag == True:
            # 完整的基数
            # trace("finish: tree_id = {}. node_id = {}.".format(self.tree_signature, selected_node.node_id))
            self.update_actual_reward(node=selected_node, benefit=curr_benefit) # 向上传播真实的更新
            
        return flag, curr_benefit, cost_true, cost_estimation
    
    # def one_step_search(self,):
    #     """
    #     {Description}

    #     Args:
    #         arg1:
    #         arg2:
    #     Returns:
    #         return1:
    #         return2:
    #     """
    #     pass


# %%
