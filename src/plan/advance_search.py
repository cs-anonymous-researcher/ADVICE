#!/usr/bin/env python
# -*- coding: utf-8 -*-


# %%

import numpy as np
import pandas as pd
from os.path import join as p_join
from matplotlib import pyplot as plt
import json, os, time, re, math
import pickle

from plan import plan_search, advance_analysis, node_extension
from estimation import plan_estimation, estimation_interface
from utility.utils import benefit_calculate, get_signature, trace
from data_interaction import mv_management

from utility import utils, common_config
from utility.common_config import benefit_config

# %%

class AdvanceNode(plan_search.ExplorationNode):
    """
    {Description}

    Members:
        info_dict:
        tree_id:
        node_id:
        root_id:
        template_id:
        mode:
        extend_table:
        init_strategy:
        action_config:
    """

    def __init__(self, info_dict, tree_id, node_id = -1, root_id = -1, template_id = -1, 
        mode = None, extend_table = None, init_strategy = "random", action_config = {}):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        # print(f"AdvanceNode.__init__: mode = {mode}.")
        assert mode in ("under-estimation", "over-estimation")
        assert init_strategy in ("random", "multi-loop", "reference"), f"AdvanceNode.__init__: init_strategy = {init_strategy}"
        super(AdvanceNode, self).__init__(info_dict=info_dict)

        self.action_config = action_config
        self.action_config['mode'] = mode

        self.init_strategy = init_strategy
        self.query_analyzer = advance_analysis.AdvanceAnalyzer(self.query_instance, 
                mode, split_budget=common_config.get_split_budget())
        self.prev_benefit = 0.0     # 过去计算的收益
        self.p_error = 0.0

        self.extension_ref: node_extension.ExtensionInstance = None   # 指向的extension_instance，用于异步的探索
        self.signature = self.generate_signature()

        self.is_complete = False    # 节点探索是否完毕（基数是否补全）
        self.mode, self.extend_table = mode, extend_table            # 添加error模式

        # 添加节点和树的ID
        self.tree_id, self.node_id, self.root_id, self.template_id = \
            tree_id, node_id, root_id, template_id

        # 可以被选择的
        self.selectable = False


    def evaluate_select_state(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        condition = ""
        if self.is_complete == False:
            self.selectable = False
            condition = "case 0"
            # return self.selectable
        else:
            self.selectable = False
            condition = "case 3"
            for action in self.action_set:
                if action not in self.children_dict.keys():
                    self.selectable = True
                    condition = "case 1"
                    break
                    # return self.selectable
                else:
                    if self.visit_child_by_action(action).selectable == True:
                        self.selectable = True
                        condition = "case 2"
                        break
                        # return self.selectable

        # print("evaluate_select_state: tree_id = {}. node_id = {}. action_set = {}. children_dict = {}. condition = {}. selectable = {}.".\
        #       format(self.tree_id, self.node_id, self.action_set, self.children_dict.keys(), condition, self.selectable))
        return self.selectable


    def set_node_id(self, node_id):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        self.node_id = node_id

    def set_root_id(self, root_id):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        # print("AdvanceNode.set_root_id: root_id = {}.".format(root_id))
        self.root_id = root_id

    def set_template_id(self, template_id):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        self.template_id = template_id

    def get_available_actions(self,):
        """
        获得可用的actions
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        action_list = []
        for action in self.action_set:
            if action not in self.children_dict.keys():
                action_list.append(action)
            else:
                # if self.visit_child_by_action(action=action).is_complete == True:     # 旧的判断依据
                if self.visit_child_by_action(action=action).selectable == True:        # 新的判断依据
                    action_list.append(action)
        
        # print("get_available_actions: tree_id = {}. node_id = {}. action_set = {}. children_dict = {}. action_list = {}. selectable = {}.".\
        #       format(self.tree_id, self.node_id, self.action_set, self.children_dict.keys(), action_list, self.selectable))
        
        return action_list

    def generate_signature(self,):
        """
        获得节点的唯一标识
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        return mv_management.meta_key_repr(in_meta=self.query_instance.\
            query_meta, workload=self.query_instance.workload)


    @utils.timing_decorator
    def initialize_actions(self, ):
        """
        覆盖之前初始化动作的方法
        
        Args:
            None
        Returns:
            action: 初始的动作
            value: 动作对应的价值
        """
        # 直接判断状态已中止的情况
        if self.is_terminate == True:
            return "terminate", -1
        
        action_list, value_list = self.query_analyzer.init_all_actions(\
            table_subset=self.selected_tables, mode=self.init_strategy)
        
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
        # print("initialize_actions: template_id = {}. root_id = {}. action_list = {}. value_list = {}. init_value_dict = {}".\
        #       format(self.template_id, self.root_id, action_list, utils.list_round(value_list, 2), self.init_value_dict))
        
        self.is_action_init = True
        return pair_list[0]


    def set_benefit_properties(self, actual_cost, expected_cost):
        """
        设置节点相关的属性
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        # print("set_benefit_properties: actual_cost = {}. expected_cost = {}.".\
        #       format(actual_cost, expected_cost))
        self.actual_cost = actual_cost
        self.expected_cost = expected_cost
        self.p_error = actual_cost / expected_cost
        

    def verify_action(self, action, card_est_input = "graph_corr_based"):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        # 获得真实的收益
        new_query_instance, benefit, actual_cost, expected_cost, actual_card, expected_card = \
            self.query_analyzer.exploit(table_name=action, config=self.action_config)
        estimate_benefit = benefit
        est_expected_cost, est_actual_cost = expected_cost, actual_cost

        # 获得估计的收益
        extension_instance = node_extension.get_extension_from_query_instance(new_query_instance)
        local_estimator = plan_estimation.PlanBenefitEstimator(query_extension=\
            extension_instance, card_est_input=card_est_input, target_table=action)
        local_estimator.mask_target_table(table=action)
        reward, eval_result_list = local_estimator.benefit_integration()

        # 展示结果
        p_error = actual_cost / expected_cost
        print(f"verify_action: node_repr = {self.query_repr()}. action = {action}. "
              f"benefit = {benefit: .2f}. p_error = {p_error: .2f}. estimation = {reward:.2f}")

        return {
            "query_text": new_query_instance.query_text,
            "query_meta": new_query_instance.query_meta,
            "actual_p_error": p_error,
            "estimated_p_error": reward
        }

    def take_action(self, action, sync = False):
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
            print(f"take_action: node_repr = {self.query_repr()}. action = {action}. "\
                f"estimated_benefit = {estimate_benefit: .2f}. estimated_p_error = {est_actual_cost / est_expected_cost: .2f}.")
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

        new_node = AdvanceNode(info_dict = info_dict, tree_id=self.tree_id, mode=self.mode, \
                               extend_table=action, init_strategy=self.init_strategy)

        # 设置节点的收益相关信息
        new_node.set_benefit_properties(est_actual_cost, est_expected_cost)
        if sync == False:
            new_node.extension_ref = new_extension_instance
        else:
            
            new_node.initialize_actions()
            new_node.is_complete = True
            new_node.selectable = True

        return new_node
    
    @utils.timing_decorator
    def get_estimated_benefit(self, sample_config, restrict_order = True):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            curr_benefit:
            estimate_expected_cost:
            estimate_actual_cost:
        """

        local_extension = self.extension_ref
        # local_estimator = plan_estimation.PlanBenefitEstimator(query_extension=local_extension, \
        #     card_est_input="equal_diff", mode=self.mode, target_table=self.extend_table)
        
        # estimate_cost_pair, eval_result_list = local_estimator.cost_pair_integration(config=sample_config)
        estimate_cost_pair = estimation_interface.estimate_plan_benefit(local_extension, 
            self.mode, self.extend_table, plan_sample_num=benefit_config["plan_sample_num"], 
            restrict_order = restrict_order)

        curr_benefit = benefit_calculate(estimate_cost_pair[0], estimate_cost_pair[1])
        return curr_benefit, estimate_cost_pair[0], estimate_cost_pair[1]

    def try_init(self, cost_estimation, cost_true):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        print(f"AdvanceNode.try_init: template_id = {self.template_id}. root_id = {self.root_id}. "\
              f"node_id = {self.node_id}. cost_est = {cost_estimation:.2f}. cost_true = {cost_true:.2f}.")
        
        if cost_estimation / cost_true > 1.05:
            self.initialize_actions()   # 针对当前节点初始化动作
        else:
            print(f"load_external_true_card: cost_true = {cost_true:.2f}. cost_estimation = {cost_estimation:.2f}.")

    @utils.timing_decorator
    def load_external_true_card(self, subquery_dict, single_table_dict, sample_config, ):
        """
        加载外部的真实基数信息
        
        Args:
            subquery_dict: 
            single_table_dict: 
            sample_config:
        Returns:
            flag: 探索过程是否完全了
            curr_benefit: 
            cost_true: 
            cost_estimation:
        """
        # 2024-03-27: 需要考虑card存在不合法的情况
        self.query_instance.add_true_card(subquery_dict, mode="subquery")
        self.query_instance.add_true_card(single_table_dict, mode="single_table")
        
        complete_flag, subquery_true, single_table_true = \
            self.extension_ref.load_external_card(subquery_dict, single_table_dict)
        extension = self.extension_ref

        # trace("complete_card: tree_id = {}. node_id = {}. flag = {}. prev_single_table = {}. prev_subquery = {}. curr_single_table = {}. curr_subquery = {}.".\
        #       format(self.tree_id, self.node_id, complete_flag, len(extension.single_table_diff), len(extension.subquery_diff), len(single_table_dict), len(subquery_dict)))
        # print("AdvanceNode.load_external_true_card: tree_id = {}. node_id = {}. flag = {}. prev_single_table = {}. prev_subquery = {}. curr_single_table = {}. curr_subquery = {}.".\
        #       format(self.tree_id, self.node_id, complete_flag, len(extension.single_table_diff), len(extension.subquery_diff), len(single_table_dict), len(subquery_dict)))
        
        if complete_flag == True:
            # 代表探索完全
            self.is_complete = True
            if self.mode == "under-estimation":
                compare_flag, cost1, cost2 = extension.two_plan_verification(subquery_dict1=extension.subquery_true, \
                    single_table_dict1=extension.single_table_true, subquery_dict2=extension.subquery_estimation, \
                    single_table_dict2=extension.single_table_estimation, keyword1="true", keyword2="estimation")
                # assert 
                table_flag = True
            elif self.mode == "over-estimation":
                plan_flag, table_flag, cost1, cost2 = extension.two_plan_verification_under_constraint(subquery_dict1=extension.subquery_true, \
                    single_table_dict1=extension.single_table_true, subquery_dict2=extension.subquery_estimation, \
                    single_table_dict2=extension.single_table_estimation, keyword1="true", keyword2="estimation", target_table=self.extend_table)
                
                # flag = (plan_flag or (not table_flag))
                # if flag == False:
                #     # 为了适应API，调整cost
                #     cost1 = cost2
                # if table_flag == False:
                #     # complete状态下table_flag为False
                #     cost1 = -cost2

            if table_flag == True:
                curr_benefit = benefit_calculate(cost1, cost2)
            else:
                curr_benefit = -1.0
                
            cost_true, cost_estimation = cost1, cost2

            # 感觉这里的逻辑也有待商榷，
            # 应该有benefit的情况下才允许继续往下探索
            # if cost_estimation / cost_true > 1.05:
            #     self.initialize_actions()   # 针对当前节点初始化动作
            # else:
            #     print(f"load_external_true_card: cost_true = {cost_true:.2f}. cost_estimation = {cost_estimation:.2f}.")

            if table_flag == True:
                self.try_init(cost_estimation, cost_true)

            self.selectable = (len(self.action_set) > 0)
            # print("load_external_true_card: tree_id = {}. node_id = {}. is_complete = {}. selectable = {}.".\
            #       format(self.tree_id, self.node_id, self.is_complete, self.selectable))
            # trace("load_external_true_card: template_id = {}. tree_id = {}. node_id = {}. is_complete = {}. selectable = {}.".\
            #       format(self.template_id, self.root_id, self.node_id, self.is_complete, self.selectable))
        else:
            # 代表部分探索，此时只考虑当前收益，所以restrict_order设为False
            curr_benefit, estimate_expected_cost, estimate_actual_cost = \
                self.get_estimated_benefit(sample_config, restrict_order=False)
            cost_true, cost_estimation = estimate_expected_cost, estimate_actual_cost

        self.set_benefit_properties(cost_estimation, cost_true)
        return complete_flag, curr_benefit, cost_true, cost_estimation


    def get_properties(self, config = ["query"]):
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


    def node_repr(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        schema_list = self.query_repr()
        alias_mapping = self.query_instance.data_manager.tbl_abbr

        return [alias_mapping[s] for s in schema_list]


    def get_action_info(self,):
        """
        用于debug，获得所有的action以及对应的value和state，
        均使用list来进行表示
        
        Args:
            None
        Returns:
            action_list:
            value_list:
            state_list:
        """
        action_list = list(self.action_set)
        value_list = []     # action对应的value
        state_list = []     # action对应的state

        for action in action_list:
            if action not in self.children_dict.keys():
                state_list.append("New")
                # value_list.append(self.UCB_score(action))
            else:
                if self.visit_child_by_action(action=action).is_complete == True:
                    state_list.append("Finish")
                    # value_list.append(self.UCB_score(action))
                else:
                    state_list.append("Unfinish")
            
            value_list.append(self.UCB_score(action))

        return action_list, value_list, state_list


# %%


class AdvanceTree(plan_search.ExplorationTree):
    """
    {Description}

    Members:
        field1:
        field2:
    """
    # def __init__(self, external_info, max_step, template_id = -1, mode = "under-estimation"):
    def __init__(self, external_info, max_step, template_id = -1, mode = None, init_strategy="random"):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        # print(f"AdvanceTree.__init__: mode = {mode}.")
        assert mode in ("under-estimation", "over-estimation")
        self.template_id, self.mode = template_id, mode      # 对应的模版ID和模式
        self.root_id, self.init_strategy = -1, init_strategy

        self.max_step = max_step
        self.current_step = 0

        self.node_dict = {}     # signature到node实例的映射
        self.tree_signature = self.generate_signature(str(external_info))
        self.tree_id = self.tree_signature
        super(AdvanceTree, self).__init__(external_info=external_info, create_root=False)

        # 创建tree的signature
        self.next_node_id = 0

        self.root = self.create_root()  # 创建根节点
        node_id = self.get_next_node_id()
        self.root.set_node_id(node_id=node_id)

        # if self.is_terminate == False:
        #     # 只考虑创建成功的case
        #     trace("new_node: tree_id = {}. node_id = {}. new_table = {}.".\
        #             format(self.tree_signature, self.root.node_id, str(self.root.query_instance.query_meta[0])))
        
        self.sample_config = plan_estimation.default_strategy   # 采用默认的策略
        self.is_blocked = False   # 是否处于阻塞(还有节点可以探索，但是当前的探索未完成)的状态

    def set_root_id(self, root_id):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        # print("AdvanceTree.set_root_id: root_id = {}.".format(root_id))
        self.root_id = root_id
        self.root.set_root_id(root_id)
        
    def generate_signature(self, in_str):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        # return "tree_" + self.root.extension_ref.get_extension_signature()
        tree_sig = "tree_" + get_signature(in_str, num_out=8)
        # print("AdvanceTree.generate_signature: tree_sig = {}.".format(tree_sig))

        return tree_sig

    def get_next_node_id(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        res_id = self.next_node_id
        self.next_node_id += 1
        return res_id
    
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
        root_node = AdvanceNode(info_dict=self.external_info, tree_id=self.tree_signature, \
                                mode=self.mode, init_strategy=self.init_strategy)

        action, value = root_node.initialize_actions()
        if action != "terminate":
            root_node.selectable = True
        else:
            root_node.selectable = False
            
        root_node.is_complete = True    # 
        if action == "terminate":
            self.is_terminate = True

        return root_node


    def has_explored_all_nodes(self, return_num = False):
        """
        判断是否探索了所有的节点
    
        Args:
            arg1:
            arg2:
        Returns:
            flag: True代表探索结束，False代表未结束
        """
        flag = True
        def map_func(node: AdvanceNode):
            return node.node_id, node.selectable, node.is_complete

        def reduce_func(result_list):
            return result_list

        result_list = self.traversal(map_func, reduce_func)
        # print("has_explored_all_nodes: root_id = {}. result_list(selectable, is_complete) = {}.".format(self.root_id, result_list))
        unfinish_list = []

        for node_id, selectable, is_complete in result_list:
            if is_complete == False or selectable == True:
                unfinish_list.append((node_id, selectable, is_complete))
                flag = False

        # if flag == False:
        #     print(f"has_explored_all_nodes: template_id = {self.template_id}. root_id = {self.root_id}. "\
        #           f"unfinish_list(node_id, selectable, is_complete) = {unfinish_list}")

        if self.is_terminate == True:
            # 强制探索终止的信号
            flag = True
            
        if return_num == True:
            return flag, len(result_list)
        else:
            return flag
    

    def get_tree_exploration_state(self,):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        # def map_func(node:AdvanceNode):
        #     return node.node_id, node.is_complete
        def map_func(node:AdvanceNode):
            return node.node_id, node.is_complete, node.p_error
        
        def reduce_func(result_list):
            return result_list
        
        result_list = self.traversal(map_func, reduce_func)
        return result_list

    def tree_state_evaluation(self,):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        step = self.current_step

        def map_func(node:AdvanceNode):
            return node.node_id, node.benefit, node.visit_number, node.p_error, node.is_complete

        def reduce_func(result_list):
            return result_list

        result_list = self.traversal(map_func, reduce_func)
        
        for node_id, benefit, visit_number, p_error, state in result_list:
            # 打印相关的路径
            trace("node_state: template_id = {}. root_id = {}. node_id = {}. step = {}. benefit = {:.2f}. visit_num = {}. p_error = {:.2f}. is_complete = {}.".\
                  format(self.template_id, self.root_id, node_id, step, benefit, visit_number, p_error, state))


    def one_step_search(self, mode = "uct", sync = False):
        """
        执行单步的搜索过程，并且返回需要探索的实例
        
        Args:
            mode:
            is_async:
        Returns:
            node_signature: 
            estimate_benefit: 
            new_node:
        """
        if self.root.selectable == False:
            flag, node_num = self.has_explored_all_nodes(return_num = True)
            print(f"one_step_search: The whole tree is unselectable! explored_all_flag = {flag}. "\
                  f"template_id = {self.template_id}. root_id = {self.root_id}. node_num = {node_num}.")
            
            if flag == True:
                self.is_terminate = True    # 设置终止了
                self.is_blocked = True      # 设置阻塞状态
            else:
                self.is_blocked = True      # 设置阻塞状态

            return "", 0.0, None

        if self.is_terminate == True:
            # raise ValueError("one_step_search: Tree exploration has finished!")
            print("one_step_search: Tree exploration has finished! "\
                  f"template_id = {self.template_id}. root_id = {self.root_id}.")
            return "", 0.0, None
        
        self.current_step += 1
        if mode == "random":
            # 随机探索
            pass
        elif mode == "uct":
            # 基于UCT信息的探索
            def visit(node, curr_type):
                if curr_type == "leaf":
                    # 到达叶节点
                    # print("one_step_search: reach to leaf node.")
                    benefit = self.calculate_benefit(node)
                    return node, benefit
                elif curr_type == "nonleaf":
                    action = self.select_action(node)
                    if action == "error":
                        # 代表出现了预期之外的行为
                        raise ValueError("one_step_search.visit: action is error!")
                        return None, 0.0
                    elif action != "terminate":
                        # print("one_step_search: call expand_node.")
                        next_node, next_mode = self.expand_node(node, action, sync)
                        return visit(next_node, next_mode)
                    else:
                        # terminate action代表这也是一个叶节点
                        # print("one_step_search: current action is terminate.")
                        return node, self.calculate_benefit(node)
                else:
                    raise ValueError("one_step_search.visit: Unsupported curr_type({})".format(curr_type))
                
            # {}
            new_node, benefit = visit(self.root, curr_type="nonleaf")
            # trace("visit: tree_id = {}. node_id = {}. step = {}.".\
            #       format(self.tree_signature, new_node.node_id, self.current_step))
            
            new_node: AdvanceNode = new_node
        else:
            raise ValueError("one_step_search: Unsupported mode({})".format(mode))
        
        if self.current_step >= self.max_step:
            # 超出搜索步骤，直接设置为终止
            self.is_terminate = True

        # 完成node信息的导入
        node_signature = new_node.signature
        self.node_dict[node_signature] = new_node

        # 完成初步的更新
        estimate_benefit = self.calculate_benefit(new_node)
        self.update_estimated_reward(node=new_node, \
            benefit=estimate_benefit, first_update=True)
        
        # self.tree_state_evaluation()    # 检查树的状态    
        # 
        return node_signature, estimate_benefit, new_node
    

    def calculate_estimated_benefit(self, node:AdvanceNode):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        return node.get_estimated_benefit(sample_config={})

    # @utils.timing_decorator
    def update_actual_reward(self, node: AdvanceNode, benefit):
        """
        涉及到探索结束的状态更新
    
        Args:
            node:
            benefit:
            is_finish:
        Returns:
            return1:
            return2:
        """
        # is_finish = False
        # if node:
        #     is_finish = True

        current_node = node
        delta_benefit = benefit - node.prev_benefit    # 
        node.prev_benefit = benefit
        while current_node is not None:
            current_node.evaluate_select_state()    # 更新状态
            current_node.benefit += delta_benefit
            current_node = current_node.parent

        return benefit

    # @utils.timing_decorator
    def update_estimated_reward(self, node: AdvanceNode, benefit, first_update):
        """
        更新估计的reward，有点类似simulation的过程
        注意，estimated的过程可能会执行多次
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        # verbose("AdvanceTree.update_estimated_reward: node = {}. benefit = {}.".\
        #         format(node.query_repr(), benefit))
        current_node = node
        delta_benefit = benefit - node.prev_benefit    # 
        node.prev_benefit = benefit

        # 第一个更新节点的benefit应该直接替代
        node.benefit = benefit                         

        while current_node is not None:
            current_node.evaluate_select_state()    # 更新状态
            if first_update == True:
                current_node.visit_number += 1
            current_node.benefit += delta_benefit
            current_node = current_node.parent

        return benefit
    
    def set_sample_config(self, sample_config):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        self.sample_config = sample_config

    @utils.timing_decorator
    def update_node_state(self, node_signature, external_info):
        """
        使用外部的结果更新搜索节点的状态
    
        Args:
            arg1:
            arg2:
        Returns:
            flag: 
            curr_benefit: 
            cost_true:
            cost_estimation:
        """
        selected_node: AdvanceNode = self.node_dict[node_signature]
        flag, curr_benefit, cost_true, cost_estimation = selected_node.load_external_true_card(\
            subquery_dict=external_info['subquery'], single_table_dict=external_info['single_table'], \
            sample_config=self.sample_config)
        
        print("update_node_state: flag = {}. curr_benefit = {}. cost_true = {}. cost_estimation = {}.".\
              format(flag, curr_benefit, cost_true, cost_estimation))

        if flag == True:
            # 完整的基数
            # trace("finish: tree_id = {}. node_id = {}.".format(self.tree_signature, selected_node.node_id))
            self.update_actual_reward(node=selected_node, benefit=curr_benefit) # 向上传播真实的更新
        else:
            # 部分的基数
            # trace("")
            # trace("unfinish: tree_id = {}. node_id = {}.".format(self.tree_signature, selected_node.node_id))
            self.update_estimated_reward(node=selected_node, benefit=curr_benefit, first_update=False)

        return flag, curr_benefit, cost_true, cost_estimation

    def select_action(self, node: AdvanceNode):
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
            # print("AdvanceTree.select_action: self.action_count = {}.".format(self.action_count))
            # raise ValueError("AdvanceTree.select_action: node({}) is terminate.".\
            #                  format(node.query_repr()))
            Warning("AdvanceTree.select_action: node({}) is terminate.".\
                             format(node.query_repr()))
            return "error"
        
        
        # 判断action
        if node.is_action_init == False:
            # 初始化所有的动作，并且返回价值最高的动作
            action, _ = node.initialize_actions()
            print(f"select_action: initialize_actions. node = {node.node_repr()}. action = {action}.")
        else:
            # 选择一个动作
            action_list = node.get_available_actions()
            action = node.highest_UCB_action(action_list=action_list)
            # print(f"select_action: highest_UCB_action. node = {node.node_repr()}. action = {action}.")

        # 额外的信息输出，用于debug
        action_list, value_list, state_list = node.get_action_info()
        # trace("select_action: tree_id = {}. node_id = {}. action_list = {}. state_list = {}. value_list = {}.".\
        #       format(self.tree_signature, node.node_id, action_list, state_list, value_list))
        
        return action

    def expand_node(self, node:AdvanceNode, action, sync):
        """
        尝试在当前的节点上扩展新的节点，如果节点存在的话
    
        Args:
            node:
            action:
        Returns:
            return1:
            return2:
        """
        # verbose("ExplorationTree.expand_node: node = {}. action = {}.".\
        #         format(node.query_repr(), action))
        
        if node.check_existence(action) == False:
            new_node = node.take_action(action=action, sync=sync)
            new_node.parent = node
            node.add_child(new_node, action)

            node_id = self.get_next_node_id()

            # 设置node_id和template_id
            new_node.set_node_id(node_id)
            new_node.set_root_id(self.root_id)
            new_node.set_template_id(self.template_id)

            # trace("new_node: tree_id = {}. node_id = {}. new_table = {}.".format(self.tree_signature, new_node.node_id, action))
            # trace("new_edge: tree_id = {}. src_id = {}. dst_id = {}.".\
            #       format(self.tree_signature, node.node_id, new_node.node_id))
            
            return new_node, "leaf"     # 表示到达叶节点了
        else:
            next_node: AdvanceNode = node.visit_child_by_action(action)
            if next_node.is_terminate == False:
                return next_node, "nonleaf"
            else:
                return next_node, "leaf"
            
# %%
