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

from utility import utils, common_config
from plan import plan_search, node_query, plan_template
from collections import defaultdict

class ForestExplorationExperiment(object):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, workload, expt_config, template_meta_path = \
        "/home/lianyuan/Research/CE_Evaluator/intermediate/stats/template_obj/internal/meta_info.json",
        split_budget = 100):
        """
        {Description}

        Args:
            workload:
            expt_config:
            template_meta_path:
        """
        self.workload = workload
        self.split_budget = split_budget

        self.template_manager = plan_template.TemplateManager(workload=self.workload, 
            inter_path = os.path.dirname(template_meta_path), split_budget=split_budget, 
            dynamic_config=common_config.dynamic_creation_config)

        self.expt_config = expt_config
        self.selected_tables = expt_config['selected_tables']   # 选取所有表进行实验

        # 模板的字典，根节点的字典，利用ID优化访问效率
        # self.template_id_dict, self.root_id_dict = {}, {}
        # self.template_plan_id, self.root_id = -1, -1
        self.root_id_dict = {}
        self.root_id = -1

        self.curr_template_plan = None       # 当前的模版查询计划
        self.curr_search_tree = None         # 当前的搜索树

        # 模板元信息的路径
        self.template_meta_path = template_meta_path
        self.template_meta_dict = utils.load_json(self.template_meta_path)

        # print(f"ForestExplorationExperiment.__init__: self.template_meta_dict = {self.template_meta_dict}")

        for k, v in self.template_meta_dict.items():
            self.root_id_dict[k] = {}

        self.template_id_list = []

    def load_if_not_exist(self, template_id):
        """
        如果模版没有出现在Manager中，将它加载进去
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        if isinstance(template_id, int):
            template_id = str(template_id)
        # print("ForestExplorationExperiment.load_if_not_exist: template_meta_dict = {}.".\
        #       format(self.template_meta_dict))
        # print("ForestExplorationExperiment.load_if_not_exist: template_id = {}.".format(template_id))
        template_path = self.template_meta_dict[template_id]["info"]["path"]
        template_key = self.template_meta_dict[template_id]["template_key"]

        if template_key in self.template_manager.template_dict.keys():
            return template_key
        else:
            self.template_manager.load_historical_template(template_path=template_path)
            return template_key

    def get_template_by_id(self, id) -> plan_template.TemplatePlan:
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        # key = self.template_id_dict[id]['key']
        key = self.load_if_not_exist(template_id=id)
        return self.template_manager.template_dict[key]

    def get_root_by_id(self, id):
        """
        {Description}
        
        Args:
            id:
            arg2:
        Returns:
            root:
            res2:
        """
        # print("get_root_by_id: self.template_plan_id = {}. id = {}. root_id_dict.keys() = {}".\
        #       format(self.template_plan_id, id, self.root_id_dict[self.template_plan_id].keys()))
        return self.root_id_dict[self.template_plan_id][id]

    def switch_template(self, template_id):
        """
        选择探索的模版

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        if isinstance(template_id, int):
            template_id = str(template_id)
        self.curr_template_plan: plan_template.TemplatePlan = self.get_template_by_id(id=template_id)
        # self.curr_template_plan.bind_grid_plan()
        
        self.template_plan_id = template_id
        return self.curr_template_plan


    def select_root(self, query_id):
        """
        选择根节点

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        self.root_id = query_id
        self.curr_search_tree = self.get_root_by_id(id=query_id)
        return self.curr_search_tree
    

    def begin_search_process(self, config = {}):
        """
        开始搜索过程
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """

        max_step = config.get("max_step", 5)
        optimal_config = self.curr_search_tree.main_search_process(\
            max_step=max_step, return_option = config['return'])
        
        return optimal_config

    def show_current_templates(self,):
        """
        显示当前的状态
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        # for id, key in self.template_id_dict.items():
        #     print("template_id = {}. feature = {}.".format(id, key))
        for id, desc in self.template_meta_dict.items():
            print("template_id = {}. desc = {}.".format(id, desc))

    def show_current_queries(self,):
        """
        显示当前的查询
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        pass

    def create_new_root(self, external_config = {}, tree_config = {"max_depth":5, "timeout": 60000}, switch_root = True):
        """
        创建新的根节点
    
        Args:
            external_config: 关于新root的配置信息
            switch_root: 是否切换为新的搜索树
        Returns:
            new_root_id: 新的根节点ID
            flag: 
        """
        new_root_id = self.root_id + 1  # 新的根节点
        workload = self.workload

        if len(external_config) == 0:
            external_config = {"target": "under", "min_card": 5000, "max_card": 1000000}

        self.curr_template_plan.grid_info_adjust()  # 调整grid的信息
        self.curr_template_plan.set_ce_handler(external_handler="internal")

        max_try_times = 10
        flag = False

        for _ in range(max_try_times):
            selected_query, selected_meta, true_card, estimation_card = \
                self.curr_template_plan.explore_init_query(external_info=external_config)        # 获得新的目标查询
            #
            root_query = node_query.get_query_instance(workload=workload, query_meta=selected_meta)
            external_info = {
                "query_instance": root_query,
                "selected_tables": self.selected_tables,
                "max_depth": tree_config['max_depth'],     # 最大深度hard-code进去,
                "timeout": tree_config['timeout']          # 查询时间限制在1min
            }
            new_search_tree = plan_search.ExplorationTree(external_info=external_info)
            if new_search_tree.is_terminate == True:
                # 说明这不是一个好的Tree
                continue

            self.root_id_dict[self.template_plan_id][new_root_id] = new_search_tree
            if switch_root == True:
                self.curr_search_tree = new_search_tree
                self.root_id = new_root_id
            flag = True
            return new_root_id, flag
        
        return new_root_id, flag

    def save_state(self,):
        """
        保存当前的状态
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """

    def root_tuning(self, ):
        """
        针对当前root的调优
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        return self.curr_search_tree.root_tuning()


    def Growing_MAB_workload_generation(self, template_id_list, step, root_config, \
                                           tree_config, search_config, total_time = None):
        """
        {Description}
    
        Args:
            template_id_list: 
            step: 
            root_config:
            tree_config: 
            search_config: 
            total_time:
        Returns:
            query_list:
            meta_list:
            result_list:
            card_dict_list:
        """
        query_list, meta_list, result_list, card_dict_list = [], [], [], []

        # TODO: 实现具体该方法

        return query_list, meta_list, result_list, card_dict_list

    def Correlated_MAB_workload_generation(self, template_id_list, step, root_config, \
                                           tree_config, search_config, total_time = None):
        """
        {Description}
    
        Args:
            template_id_list: 
            step: 
            root_config:
            tree_config: 
            search_config: 
            total_time:
        Returns:
            query_list:
            meta_list:
            result_list:
            card_dict_list:
        """
        query_list, meta_list, result_list, card_dict_list = [], [], [], []

        # TODO: 实现具体该方法

        return query_list, meta_list, result_list, card_dict_list
    
    def exploration_reward(self, query_list, result_list):
        """
        计算一次探索的收益情况，可能需要考虑多种收益的问题

        最简单的情况：统计所有查询中p_error的平均值
        更复杂的情况：
        
        Args:
            query_list:
            result_list:
        Returns:
            reward:
        """
        p_error_list = []
        for item in result_list:
            p_error_list.append(item[2])

        # output位于[0, 1]
        reward = np.average(p_error_list) - 1.0
        reward = reward / 2.0
        if reward >= 1.0:
            reward = 1.0
        return reward

    def Epsilon_Greedy_workload_generation(self, template_id_list, step, root_config, \
                                           tree_config, search_config, mode = "linear", total_time = None):
        """
        {Description}
        
        Args:
            template_id_list: 模版ID的列表
            step: 总的搜索步数
            tree_config: 创建树的设置
            search_config: 搜索过程的设置
            mode: epsilon_decay的模式
        Returns:
            query_list:
            meta_list:
            result_list:
            card_dict_list:
        """
        start_time = time.time()

        # 解析模版ID
        template_id_list = self.parse_template_id_list(template_id_list)

        visit_dict = defaultdict(int)
        reward_dict = defaultdict(list)

        def linear_iter_gen(start_val, end_val, total_step):
            delta = (start_val - end_val) / total_step
            # 生成linear epsilon的迭代器
            def local_func(in_val):
                if in_val - delta <= end_val:
                    out_val = in_val
                else:
                    out_val = in_val - delta
                return out_val

            return local_func
        
        def exponential_iter_gen(start_val, end_val, total_step):
            if end_val <= 0.0:
                # 避免出现0的情况
                end_val = 1e-3
            factor = (end_val / start_val) ** (1 / total_step)
            print("factor = {}.".format(factor))
            # 生成exponential epsilon的迭代器
            def local_func(in_val):
                if in_val * factor <= end_val:
                    out_val = in_val
                else:
                    out_val = in_val * factor

                return out_val
            
            return local_func
        
        # 手动指定开始到结束的概率值
        start_val, end_val = 1.0, 0.1
        if mode == "linear":
            iter_func = linear_iter_gen(start_val=start_val, end_val=end_val, total_step=step)
        elif mode == "exponential":
            iter_func = exponential_iter_gen(start_val=start_val, end_val=end_val, total_step=step)

        query_list, meta_list, result_list, card_dict_list = [], [], [], []
        curr_step = 0
        curr_epsilon = start_val

        def get_random_template():
            # 随机选择template
            return np.random.randint(low=0, high=len(template_id_list))
        
        def get_best_template():
            # 取平均reward最大的template
            # avg_list = [np.average(reward_dict[idx]) for idx in range(len(template_id_list))]
            avg_list = []
            for idx in range(len(template_id_list)):
                if len(reward_dict[idx]) == 0:
                    avg_list.append(0.5)
                else:
                    avg_list.append(np.average(reward_dict[idx]))

            return np.argmax(avg_list)
        
        while True:
            indicator = np.random.uniform(0, 1)
            if indicator < curr_epsilon:
                # 随机选择动作
                idx = get_random_template()
                print("ForestExplorationExperiment.Epsilon_Greedy_workload_generation: get_random_template. idx = {}.".format(idx))
            else:
                # 选择average reward最大的动作
                idx = get_best_template()
                print("ForestExplorationExperiment.Epsilon_Greedy_workload_generation: get_best_template. idx = {}.".format(idx))

            # 更新epsilon
            curr_epsilon = iter_func(curr_epsilon)

            self.switch_template(template_id=template_id_list[idx])                          # 选择模版
            # new_root_id, flag = self.create_new_root(external_config=tree_config)          # 创建新的节点
            new_root_id, flag = self.create_new_root(external_config=root_config, tree_config=tree_config)          # 创建新的节点

            if flag == True:
                # 考虑创建节点也会有失败的情况，需要跳过某些节点
                query_local, meta_local, result_local, card_dict_local = self.begin_search_process(config = search_config)  # 执行搜索的过程

                # 这里需要计算reward
                reward = self.exploration_reward(query_list=query_local, result_list=result_local)
                print("Epsilon_Greedy_workload_generation: reward = {}.".format(reward))
            else:
                reward = 0.0

            visit_dict[idx] += 1
            reward_dict[idx].append(reward)

            # 添加结果到全局
            query_list.extend(query_local)
            meta_list.extend(meta_local)
            result_list.extend(result_local)
            card_dict_local.extend(card_dict_local)

            curr_step += 1
            if curr_step >= step:
                break
            
            curr_time = time.time()
            if total_time is not None and curr_time - start_time > total_time:
                break

        return query_list, meta_list, result_list, card_dict_list

    def parse_template_id_list(self, template_id_list):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        if template_id_list == "all":
            # 返回所有模版
            template_id_res = deepcopy(list(self.template_meta_dict.keys()))
        elif isinstance(template_id_list, int):
            # 返回指定个数的模板
            template_id_res = deepcopy(list(self.template_meta_dict.keys())[:min(template_id_list, len(self.template_meta_dict))])
        elif isinstance(template_id_list, list):
            # 
            template_id_res = deepcopy(template_id_list)
        else:
            raise TypeError("parse_template_id_list: Unsupported type '{}'.".format(type(template_id_list)))

        # print("parse_template_id_list: template_id_res = {}.".format(template_id_res))
        return template_id_res
    

    def polling_based_workload_generation(self, template_id_list, step, root_config, tree_config, search_config, total_time = None):
        """
        基于轮询的workload生成
    
        Args:
            template_id_list, step, root_config, tree_config, search_config, total_time = None
        Returns:
            query_list:
            meta_list:
            result_list:
            card_dict_list:
        """
        start_time = time.time()

        # 解析模版ID
        template_id_list = self.parse_template_id_list(template_id_list)

        query_list = []
        result_list = []
        meta_list = []
        card_dict_list = []


        for i in range(step):
            idx = i % len(template_id_list)     # 获得当前模版的ID
            # 选择模版
            self.switch_template(template_id=template_id_list[idx])                        
            # 创建新的节点
            new_root_id, flag = self.create_new_root(external_config=root_config, tree_config = tree_config)
            if flag == False:
                # 找不到节点直接跳过
                continue

            # 执行搜索的过程                  
            query_local, meta_local, result_local, card_dict_local = self.begin_search_process(config = search_config)

            query_list.extend(query_local)
            meta_list.extend(meta_local)
            result_list.extend(result_local)
            card_dict_list.extend(card_dict_local)

            curr_time = time.time()
            if total_time is not None and curr_time - start_time > total_time:
                # 超时直接退出
                break

        return query_list, meta_list, result_list, card_dict_list


    def expt_summary(self, query_list, result_list, config):
        """
        {Description}
    
        Args:
            query_list: 查询列表
            result_list: 结果列表
            config: 结果展示的配置
        Returns:
            return1:
            return2:
        """
        pass
