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

from utility import utils
from plan import plan_init, plan_analysis, node_query, advance_search
from utility.utils import set_verbose_path, verbose
from collections import defaultdict

from experiment import forest_exploration
import socket
import psycopg2 as pg
from asynchronous import construct_input, task_management, state_inspection
from estimation import exploration_estimation
from query import ce_injection, query_exploration
from data_interaction import data_management, mv_management
from grid_manipulation import grid_preprocess

# %% 轨迹打印
from utility.utils import trace, SignatureSerializer
from utility import utils
from result_analysis import res_statistics
# %%
# 自己服务器下的配置
from utility.common_config import default_resource_config, dynamic_resource_config

# %%

class ParallelForestExploration(forest_exploration.ForestExplorationExperiment):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, workload, expt_config, expl_estimator = exploration_estimation.DummyEstimator(), \
        resource_config: dict = default_resource_config, max_expl_step = 10, tmpl_meta_path = \
            "/home/lianyuan/Research/CE_Evaluator/intermediate/stats/template_obj/internal/meta_info.json",
        init_query_config = {"target": "under", "min_card": 500, "max_card": 1000000, "mode": "sample-based"}, 
        tree_config = {"max_depth":5, "timeout": 60000}, init_strategy = "multi-loop", split_budget = 100):
        """
        {Description}

        Args:
            workload:
            expt_config: 
            expl_estimator: 
            resource_config: 
            max_expl_step:
            tmpl_meta_path: 模版元信息的保存路径
            init_query_config:
            tree_config:
            init_strategy:
            split_budget:
        """
        self.init_strategy = init_strategy
        self.init_query_config, self.tree_config = init_query_config, tree_config
        # self.split_budget = split_budget
        
        self.db_conn = pg.connect(**construct_input.conn_dict(workload))
        self.inspector = state_inspection.WorkloadInspector(workload=workload, db_conn=self.db_conn)
        
        # ce_str = expt_config.get("ce_handler", "internal")
        ce_str = expt_config.get("ce_handler")
        self.ce_handler, self.ce_str = ce_injection.get_ce_handler_by_name(workload, ce_str), ce_str

        # 创建query_instance所需要的元素，避免db_conn太多的问题
        self.external_dict = {
            "data_manager": data_management.DataManager(wkld_name=workload), 
            "mv_manager": mv_management.MaterializedViewManager(workload=workload),
            "ce_handler": self.ce_handler, 
            "query_ctrl": query_exploration.QueryController(workload=workload), 
            "multi_builder": None
        }

        self.external_dict['bins_builder'] = grid_preprocess.BinsBuilder(workload=workload,
            mv_manager_ref=self.external_dict['mv_manager'], data_manager_ref=self.external_dict['data_manager'])
        
        self.agent = task_management.TaskAgent(workload=workload, inspector=self.inspector)

        # 注册相关的函数信息
        self.agent.load_external_func(self.update_exploration_state, "eval_state")
        self.agent.load_external_func(self.launch_short_task, "short_task")
        self.agent.load_external_func(self.adjust_long_task, "long_task")

        super(ParallelForestExploration, self).__init__(workload, expt_config, tmpl_meta_path)
        
        # 资源调度信息
        self.resource_config = resource_config
        self.expl_estimator = expl_estimator

        # 
        self.sig_serializer = SignatureSerializer()
        
        self.reset_search_state()       # 重置状态
        # 单棵树最大的探索步骤
        self.max_expl_step = max_expl_step
        self.exploration_result = utils.multi_level_dict(level=3)

        # 选择节点模式
        self.set_search_config()    # 默认选择所有的模版
        self.set_selection_mode(new_mode="default")

        self.workload_state_dict = {}

    def update_template_info(self, new_meta_path):
        """
        更新模版的状态，载入新的template信息
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        self.template_meta_path = new_meta_path
        self.template_meta_dict = utils.load_json(new_meta_path)
        
        self.template_manager.template_dict = {}    # 消去template_dict的内容
        self.reset_search_state()

    def reset_search_state(self,):
        """
        重置搜索状态，应用于多次实验的场景
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        self.inspector.reset_monitor_state()    # 更新观测者的信息

        self.root_id_dict = {}
        self.root_id = -1

        self.curr_template_plan = None       # 当前的模版查询计划
        self.curr_search_tree = None         # 当前的搜索树
        
        for k, v in self.template_meta_dict.items():
            self.root_id_dict[k] = {}

        # 模板探索的信息
        self.template_explore_info = []

        # 任务的保存信息
        self.task_info_dict = {}

        # 搜索树的信息保存
        self.tree_info_dict = {
            "tree_ref": {},
            "tree2node": {},
            "node2tree": {}
        }

        # node_signature到node实例引用的字典
        self.node_ref_dict = {}

        # 每一个template对应的最近root
        self.latest_root_dict = {}

        # 长任务调度信息
        """
        {
            "task_signature":{
                "unit_benefit": 
                "state":
            },
            ...
        }
        """
        self.schedule_info_dict = defaultdict(lambda: {"unit_benefit": 0.0, "finished": False})

        self.suspend_list, self.restore_list, self.terminate_list = [], [], []
        # template对应的所有tree的信息
        self.template_tree_info = defaultdict(lambda: {
            "finish_group": set(),  # 已经探索完的树
            "block_group": set(),   # 由于探索未完成而阻塞的树
            "active_group": set()   # 活跃的树，可以进行one_step_search
        })

        #
        self.sig_serializer.reset()

        
    def select_latest_root(self,):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        try:
            root_id = self.latest_root_dict[self.template_plan_id]
            return root_id, self.select_root(root_id)
        except KeyError:
            return -1, None


    def q_error_evaluation(self, mode, true_card, est_card, threshold = 2.5):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        assert mode in ("over-estimation", "under-estimation"), f"q_error_evaluation: mode = {mode}"

        if mode == "over-estimation":
            return ((1.0 + est_card) / (1.0 + true_card)) >= threshold
        elif mode == "under-estimation":
            return ((1.0 + true_card) / (1.0 + est_card)) >= threshold
        

    @utils.timing_decorator
    # def create_new_root(self, external_config = {}, tree_config = {"max_depth":5, "timeout": 60000}, switch_root = True):
    def create_new_root(self, init_query_config, tree_config, switch_root = True):
        # new_root_id = self.root_id + 1  # 新的根节点
        new_root_id = len(self.root_id_dict[self.template_plan_id]) + 1
        
        workload = self.workload

        self.curr_template_plan.grid_info_adjust()  # 调整grid的信息
        self.curr_template_plan.set_ce_handler(external_handler=self.ce_handler)    # 设置基数估计器

        # 最多建n次，失败了则代表创建节点失败，
        # 但是考虑到多轮探索的情况，实际感觉不需要retry，直接失败得了
        max_try_times = 3
        flag = False

        # TODO: 优化root选择的策略
        for i in range(max_try_times):
            selected_query, selected_meta, true_card, estimation_card = \
                self.curr_template_plan.explore_init_query(external_info=init_query_config)        # 获得新的目标查询
            
            if self.q_error_evaluation(self.curr_template_plan.mode, true_card, estimation_card) == False:
                # 探索到的查询q_error过小
                print(f"create_new_root: q_error doesn't reach threshold. true_card = {true_card}. est_card = {estimation_card}.")
                continue
            
            root_query = node_query.get_query_instance(workload=workload, query_meta=selected_meta, \
                ce_handler = self.ce_handler, external_dict=self.external_dict)

            subquery_dict, single_table_dict = \
                self.curr_template_plan.get_plan_cardinalities(in_meta=selected_meta, query_text=selected_query)

            # print("create_new_root: try_time = {}. subquery_dict = {}. single_table_dict = {}.".\
            #       format(i, subquery_dict, single_table_dict))
            
            # query_instance导入真实基数
            root_query.add_true_card(subquery_dict, mode="subquery")
            root_query.add_true_card(single_table_dict, mode="single_table")

            external_info = {
                "query_instance": root_query,
                "selected_tables": self.selected_tables,
                "max_depth": tree_config['max_depth'],     # 最大深度hard-code进去,
                "timeout": tree_config['timeout']          # 查询时间限制在1min
            }

            # 使用高级搜索树，并且设置template_id
            new_search_tree = advance_search.AdvanceTree(external_info=external_info, \
                max_step = self.max_expl_step, template_id=self.template_plan_id, \
                mode=self.curr_template_plan.mode, init_strategy=self.init_strategy)

            if new_search_tree.is_terminate == True:
                # 说明这不是一个好的Tree
                continue
            
            new_search_tree.set_root_id(root_id=new_root_id)    # 设置新的根节点ID
            self.root_id_dict[self.template_plan_id][new_root_id] = new_search_tree
            if switch_root == True:
                self.curr_search_tree = new_search_tree
                self.root_id = new_root_id
            flag = True
            self.latest_root_dict[self.template_plan_id] = new_root_id
            return new_root_id, flag

        self.latest_root_dict[self.template_plan_id] = new_root_id
        return new_root_id, flag
    
    def set_search_config(self, template_id_list = "all"):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        self.template_id_list = self.parse_template_id_list(template_id_list)

        # 绑定template_plan的引用
        for id in self.template_id_list:
            curr_template = self.get_template_by_id(id)
            curr_template.bind_template_plan()

        self.template2tree = {}

    def select_template2explore(self, mode = "random"):
        """
        选择想探索的模版
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        template_id_list = self.template_id_list
        template_id = template_id_list[0]
        if mode == "random":
            template_id = np.random.choice(template_id_list, size=1)[0]
        elif mode == "polling":
            template_id = template_id_list[self.next_idx]
            self.next_idx = (self.next_idx + 1) % len(self.template_id_list)
        elif mode == "epsilon_greedy":
            template_id = self.Epsilon_Greedy_selection()
        elif mode == "MAB":
            template_id = self.Correlated_MAB_selection()
        else:
            raise ValueError("select_template2explore: Unsupported mode({})".format(mode))

        return template_id

    def tree_state_transition(self, template_id, root_id, prev_state = None, curr_state = None):
        """
        树的状态转移
        
        Args:
            template_id: 
            root_id: 
            prev_state: 
            curr_state:
        Returns:
            res1:
            res2:
        """
        # print("tree_state_transition: template_id = {}. root_id = {}. prev_state = {}. curr_state = {}.".\
        #       format(template_id, root_id, prev_state, curr_state))
        # print("tree_state_transition: template_tree_before = {}.".format(self.template_tree_info[template_id]))
        if prev_state is not None:
            prev_key = prev_state + "_group"
            try:
                self.template_tree_info[template_id][prev_key].remove(root_id)
            except KeyError as e:
                pass
            
        curr_key = curr_state + "_group"
        self.template_tree_info[template_id][curr_key].add(root_id)
        # print("tree_state_transition: template_tree_after = {}.".format(self.template_tree_info[template_id]))


    @utils.timing_decorator
    def select_tree2explore(self, template_id)->advance_search.AdvanceTree:
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            tree:
            flag:
        """
        # curr_template: plan_init.TemplatePlan = self.template_id_list[template_id]
        self.switch_template(template_id=template_id)

        latest_id, latest_tree = self.select_latest_root()
        latest_tree: advance_search.AdvanceTree = latest_tree

        def select_next_tree():
            # 选择下一棵树
            active_group = self.template_tree_info[template_id]['active_group']
            if len(active_group) == 0:
                # 
                res_id, flag = self.create_new_root(\
                    self.init_query_config, self.tree_config, switch_root=True)
                if flag == True:
                    # self.template_tree_info[template_id]['active_group'].add(res_id)
                    self.tree_state_transition(template_id=template_id, \
                        root_id=res_id, curr_state='active')
            else:
                # 
                flag = True
                # print("act")
                # res_id = np.random.choice(active_group, 1)[0]
                res_id = np.random.choice(list(active_group))
            return res_id, flag            

        condition = ""
        if latest_tree is None: 
            condition = "case 0"
            root_id, flag = select_next_tree()
            print("select_tree2explore: new_root_id = {}. flag = {}".format(root_id, flag))
            if flag == False:
                # 失败的情况
                return None, flag
            selected_tree = self.select_root(query_id=root_id)
        elif latest_tree.is_terminate == True:
            condition = "case 1"
            # 代表没有新的树，或者已经被探索完了
            self.tree_state_transition(template_id, latest_id, 'active', 'finish')
            # new_root_id, flag = self.create_new_root(switch_root=False)
            root_id, flag = select_next_tree()
            if flag == False:
                # 失败的情况
                return None, flag
            selected_tree = self.select_root(query_id=root_id)
        elif latest_tree.is_blocked == True:
            condition = "case 2"
            # 进入tree阻塞的状态
            self.tree_state_transition(template_id, latest_id, 'active', 'block')
            root_id, flag = select_next_tree()
            if flag == False:
                # 失败的情况
                return None, flag
            selected_tree = self.select_root(query_id=root_id)
        else:
            condition = "case 3"
            selected_tree = latest_tree
            root_id, flag = latest_id, True

        if root_id is not None:
            self.latest_root_dict[self.template_plan_id] = root_id

        print("select_tree2explore: template_id = {}. root_id = {}. condition = {}. flag = {}".\
              format(template_id, root_id, condition, flag))
        
        state_dict = self.template_tree_info[template_id]
        print("select_tree2explore: active_group = {}. block_group = {}. finish_group = {}.".\
              format(state_dict['active_group'], state_dict['block_group'], state_dict['finish_group']))

        # 成功的情况
        return selected_tree, True
    
    def Epsilon_Greedy_selection(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        template_id_list = self.template_id_list

        def get_random_template():
            # 随机选择template
            idx = np.random.randint(low=0, high=len(template_id_list))
            return template_id_list[idx]
        
        def get_best_template():
            # 取平均reward最大的template
            reward_dict = self.exploration_result_aggregation(\
                level="template", mode="p_error")
            # return np.argmax(avg_list)
            reward_dict = self.dict_complement(reward_dict)
            return utils.dict_max(reward_dict)[0]
        
        indicator = np.random.uniform(0, 1)
        if indicator < self.curr_epsilon:
            # 随机选择动作
            template_id = get_random_template()
            print("Epsilon_Greedy_workload_generation: get_random_template. idx = {}.".format(template_id))
        else:
            # 选择average reward最大的动作
            template_id = get_best_template()
            print("Epsilon_Greedy_workload_generation: get_best_template. idx = {}.".format(template_id))

        # 更新epsilon的值
        self.curr_epsilon = self.iter_func(self.curr_epsilon)

        return template_id

    def exploration_result_aggregation(self, level, mode = "count"):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        assert level in ("tree", "template")
        assert mode in ("count", "p_error", "benefit")
        aggr_dict = defaultdict(list)

        for tmpl_id, tmpl_result in self.exploration_result.items():
            for root_id, tree_result in tmpl_result.items():
                for node_id, val in tree_result.items():
                    if level == "template_id":
                        aggr_dict[tmpl_id].append(val)
                    elif level == "tree_id":
                        aggr_dict[(tmpl_id, root_id)].append(val)
        
        in_func = lambda a: a

        def error2benefit(p_error, max_bound = 5.0):
            if p_error >= max_bound:
                return 1.0
            elif p_error <= 1.0:
                return 0.0
            else:
                return (p_error - 1.0) / (max_bound - 1.0)

        if mode == "count":
            in_func = lambda a: len(a)
        elif mode == "p_error":
            in_func = lambda a: np.average([item[1] / item[0] for item in a])
        elif mode == "benefit":
            in_func = lambda a: np.average([error2benefit(item[1] / item[0]) for item in a])

        result_dict = utils.dict_apply(aggr_dict, in_func)
        return result_dict


    def dict_complement(self, existing_dict, default_val = 100):
        key_list = self.template_id_list
        for k in key_list:
            if k not in existing_dict:
                existing_dict[k] = default_val
        return existing_dict

    def Correlated_MAB_selection(self,):
        """
        先用MAB实现一个最简单的
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        # 计算访问次数和累计收益
        benefit_dict = self.exploration_result_aggregation(\
            level="template", mode="benefit")
        count_dict = self.exploration_result_aggregation(\
            level="template", mode="count")
        
        total_count = np.sum(count_dict.values())
        ops_func = lambda reward, count: reward + \
            math.sqrt((2 * math.log(total_count)) / float(count))
        
        ucb_value_dict = utils.dict_apply(utils.dict_concatenate(\
            benefit_dict, count_dict), ops_func)

        ucb_value_dict = self.dict_complement(ucb_value_dict)
        # 选择动作
        print(f"Correlated_MAB_selection: ucb_value_dict = {ucb_value_dict}")
        tmpl_id, value = utils.dict_max(ucb_value_dict)

        # 返回结果
        return tmpl_id

    def set_selection_mode(self, new_mode, config = {}):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        assert new_mode in ("default", "polling_based", "Epsilon_Greedy", "Correlated_MAB"), \
            f"set_selection_mode: new_mode = {new_mode}"

        name_mapping = {
            "default": "random",    # 默认情况下采用随机选择的模式
            "polling_based": "polling",
            "Epsilon_Greedy": "epsilon_greedy",
            "Correlated_MAB": "MAB"
        }

        self.selection_mode = name_mapping[new_mode]   
        
        if new_mode == "polling_based":
            self.next_idx = 0     # 下一个template的id
        elif new_mode == "Epsilon_Greedy":
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
            start_val, end_val = 1.0, 0.1
            total_step = 100
            if config['mode'] == "linear":
                self.iter_func = linear_iter_gen(start_val, end_val, total_step)
            elif config['mode'] == "exponential":
                self.iter_func = exponential_iter_gen(start_val, end_val, total_step)

            self.curr_epsilon = start_val

        elif new_mode == "Correlated_MAB":
            pass
        elif new_mode == "":
            pass



    @utils.timing_decorator
    def start_single_task(self,):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        template_try_max = 10
        flag = False
        for _ in range(template_try_max):
            # 根据不同的mode选择template
            template_id = self.select_template2explore(self.selection_mode) 
            print("start_single_task: template_id = {}".format(template_id))
            
            # 选择node，如果创建失败，则选择其他template
            curr_tree, flag = self.select_tree2explore(template_id)
            if flag == True:
                break
        
        if flag == False:
            print("start_single_task: select_template2explore/select_tree2explore fails")
            return {}
                    
        curr_tree: advance_search.AdvanceTree = curr_tree
        tree_sig = curr_tree.tree_signature
        if tree_sig not in self.tree_info_dict['tree_ref'].keys():
            self.tree_info_dict['tree_ref'][tree_sig] = curr_tree
            self.tree_info_dict['tree2node'][tree_sig] = set()

        # 启动任务，单步搜索
        node_signature, estimate_benefit, new_node = curr_tree.one_step_search()

        if node_signature == "":
            # 单步探索失败的情况
            print("start_single_task: one_step_search fails")
            return {}
        
        self.tree_info_dict['tree2node'][tree_sig].add(node_signature)
        self.tree_info_dict['node2tree'][node_signature] = tree_sig

        # 
        try:
            expected_cost, actual_cost = new_node.expected_cost, new_node.actual_cost
        except AttributeError as e:
            print("start_single_task: meet error. tree_id = {}. node_id = {}.".\
                  format(tree_sig, new_node.node_id))
            raise(e)
        p_error = actual_cost / expected_cost

        # 启动任务
        subquery_res, single_table_res = new_node.extension_ref.\
            true_card_plan_async_under_constaint(proc_num=5)
        
        subquery_cost, single_table_cost = utils.dict_apply(subquery_res, \
            lambda a: a['cost']), utils.dict_apply(single_table_res, lambda a: a['cost'])
        total_cost = sum(subquery_cost.values()) + sum(single_table_cost.values())

        # 获得签名，用于解析路径
        signature = new_node.extension_ref.get_extension_signature()    
        self.inspector.load_card_info(signature=signature, subquery_dict=\
            subquery_res, single_table_dict=single_table_res)

        utils.trace("explore_node_async: template_id = {}. root_id = {}. node_id = {}. ext_signature = {}.".\
                    format(curr_tree.template_id, curr_tree.root_id, new_node.node_id, signature))

        # self.task_info_dict[node_signature] = (estimate_benefit, new_node)
        curr_task_info = {
            "task_signature": signature,
            "state": "activate",
            "process": {}, "query_info": {}, "estimation": {},
            "query_cost": (subquery_cost, single_table_cost),   # 每个查询的预期cost
            "total_cost": total_cost,                           # 预期总的cost
            "total_time": total_cost * self.expl_estimator.get_learned_factor(),       # 预期总的执行时间
            "elapsed_time": 0.0,         # 已经运行的时间
            "start_time": time.time(),   # 
            "end_node": True
        }
        curr_task_info['estimation'] = {
            "benefit": estimate_benefit,
            "expected_cost": expected_cost,
            "actual_cost": actual_cost,
            "p_error": p_error
        }
        self.node_ref_dict[node_signature] = new_node
        self.task_info_dict[node_signature] = curr_task_info

        # 将signature序列化
        self.sig_serializer.add_signature(node_signature)
        return curr_task_info


    @utils.timing_decorator
    def update_tree_state(self, node_signature, subquery_new, single_table_new):
        """
        更新树的状态
        
        Args:
            arg1:
            arg2:
        Returns:
            flag: True代表探索完成，False代表探索未完成
            benefit: 
            cost_true: 
            cost_estimation:
        """
        tree_signature = self.tree_info_dict['node2tree'][node_signature]
        tree_instance: advance_search.AdvanceTree = self.tree_info_dict['tree_ref'][tree_signature]

        external_dict = {
            "subquery": subquery_new,
            "single_table": single_table_new
        }

        # 更新节点的状态
        expl_flag, benefit, cost_true, cost_estimation = tree_instance.\
            update_node_state(node_signature=node_signature, external_info=external_dict)

        template_id, root_id = tree_instance.template_id, tree_instance.root_id
        if expl_flag == True and tree_instance.is_blocked == True:
            # 考虑探索完全的情况，可能从block转换成active
            if tree_instance.root.selectable == True:
                # 根节点重新可选择的状态
                tree_instance.is_blocked = False

                # 允许再次探索
                # print("update_tree_state: template_id = {}. tree_id = {}. from block to active".\
                #       format(template_id, root_id))

                self.tree_state_transition(template_id, root_id, 'block', 'active')
            else:
                # 如果所有节点探索完毕，进入finish状态
                flag = tree_instance.has_explored_all_nodes()
                if flag == True:
                    self.tree_state_transition(template_id, root_id, 'block', 'finish')

        return expl_flag, benefit, cost_true, cost_estimation

    @utils.timing_decorator
    def update_exploration_state(self,):
        """
        更新状态函数，会被agent调用
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        state_dict, query_list, meta_list, result_list, card_dict_list = \
            self.evaluate_running_tasks()
        
        self.resource_allocation(state_dict=state_dict)
        return query_list, meta_list, result_list, card_dict_list

    def add_complete_instance(self, node_sig, cost_true, cost_estimation):
        """
        {Description}
    
        Args:
            node_sig:
            cost_true:
            cost_estimation:
        Returns:
            query:
            meta:
            result:
            card_dict:
        """
        curr_node: advance_search.AdvanceNode = self.node_ref_dict[node_sig]
        ext_ref = curr_node.extension_ref
        query, meta = ext_ref.query_text, ext_ref.query_meta
        p_error = cost_estimation / cost_true
        result = p_error, cost_estimation, cost_true
        card_dict = {
            "true": {
                "subquery": ext_ref.subquery_true,
                "single_table": ext_ref.single_table_true
            },
            "estimation": {
                "subquery": ext_ref.subquery_estimation,
                "single_table": ext_ref.single_table_estimation
            }
        }

        self.exploration_result[curr_node.template_id][curr_node.root_id][curr_node.node_id] = \
            cost_estimation, cost_true
        
        return query, meta, result, card_dict


    @utils.timing_decorator
    def evaluate_running_tasks(self, ):
        """
        更新森林的状态，针对每一个task，需要包含以下的成员信息:

        "node_signature": {
            "task_signature": "",
            "state": "activate|suspend|finish|terminate",
            "process": {
                "pid": {
                    "state": "activate|suspend|finish|terminate"    # 当前状态
                    "key_list": [],         # 对应query的key列表
                    "cpu_time": "",         # 当前的cpu_time
                    "correlated_list": []   # 相关的进程列表
                },
                ...
            },
            "query_info": {
                "key": {
                    "exec_time": "执行时间",
                    "cardinality": "基数大小" 
                }
            },
            "estimation": {
                "missing_subquery": set(),
                "missing_single_table": set(),
                "benefit": "收益",
                "expected_cost": "",
                "actual_cost": "",
                "p_error":
            },
            "total_elapsed_time": ""
        }
        
        Args:
            None
        Returns:
            state_dict: 状态字典
            query_list: 更新的查询列表
            meta_list: 更新的元信息列表
            result_list: 更新的结果列表
            card_dict_list: 更新的基数字典列表
        """
        # 总的信息
        result_dict = self.inspector.get_workload_state()
        self.workload_state_dict = result_dict   # 保存每次的workload字典

        state_dict = {
            'short_proc_num': 0,         # short_proc_num，活跃的短任务进程数
            'new_complete_queries': []   # 新完成的查询信息，包含文本、cost和真实执行时间
        }
        delete_list = []

        query_list, meta_list, result_list, card_dict_list = [], [], [], []

        # 用于打印状态的变量
        self.current_time = time.time()
        elapsed_time = self.current_time - self.start_time
        stable_num, update_num, finish_num = 0, 0, 0   # 分别代表静止、更新以及结束

        for node_sig in self.task_info_dict.keys():
            extension_sig = self.task_info_dict[node_sig]['task_signature']
            try:
                local_dict: dict = result_dict[extension_sig]
            except KeyError as e:
                # 处理找不到结果的情况
                curr_time_str = time.strftime("%H:%M:%S", time.localtime())
                print("evaluate_running_tasks: Unfound extension_sig is {}. curr_time = {}".\
                      format(extension_sig, curr_time_str))
                
                for extension_sig, state in self.inspector.extension_info.items():
                    print("evaluate_running_tasks: extension_sig = {}. complete = {}.".format(extension_sig, state['complete']))
                
                raise(e)
            
            # 完成process的update
            for k, v in local_dict['cpu_time_dict'].items():
                cpu_time, finish = v
                if k not in self.task_info_dict[node_sig]['process'].keys():
                    self.task_info_dict[node_sig]['process'][k] = {
                        "state": "active"
                    }

                self.task_info_dict[node_sig]['process'][k]['cpu_time'] = cpu_time
                state = self.task_info_dict[node_sig]['process'][k]["state"]
                if state == "suspend" or state == "terminate":
                    pass
                elif finish == True:
                    state = "finish"
                elif finish == False:
                    state = "active"
                self.task_info_dict[node_sig]['process'][k]['state'] = state

            # 完成query的update
            new_subquery, new_single_table = 0, 0
            try:
                subquery_cost, single_table_cost = self.task_info_dict[node_sig]['query_cost']
            except KeyError as e:
                print(f"meet KeyError. task_info_dict[node_sig] = {self.task_info_dict[node_sig].keys()}")
                raise e
            
            for k, v in local_dict['subquery_dict'].items():
                if k not in self.task_info_dict[node_sig]['query_info'].keys():
                    new_subquery += 1
                    self.task_info_dict[node_sig]['query_info'][k] = {
                        "exec_time": local_dict['subquery_time'][k],
                        "cardinality": v
                    }
                    try:
                        state_dict['new_complete_queries'].append({
                            "query": (node_sig, k),
                            "time": local_dict['subquery_time'][k],
                            "cost": subquery_cost[k]
                        })
                    except KeyError as e:
                        print(f"evaluate_running_tasks: meet KeyError. subquery_cost.keys = "\
                            f"{subquery_cost.keys()}. local_dict.keys = {local_dict['subquery_dict'].keys()}.")
                        raise e

                else:
                    continue
            
            for k, v in local_dict['single_table_dict'].items():
                if k not in self.task_info_dict[node_sig]['query_info'].keys():
                    new_single_table += 1
                    self.task_info_dict[node_sig]['query_info'][k] = {
                        "exec_time": local_dict['single_table_time'][k],
                        "cardinality": v
                    }
                    state_dict['new_complete_queries'].append({
                        "query": (node_sig, k),
                        "time": local_dict['single_table_time'][k],
                        "cost": single_table_cost[k]
                    })
                else:
                    continue
            
            # 短任务状态更新
            if local_dict['cpu_time_total'] < self.resource_config['long_task_threshold']:
                # 短任务的情况
                state_dict['short_proc_num'] += local_dict['running_num']
            else:
                # 长任务的情况
                pass

            # 搜索树更新
            exploration_flag = False

            if (new_single_table + new_subquery) > 0:
                exploration_flag, benefit, cost_true, cost_estimation = self.update_tree_state(node_signature = node_sig, \
                    subquery_new = local_dict['subquery_dict'], single_table_new = local_dict['single_table_dict'])
                
                local_info = self.task_info_dict[node_sig]
                if benefit > 0:
                    local_info['estimation']['benefit'] = benefit
                    local_info['estimation']['p_error'] = cost_estimation / cost_true
                    local_info['estimation']['expected_cost'] = cost_true
                    local_info['estimation']['actual_cost'] = cost_estimation

                if exploration_flag == False and local_info['end_node'] == True:
                    # 这里的剩余时间计算方法有待更新，根据cost进行修正
                    cost_total = local_info["total_cost"]
                    subquery_cost, single_table_cost = local_info["query_cost"]
                    cost_exist = sum([subquery_cost[k] for k in local_dict['subquery_dict'].keys()]) + \
                        sum([single_table_cost[k] for k in local_dict['single_table_dict'].keys()])

                    local_info['total_time'] = local_dict['cpu_time_total'] * (cost_total / cost_exist)
                    left_time = local_info['total_time'] - local_dict['cpu_time_total']

                    # 长短任务调度信息的更新
                    unit_benefit = self.calculate_unit_benefit(left_time, cost_true, cost_estimation / cost_true)   # 
                    self.schedule_info_dict[node_sig]['unit_benefit'] = unit_benefit

                    # 打印收益计算信息(感觉这段代码位置还得调整)
                    print(f"evaluate_running_tasks: total_time = {local_info['total_time']:.2f}. elapsed_time = {local_dict['cpu_time_total']:.2f}. "\
                        f"left_time = {left_time:.2f}. cost_true = {cost_true:.2f}. p_error = {cost_estimation / cost_true:.2f}.")
                else:
                    # 兼容后面的逻辑，表示优先执行
                    self.schedule_info_dict[node_sig]['unit_benefit'] = 1e7

                # print("evaluate_running_tasks: exploration_flag = {}.".format(exploration_flag))
                if exploration_flag == True:
                    if benefit > 0 and cost_true >= 1.0 and cost_estimation >= 1.0:
                        # 探索完全，添加相应的结果
                        query, meta, result, card_dict = self.add_complete_instance(node_sig, cost_true, cost_estimation)
                        query_list.append(query), meta_list.append(meta)
                        result_list.append(result), card_dict_list.append(card_dict)
                    else:
                        # 2024-03-27: 表示探索异常退出的情况
                        print(f"evaluate_running_tasks: benefit = {benefit:.2f}. cost_true = {cost_true:.2f}. cost_estimation = {cost_estimation:.2f}.")   # 打印相关信息
                    finish_num += 1
                    self.schedule_info_dict[node_sig]['finished'] = True
                else:
                    update_num += 1
                    self.schedule_info_dict[node_sig]['finished'] = False
            else:
                # stable状态下，根据上一次的时间来
                local_info = self.task_info_dict[node_sig]
                left_time = local_info['total_time'] - local_dict['cpu_time_total']
                if left_time < 0.0:
                    left_time = 5.0     # 默认的剩余时间

                cost_true = local_info['estimation']['expected_cost']
                cost_estimation = local_info['estimation']['actual_cost']
                unit_benefit = self.calculate_unit_benefit(left_time, cost_true, cost_estimation / cost_true)
                self.schedule_info_dict[node_sig]['unit_benefit'] = unit_benefit
                stable_num += 1

            # print("evaluate_running_tasks: node_sig = {}. new_single_table = {}. new_subquery = {}. flag = {}.".\
            #       format(node_sig, new_single_table, new_subquery, exploration_flag))

            # 如果更新完全结束了，就从task_info_dict中删除
            if exploration_flag == True:
                delete_list.append(node_sig)
        
        # print("evaluate_running_tasks: len(task_info_dict) = {}. len(result_dict) = {}. delete_list = {}".\
        #       format(len(self.task_info_dict), len(result_dict), delete_list))
        # trace("running_state: delta_time = {:.2f}. total_num = {}. stable_num = {}. update_num = {}. finish_num = {}.".\
        #         format(elapsed_time, len(result_dict), stable_num, update_num, finish_num))
        for node_sig in delete_list:
            del self.task_info_dict[node_sig]
            try:
                del self.schedule_info_dict[node_sig]
            except KeyError as e:
                continue
            
        return state_dict, query_list, meta_list, result_list, card_dict_list
    

    def calculate_unit_benefit(self, left_time, expected_cost, p_error):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        if left_time <= 0:
            left_time = 10.0
            
        return p_error * expected_cost / left_time

    # @utils.timing_decorator
    def resource_allocation(self, state_dict):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        # print("resource_allocation: state_dict = {}.".format(state_dict))
        # 
        config = self.resource_config
        mode = config['assign_mode']
        proc_num = config['proc_num_per_task']

        print("resource_allocation: config['short_proc_num'] = {}. state_dict['short_proc_num'] = {}.".\
              format(config['short_proc_num'], state_dict['short_proc_num']))
        
        if mode == 'stable':
            self.start_task_num = int(np.ceil(config['short_proc_num'] - \
                                    state_dict['short_proc_num']) / proc_num)
            # 
        elif mode == 'dynamic':
            # long_average_benefit = 0.0
            long_avg_benefit = self.long_task_management()
            short_avg_benefit = self.expl_estimator.get_expected_benefit()

            print("resource_allocation: long_avg_benefit = {:.2f}. short_avg_benefit = {:.2f}".\
                  format(long_avg_benefit, short_avg_benefit))
            factor = config['shift_factor']

            # 调整两类任务的进程限制
            if long_avg_benefit > short_avg_benefit and config['short_proc_num'] < config['total_proc_num']:
                config['short_proc_num'] += factor
                config['long_proc_num'] -= factor
            elif long_avg_benefit > short_avg_benefit and config['long_proc_num'] < config['total_proc_num']:
                config['short_proc_num'] -= factor
                config['long_proc_num'] += factor
            else:
                # 不进行调整
                pass

            config = self.resource_config
            self.start_task_num = int(np.ceil(config['short_proc_num'] - \
                                    state_dict['short_proc_num']) / proc_num)
        else:
            raise ValueError("resource_allocation: Unsupported assign_mode({})".format(mode))

    @utils.timing_decorator
    def launch_short_task(self,):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        start_task_num = self.start_task_num
        for id in range(start_task_num):
            print("launch_short_task: id = {}. start_task_num = {}.".format(id, start_task_num))
            self.start_single_task()

    @utils.timing_decorator
    def long_task_management(self,):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        task_list = []
        available_task_num = self.resource_config['long_proc_num']

        for k, v in self.schedule_info_dict.items():
            try:
                task_list.append((v['unit_benefit'], k, v['state']))
            except KeyError as e:
                task_list.append((v['unit_benefit'], k, 'activate'))

        task_list.sort(key = lambda a: a[0], reverse=True)
        self.suspend_list, self.restore_list = [], []

        for idx, task in enumerate(task_list):
            if idx < available_task_num:
                # 设置restore_list
                if task[2] == "suspend":
                    self.schedule_info_dict[task[1]]['state'] = "activate"
                    self.task_info_dict[task[1]]['state'] = "activate"
                    self.restore_list.append(self.task_info_dict[task[1]]['task_signature'])
            else:
                # 设置suspend_list
                if task[2] == "activate":
                    self.schedule_info_dict[task[1]]['state'] = "suspend"
                    self.task_info_dict[task[1]]['state'] = "suspend"
                    self.suspend_list.append(self.task_info_dict[task[1]]['task_signature'])

        func = self.sig_serializer.translate_list
        print(f"long_task_management: restore_list = {func(self.restore_list)}. "\
              f"suspend_list = {func(self.suspend_list)}.")
        
        print("long_task_management: task_list = "\
              f"{[(item[0], self.sig_serializer.translate_signature(item[1]), item[2]) for item in task_list]}.")
        
        # 返回minimum unit benefit
        if len(task_list) != 0:
            print(f"long_task_management: available_task_num = {available_task_num}. len(task_list) = {len(task_list)}. "
                f"benefit = {task_list[min(available_task_num, len(task_list)) - 1][0]:.2f}.")

        if 0 <= available_task_num <= len(task_list) and len(task_list) > 0:
            try:
                return task_list[available_task_num - 1][0]
            except IndexError as e:
                print(f"long_task_management: meet IndexError. available_task_num = {available_task_num}. len(task_list) = {len(task_list)}.")
                raise e
        else:
            return 1.0

    # @utils.timing_decorator
    def adjust_long_task(self,):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        # 更新长任务的状态
        for signature in self.suspend_list:
            self.agent.suspend_instance(signature)

        for signature in self.restore_list:
            self.agent.restore_instance(signature)

        for signature in self.terminate_list:
            self.agent.terminate_instance(signature)
    

    def get_forest_state(self, print_info=True):
        """
        获得整一个森林在搜索最后的状态
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        template_result_dict = {}

        print(self.template_id_list)
        
        for template_id in self.template_id_list:

            template_result_dict[template_id] = {}
            local_dict = self.template_tree_info[template_id]
            root_id_list = list(local_dict['active_group']) + \
                list(local_dict['block_group']) + list(local_dict['finish_group'])

            root_id_list = sorted(set(root_id_list))
            for root_id in root_id_list:
                try:
                    curr_tree: advance_search.AdvanceTree = self.root_id_dict[template_id][root_id]
                except Exception as e:
                    print(f"get_forest_state: meet KeyError. root_id_list = {str(root_id_list)}.")
                    print(f"get_forest_state: meet KeyError. template_id = {str(template_id)}.")
                    print(f"get_forest_state: meet KeyError. available_roots = {self.root_id_dict[template_id].keys()}.")
                    raise e
                
                state_list = curr_tree.get_tree_exploration_state()

                if print_info == True:
                    for node_id, state, p_error in state_list:
                        utils.trace("final_forest_state: template_id = {}. root_id = {}. node_id = {}. state = {}. p_error = {:.2f}".\
                                    format(template_id, root_id, node_id, state, p_error))
                template_result_dict[template_id][root_id] = state_list

        return template_result_dict
    
    def set_experiment_config(self, root_config = {}, tree_config = {}, search_config = {}):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        if root_config is not None and len(root_config) > 1:
            self.root_config = root_config
            self.init_query_config = root_config

        if tree_config is not None and len(tree_config) > 1:
            self.tree_config = tree_config

        if search_config is not None and len(search_config) > 1:
            self.search_config = search_config

    def polling_based_workload_generation(self, template_id_list = None, root_config = {}, \
            tree_config = {}, search_config = {}, total_time = 600):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            query_list: 
            meta_list: 
            result_list: 
            card_dict_list:
        """
        if template_id_list is not None:
            self.set_search_config(template_id_list)

        self.set_experiment_config(root_config, tree_config, search_config)
        self.set_selection_mode("polling_based")
        result = self.parallel_workload_generation(total_time)
        return result
    
    def Epsilon_Greedy_workload_generation(self, template_id_list = None, root_config = {}, \
            tree_config = {}, search_config = {}, total_time = 600, mode = "linear"):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            query_list: 
            meta_list: 
            result_list: 
            card_dict_list:
        """
        if template_id_list is not None:
            self.set_search_config(template_id_list)

        self.set_experiment_config(root_config, tree_config, search_config)
        self.set_selection_mode(new_mode="Epsilon_Greedy", config={"mode": mode})
        result = self.parallel_workload_generation(total_time=total_time)
        return result
    
    def Correlated_MAB_workload_generation(self, template_id_list = None, \
            root_config = {}, tree_config = {}, search_config = {}, total_time = 600):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            query_list: 
            meta_list: 
            result_list: 
            card_dict_list:
        """
        if template_id_list is not None:
            self.set_search_config(template_id_list)

        self.set_experiment_config(root_config, tree_config, search_config)
        self.set_selection_mode(new_mode="Correlated_MAB")
        result = self.parallel_workload_generation(total_time = total_time)
        return result
    

    def parallel_workload_generation(self, total_time, save_result = False, extra_info = {}):
        """
        并行的查询负载生成方法
    
        Args:
            total_time:
            save_result:
            extra_info:
        Returns:
            query_list: 
            meta_list: 
            result_list: 
            card_dict_list:
        """
        self.start_time = time.time()

        query_list, meta_list, result_list, card_dict_list =\
            self.agent.main_process(total_time)
        
        time.sleep(1)   # 暂停一秒等待结束
        
        # 任务结束了，中止所有还在运行的进程
        self.agent.terminate_suspended_process()
        self.agent.terminate_all_instances()
        res_summarizer = res_statistics.ResultSummarizer()

        if save_result == True:
            extra_info['total_time'] = total_time
            extra_info['ce_handler'] = self.ce_str

            f_name = "{}_{}_{}".format(self.workload, time.strftime("%Y%m%d%H%M", \
                time.localtime()), utils.get_signature(str(self), num_out=5))

            res_summarizer.save_result(f_name, query_list, \
                meta_list, result_list, card_dict_list, extra_info)
        
        # 打印forest状态，便于验证结果
        self.get_forest_state()     
        return query_list, meta_list, result_list, card_dict_list
    
# %%
