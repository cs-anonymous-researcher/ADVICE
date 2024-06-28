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
from experiment import stateful_exploration
from estimation import exploration_estimation
from utility.common_config import default_resource_config
from asynchronous import task_management
from collections import defaultdict
from plan import benefit_oriented_search

from utility import utils, common_config
from estimation import state_estimation
from plan import plan_template, node_query

from typing import Set
from collections import deque

# %%

def task_factory():
    return {
        "reward": 1e5,      
        "state": "execute",         # 当前状态
        "prev_state": "execite",    # 上一个状态
        "finished": False,          # 是否结束
        "pause_count": 0            # 暂停计数
    }

# %%

class BenefitOrientedExploration(stateful_exploration.StatefulExploration):
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
        tree_config = {"max_depth":5, "timeout": 60000}, init_strategy = "multi-loop", warm_up_num = 2, 
        card_est_input = "graph_corr_based", action_selection_mode = "local", root_selection_mode = "normal",\
        noise_parameters = None):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        super().__init__(workload, expt_config, expl_estimator, resource_config, max_expl_step, \
            tmpl_meta_path, init_query_config, tree_config, init_strategy, warm_up_num, \
            card_est_input, action_selection_mode, root_selection_mode, noise_parameters)
        
        # 使用新的WarmUper
        self.warm_uper = BenefitOrientedWarmUp(self, warm_up_num)

        # 使用新的agent类型，并且注册新的函数
        self.agent = task_management.AdvancedAgent(workload, self.inspector)

        self.agent.load_external_func(self.update_exploration_state, "eval_state")
        self.agent.load_external_func(self.launch_short_task, "short_task")
        self.agent.load_external_func(self.adjust_long_task, "long_task")
        self.agent.load_external_func(self.construct_candidate_task, "candidate_task")

        # self.root_id_dict = {} # 从[template_id][root_id]到tree instance的字典

        # # 查询树状态的字典，从[template_id][root_id]到tree的状态，但是感觉涉及到状态的维护，不太好实现，可能会增加代码的复杂度。
        # 另一种
        # """
        # example_state = {
        #     "": 
        # }
        # """
        # self.tree_state_dict = defaultdict(dict)

        # 创建面向调度任务信息保存列表
        self.early_termination_count = 10
        config = self.resource_config
        self.total_task_num = np.ceil(config['short_proc_num'] / config['proc_num_per_task'])
        self.unfinish_task_dict = {}
        self.historical_task_dict = {}
        # 用于计算recent_benefit
        self.historical_queue = deque(maxlen=15)

    def recent_historical_benefit(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        if len(self.historical_queue) == 0:
            return 0.0
        else:
            return np.median([item['reward'] for item in self.historical_queue])

    def select_query_tree(self, tmpl_id):
        """
        选择Template下的Query Tree
    
        Args:
            tmpl_id:
            arg2:
        Returns:
            selected_tree: 如果是一个类的实例，表示创建成功，如果是None
            return2:
        """
        # 首先判断是否需要创建新树
        create_flag = False
        tree_benefit_dict = {}
        for root_id, tree_instance in self.root_id_dict[tmpl_id].items():
            tree_instance: benefit_oriented_search.BenefitOrientedTree = tree_instance
            
            tree_benefit_dict[root_id] = (tree_instance.calculate_exploration_benefit(), 
                tree_instance.calculate_historical_benefit(), tree_instance.get_tree_state())

        forest_exploration_benefit = np.max([item[0] for item in tree_benefit_dict.values()])
        forest_historical_benefit = np.average([item[1] for item in tree_benefit_dict.values()])
        available_num = sum([1 for item in tree_benefit_dict.values() if item[2] == "available"])

        print(f"select_query_tree: exploration_benefit = {forest_exploration_benefit:.2f}. "\
              f"historical_benefit = {forest_historical_benefit:.2f}. available_num = {available_num}")

        if forest_exploration_benefit <= forest_historical_benefit or available_num <= 0:
            local_flag, selected_tree = self.construct_new_tree(tmpl_id, tree_config=self.tree_config)
            if local_flag == False:
                return None
            create_flag = True

        # 如果创建了新树就直接选择，否则就直接从已有未完成的树中选benefit最大的
        if create_flag == False:
            selected_idx = sorted(tree_benefit_dict.items(), key=lambda a: a[1][0], reverse=True)[0][0]
            selected_tree = self.root_id_dict[tmpl_id][selected_idx]

        return selected_tree
    
    def explore_query_by_q_error(self, tmpl_id, curr_template_plan: plan_template.TemplatePlan, init_query_config):
        """
        {Description}
    
        Args:
            template_id:
            template:
            init_query_config:
        Returns:
            query_instance:
            exploration_dict:
        """
        workload = self.workload
        
        # TODO: 优化root选择的策略
        selected_query, selected_meta, true_card, estimation_card = \
            curr_template_plan.explore_init_query(external_info=init_query_config)        # 获得新的目标查询
        
        # 屏蔽Q_Error的影响，用做测试
        # 2024-03-19: threshold设为0，放宽init_query的条件
        threshold = 1.0
        if self.q_error_evaluation(curr_template_plan.mode, true_card, estimation_card, threshold) == False:
            # 探索到的查询q_error过小
            print(f"create_new_root: q_error doesn't reach threshold. template_id = {tmpl_id}. threshold = {threshold:.2f}. mode = {curr_template_plan.mode}. true_card = {true_card}. est_card = {estimation_card}.")
            return None, {}
        
        if selected_meta == ([], []) or len(selected_meta[0]) < 1:
            print(f"create_new_root: cannot find valid queries. template_id = {tmpl_id}. query_meta = {selected_meta}.")
            return None, {}
        
        try:
            subquery_dict, single_table_dict = curr_template_plan.get_plan_cardinalities(\
                in_meta=selected_meta, query_text=selected_query)
        except KeyError as e:
            print(f"create_new_root: meet KeyError. query_meta = {selected_meta}. true_card = {true_card}. est_card = {estimation_card}.")
            raise e
        
        root_query = node_query.get_query_instance(workload=workload, query_meta=selected_meta, \
            ce_handler = self.ce_handler, external_dict=self.external_dict)

        # query_instance导入真实基数
        root_query.add_true_card(subquery_dict, mode="subquery")
        root_query.add_true_card(single_table_dict, mode="single_table")

        return root_query, {}
    
    def construct_new_tree(self, tmpl_id, max_try_times = 3, tree_config = {}, mode = "advance"):
        """
        不同于TemplateWarmup和TaskSelector的内容
    
        Args:
            tmpl_id:
            max_try_times:
            tree_config:
            mode: "normal"表示, "advance"表示同时考虑
        Returns:
            flag:
            new_search_tree:
        """
        t1 = time.time()
        flag, new_search_tree = False, None
        curr_template_plan = self.get_template_by_id(tmpl_id)

        # 在template_plan中选择新的grid_plan_id，执行更加多样化的探索
        curr_grid_plan_id = curr_template_plan.select_grid_plan(mode="history")
        curr_template_plan.bind_grid_plan(curr_grid_plan_id)
        curr_template_plan.grid_info_adjust()                                            # 调整grid的信息
        curr_template_plan.set_ce_handler(external_handler=self.ce_handler)     # 设置基数估计器

        try:
            new_root_id = len(self.root_id_dict[tmpl_id]) + 1
        except KeyError as e:
            print(f"construct_new_tree: meet KeyError. root_id_dict = {self.root_id_dict.keys()}. template_meta_dict = {self.template_meta_dict.keys()}")
            raise e
        
        state_manager: state_estimation.StateManager = self.state_manager_dict[tmpl_id]
        t2 = time.time()
        time_list = []

        for idx in range(max_try_times):
            expl_mode, eval_num = common_config.exploration_mode, common_config.exploration_eval_num

            # 2024-03-20: 更新函数调用方法
            if mode == "normal":
                root_query, exploration_info = self.explore_query_by_q_error(tmpl_id, 
                    curr_template_plan, self.init_query_config)
            elif mode == "advance":
                root_query, exploration_info = self.explore_query_by_state_manager(
                    tmpl_id, curr_template_plan, self.init_query_config, expl_mode, eval_num)
            else:
                raise ValueError(f"construct_new_tree: idx = {idx}. invalid mode = {mode}.")
            
            if root_query is None:
                continue

            external_info = {
                "query_instance": root_query,
                "selected_tables": self.selected_tables,
                "max_depth": tree_config['max_depth'],     # 最大深度hard-code进去,
                "timeout": tree_config['timeout']          # 查询时间限制在1min
            }

            # 使用高级搜索树，并且设置template_id
            max_step = 100
            new_search_tree = benefit_oriented_search.BenefitOrientedTree(self.workload, 
                external_info, max_step, tmpl_id, curr_template_plan.mode, 
                self.init_strategy, state_manager, exploration_info)
            
            t8 = time.time()

            # 设置新的根节点ID
            new_search_tree.set_root_id(root_id=new_root_id)    
            self.root_id_dict[tmpl_id][new_root_id] = new_search_tree

            # 20240223: 设置new_root_id，tmpl_id和grid_plan_id的对应关系
            print(f"BenefitOrientedExploration.construct_new_tree: root_id = {new_root_id}. "\
                  f"tmpl_id = {tmpl_id}. grid_plan_id = {curr_grid_plan_id}.")
            self.grid_plan_mapping[(tmpl_id, new_root_id)] = curr_grid_plan_id

            # self.latest_root_dict[tmpl_id] = new_root_id
            # 检测初始树的状态
            tree_state = new_search_tree.get_tree_state()
            flag = (tree_state == "available")
            t9 = time.time()

            if flag == True:
                break
            
        # # 时间处理
        # # 打印循环前的时间
        # print(f"construct_new_tree: t2 - t1 = {t2 - t1:.2f}")
        # # 打印循环时间
        # for idx, (t3, t4, t5) in enumerate(time_list):
        #     print(f"construct_new_tree: idx = {idx}. t4 - t3 = {t4 - t3:.2f}. t5 - t4 = {t5 - t4:.2f}.")
        # # 打印循环后的时间
        # print(f"construct_new_tree: t7 - t6 = {t7 - t6:.2f}. t8 - t7 = {t8 - t7:.2f}. t9 - t8 = {t9 - t8:.2f}.")

        return flag, new_search_tree
    

    def select_candidate_node(self, query_tree: benefit_oriented_search.BenefitOrientedTree):
        """
        选择QueryTree中benefit最大的candidate node
    
        Args:
            query_tree:
            arg2:
        Returns:
            selected_node:
            return2:
        """
        try:
            selected_node = query_tree.select_candidate_node(mode="benefit")
        except AttributeError as e:
            print(f"select_candidate_node: invalid query_tree = {query_tree}.")
            raise e
        
        return selected_node

    def resource_allocation(self, state_dict):
        """
        {Description}
        
        Args:
            state_dict:
            arg2:
        Returns:
            res1:
            res2:
        """
        # print(f"resource_allocation: state_dict = {state_dict}.")
        
        # 处理更新的reward结果
        delete_list = []
        for task_key, t in self.unfinish_task_dict.items():
            if t['state'] == "execute":
                if self.schedule_info_dict[task_key]['finished'] == True:
                    self.historical_task_dict[task_key] = t
                    self.historical_queue.append(t)
                    delete_list.append(task_key)
                    # del self.unfinish_task_dict[task_key]
                    continue
                
                try:
                    t['reward'] = self.schedule_info_dict[task_key]['unit_benefit']
                except Exception as e:
                    print(f"func_name: meet Error. task_key = {task_key}. schedule_info_dict = {self.schedule_info_dict}.")
                    raise e


        # 统一删除结果
        for task_key in delete_list:
            del self.unfinish_task_dict[task_key]

        #
        task_order = sorted(self.unfinish_task_dict.keys(), key=lambda a: 
            self.unfinish_task_dict[a]['reward'], reverse=True)
        recent_benefit = self.recent_historical_benefit()
        active_task_num = 0

        for task_key in task_order:
            t = self.unfinish_task_dict[task_key]
            if (active_task_num >= self.total_task_num or recent_benefit > t['reward']):
                if t['pause_count'] >= self.early_termination_count:
                    t['state'] = 'terminate'
                    self.historical_task_dict[task_key] = self.unfinish_task_dict[task_key]
                    self.historical_queue.append(t)
                    del self.unfinish_task_dict[task_key]
                else:
                    t['prev_state'] = t['state']    # 更新prev_state
                    t['state'] = 'pause'
                    t['pause_count'] += 1
            else:
                t['prev_state'] = t['state']        # 更新prev_state
                t['state'] = 'execute'
                t['pause_count'] = 0

    def update_exploration_state(self,):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        state_dict, query_list, meta_list, result_list, card_dict_list = \
            self.evaluate_running_tasks()
        
        self.resource_allocation(state_dict=state_dict)
        return query_list, meta_list, result_list, card_dict_list


    def launch_short_task(self,):
        """
        生成新的任务

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        total_task_num = 5      # 默认设置为5个任务，之后考虑根据探索状态调整

        if self.warm_uper.is_finish == True:
            # warm up的状态
            self.current_phase = "exploration"
            if common_config.enable_new_template == True:
                self.eval_isolated_status()
                self.eval_template_status()
        else:
            self.current_phase = "warm_up"

        if self.current_phase == "warm_up":
            task_list = self.warm_uper.construct_new_tasks(total_task_num)
        else:
            task_list = []

        left_task_num = total_task_num - len(task_list)

        if (left_task_num > 0 and common_config.warmup_barrier == False) or self.current_phase == "exploration":
            # exploration的状态
            for _ in range(left_task_num):
                # 选择Template
                template_id = self.selector.select_template()
                # 选择QueryTree
                query_tree = self.select_query_tree(template_id)
                if query_tree is None:
                    continue

                # 选择CandidateNode
                candidate_node: benefit_oriented_search.BenefitOrientedNode = \
                    self.select_candidate_node(query_tree)
                
                # 启动CandidateNode对应的任务
                assert candidate_node.node_state == "candidate"
                # 改成保留预定的状态
                candidate_node.set_node_state("reserved")   
                task_list.append((candidate_node, query_tree))

        ref_tree_set: Set[benefit_oriented_search.BenefitOrientedTree] = set()
        for candidate_node, ref_tree in task_list:
            # 更新ref_tree_set
            ref_tree_set.add(ref_tree)

            # 遍历task_list，依次处理任务
            candidate_node: benefit_oriented_search.BenefitOrientedNode = candidate_node
            ref_tree: benefit_oriented_search.BenefitOrientedTree = ref_tree

            try:
                node_signature = candidate_node.signature
            except AttributeError as e:
                print(f"BenefitOrientedExploration.launch_short_task: meet AttributeError. "\
                      f"candidate_node = {candidate_node}. ref_tree = {ref_tree}.")
                raise e

            tree_sig = ref_tree.tree_signature
            if tree_sig not in self.tree_info_dict['tree_ref'].keys():
                self.tree_info_dict['tree_ref'][tree_sig] = ref_tree
                self.tree_info_dict['tree2node'][tree_sig] = set()

            self.tree_info_dict['tree2node'][tree_sig].add(node_signature)
            self.tree_info_dict['node2tree'][node_signature] = tree_sig

            if ref_tree.exploration_info != {} and len(ref_tree.exploration_info) > 1:
                benefit, subquery_res, single_table_res = candidate_node.launch_calculation(
                    proc_num=5, timeout=None, with_card_dict=True)
            else:
                benefit, subquery_res, single_table_res = candidate_node.launch_calculation(proc_num=5, timeout=common_config.warmup_timeout, with_card_dict=True)

            try:
                expected_cost, actual_cost = candidate_node.expected_cost, candidate_node.actual_cost
            except AttributeError as e:
                print("start_single_task: meet error. tree_id = {}. node_id = {}.".\
                    format(tree_sig, candidate_node.node_id))
                raise(e)
            
            p_error = actual_cost / expected_cost

            subquery_cost, single_table_cost = utils.dict_apply(subquery_res, \
                lambda a: a['cost']), utils.dict_apply(single_table_res, lambda a: a['cost'])
            total_cost = sum(subquery_cost.values()) + sum(single_table_cost.values())

            signature = candidate_node.extension_ref.get_extension_signature()    
            self.inspector.load_card_info(signature, subquery_res, single_table_res)
            
            curr_task_info = {
                "task_signature": signature, "state": "activate",
                "process": {}, "query_info": {}, "estimation": {},
                "query_cost": (subquery_cost, single_table_cost),   # 每个查询的预期cost
                "total_cost": total_cost,                           # 预期总的cost
                "total_time": total_cost * self.expl_estimator.get_learned_factor(),       # 预期总的执行时间
                "elapsed_time": 0.0,
                "start_time": time.time(),   # 增加任务的开始时间
                "end_node": candidate_node.is_end()
            }

            curr_task_info['estimation'] = {
                "benefit": benefit,
                "expected_cost": expected_cost,
                "actual_cost": actual_cost,
                "p_error": p_error
            }

            self.node_ref_dict[node_signature] = candidate_node
            self.task_info_dict[node_signature] = curr_task_info

            # 构造新的task加入到
            new_task = task_factory()
            self.unfinish_task_dict[node_signature] = new_task

        for ref_tree in ref_tree_set:
            ref_tree.update_all_node_ids()

        return self.task_info_dict
    
    def adjust_long_task(self,):
        """
        调整位于数据库中的任务
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        # 
        self.suspend_list, self.restore_list, self.terminate_list = [], [], []
        for key, task in self.unfinish_task_dict.items():
            if task['prev_state'] == task['state'] and task['state'] != 'terminate':
                # 状态不改变，直接跳过不做处理
                continue
            elif task['prev_state'] == 'pause' and task['state'] == 'execute':
                # 从暂停到执行
                self.restore_list.append(key)
            elif task['prev_state'] == 'execute' and task['state'] == 'pause':
                # 从执行到暂停
                self.suspend_list.append(key)
            elif task['state'] == 'terminate':
                # 
                self.terminate_list.append(key)

        # 完成实际的操作
        for signature in self.suspend_list:
            self.agent.suspend_instance(signature)

        for signature in self.restore_list:
            self.agent.restore_instance(signature)

        for signature in self.terminate_list:
            self.agent.terminate_instance(signature)


    def construct_candidate_task(self,):
        """
        扫描所有的query tree，尝试构建新的candidate node（需要考虑提前构造query tree的问题）
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        pass

    @utils.timing_decorator
    def evaluate_running_tasks(self, ):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            state_dict:
            query_list:
            meta_list:
            result_list:
            card_dict_list:
        """
        state_dict, query_list, meta_list, result_list, card_dict_list = super().evaluate_running_tasks()

        print("BenefitOrientedExploration.evaluate_running_tasks: print current state")
        # print(f"task_info_dict = {self.task_info_dict}")
        print(f"workload_state_dict = {self.workload_state_dict}")
        print(f"schedule_info_dict = {self.schedule_info_dict}")

        return state_dict, query_list, meta_list, result_list, card_dict_list
# %%

class BenefitOrientedWarmUp(stateful_exploration.TemplateWarmUp):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, explorer: BenefitOrientedExploration, max_tree_num = 3, max_try_times = 3):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        self.explorer = explorer
        self.template_expl_dict = {}
        self.finish_num_dict = {}
        self.max_num_dict = {}      # 表示每个template的warm_up次数
        self.reset_template_info(max_tree_num)

        # self.max_tree_num = max_tree_num
        # self.template_case_cache = defaultdict(list)   # 用于收益匹配的实例

        self.max_try_times = max_try_times
        self.is_finish = False

    def construct_new_tasks(self, task_num = 10, time_limit = 20):
        """
        {Description}

        Args:
            task_num:
            time_limit:
        Returns:
            task_list: (candidate_node, ref_tree)组成的列表
            return2:
        """
        if self.eval_warm_up_state() == True:
            return []
        
        task_list = []

        time_start = time.time()
        for tmpl_id, v in self.template_expl_dict.items():
            last_tree = None
            tree_list: list = v

            if self.finish_num_dict[tmpl_id] >= self.max_num_dict[tmpl_id]:
                continue

            if len(v) == 0:
                flag, last_tree = self.explorer.construct_new_tree(tmpl_id, 
                    self.max_try_times, self.explorer.tree_config, mode="normal")
                if flag == True:
                    tree_list.append(last_tree)
            else:
                # 选择最后一棵树
                last_tree: benefit_oriented_search.BenefitOrientedTree = v[-1]
                tree_state = last_tree.get_tree_state()
                if tree_state == "archived" or tree_state == "failed":
                    if self.finish_num_dict[tmpl_id] + 1 < self.max_num_dict[tmpl_id]:
                        # 树已经被探索完了
                        self.finish_num_dict[tmpl_id] += 1 # 更新已完成模版个数
                        last_tree = self.create_new_tree(tmpl_id)
                        if last_tree is not None:
                            tree_list.append(last_tree)
                    else:
                        last_tree = None

            if last_tree is not None:
                last_tree: benefit_oriented_search.BenefitOrientedTree = last_tree
                local_list = last_tree.enumerate_all_actions()
                task_local = [(node, last_tree) for benefit, node in local_list]
                # print(f"construct_new_tasks: enumerate_all_actions. len(local_list) = {len(local_list)}.")
                if len(task_local) > 0:
                    last_tree.update_all_node_ids()
                    task_list.extend(task_local)
                else:
                    # 感觉这里不应该更新finish_num
                    flag = last_tree.has_explored_all_nodes()
                    print(f"construct_new_tasks: tmpl_id = {last_tree.template_id}. root_id = {last_tree.root_id}. "\
                        f"has_explored_all_nodes = {flag}. len(local_list) = {len(local_list)}.")
            else:
                # print(f"construct_new_tasks: create_new_tree fails. tmpl_id = {tmpl_id}")
                if self.finish_num_dict[tmpl_id] < self.max_num_dict[tmpl_id]:
                    self.finish_num_dict[tmpl_id] += 1
            
            time_end = time.time()
            # early stopping
            if len(task_list) >= task_num or (time_end - time_start > time_limit and len(task_list) >= 1):
                break

        print(f"BenefitOrientedWarmUp.construct_new_tasks: delta_time = {time_end - time_start:.2f}. "\
              f"task_num = {task_num}. len(task_list) = {len(task_list)}")
        
        for idx, (candidate_node, _) in enumerate(task_list):
            assert candidate_node.node_state == "candidate", \
                f"BenefitOrientedWarmUp.construct_new_tasks: idx = {idx}. node_state = {candidate_node.node_state}."

        return task_list

# %%
