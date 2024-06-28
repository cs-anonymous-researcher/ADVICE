#!/usr/bin/env python
# -*- coding: utf-8 -*-


# %%

import numpy as np
import pandas as pd
from os.path import join as p_join
from matplotlib import pyplot as plt
import json, os, time, re, math
import pickle
import shutil

# %%
from experiment import parallel_exploration
from estimation import state_estimation

from utility import utils
from plan import node_query, stateful_analysis, stateful_search, plan_template, plan_init
from utility.utils import set_verbose_path, verbose
from collections import defaultdict

from experiment import forest_exploration, root_evaluation
import socket
import psycopg2 as pg
from asynchronous import construct_input, task_management, state_inspection
from estimation import exploration_estimation
from query import ce_injection, query_exploration
from data_interaction import data_management, mv_management
from grid_manipulation import grid_preprocess

from utility.utils import trace
from utility import utils, common_config, workload_spec
from result_analysis import res_statistics, case_analysis
from utility.common_config import default_resource_config, dynamic_resource_config
from estimation import estimation_interface, external_card_estimation, case_based_estimation
from typing import Any
from copy import deepcopy

# %%

class StatefulExploration(parallel_exploration.ParallelForestExploration):
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
            workload:
            expt_config:
            expl_estimator:
            resource_config:
            max_expl_step:
            tmpl_meta_path:
            init_query_config:
            tree_config:
            init_strategy:
            warm_up_num:
            card_est_input:
        """
        # 打印本次实验的所有入参，用于debug
        print(f"\nStatefulExploration.__init__: workload = {workload}. expt_config = {expt_config}. \nexpl_estimator = {expl_estimator}. "\
              f"resource_config = {resource_config}. \nmax_expl_step = {max_expl_step}. tmpl_meta_path = {tmpl_meta_path}. "\
              f"\ninit_query_config = {init_query_config}. \ntree_config = {tree_config}. init_strategy = {init_strategy}. "\
              f"warm_up_num = {warm_up_num}. \ncard_est_input = {card_est_input}. action_selection_mode = {action_selection_mode}. "\
              f"root_selection_mode = {root_selection_mode}. noise_parameters = {noise_parameters}\n")

        if isinstance(expl_estimator, str):
            expl_estimator = exploration_estimation.name_mapping[expl_estimator]

        super(StatefulExploration, self).__init__(workload, expt_config, expl_estimator, \
            resource_config, max_expl_step, tmpl_meta_path, init_query_config, tree_config, init_strategy)

        # 删除不需要的成员 
        # del self.template_plan_id
        del self.root_id
        del self.curr_search_tree
        del self.curr_template_plan

        # 添加新的成员
        self.state_manager_dict = {}
        self.warm_uper = TemplateWarmUp(self, max_tree_num = warm_up_num)
        self.warm_up_num = warm_up_num
        self.selector = TaskSelector(self)
        self.current_phase = "warm_up"
        self.extra_info_list = []
        self.start_task_num = 5
        self.card_est_input, self.action_selection_mode, self.root_selection_mode, self.noise_parameters = \
            card_est_input, action_selection_mode, root_selection_mode, noise_parameters

        self.time_list = []
        self.grid_plan_mapping = {}

        # root exploration过程中的辅助类字典
        self.root_evaluator_dict = {}

        # 2024-03-12: 用来保存没有对应template的对象
        self.isolated_instance_dict = defaultdict(list)
        self.reference_only_dict, self.reference_signature_dict = defaultdict(list), defaultdict(set)

        self.template_creation_info = {}    # template创建信息

        # 绑定功能函数
        self.agent.load_external_func(self.update_exploration_state, "eval_state")
        self.agent.load_external_func(self.launch_short_task, "short_task")
        self.agent.load_external_func(self.adjust_long_task, "long_task")

        # self.agent.load_external_func(self.exit_func, "exit")

    def get_root_evaluator(self, tmpl_id, grid_plan_instance: plan_template.GridPlan):
        """
        根据tmpl_id以及grid_plan_instance获得对应的root_evaluator
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        grid_id = grid_plan_instance.grid_plan_id
        if (tmpl_id, grid_id) in self.root_evaluator_dict:
            return self.root_evaluator_dict[(tmpl_id, grid_id)]
        else:
            curr_evaluator = root_evaluation.RootSelector(grid_plan_instance)
            self.root_evaluator_dict[(tmpl_id, grid_id)] = curr_evaluator
            return curr_evaluator

    def load_expt_state(self, in_obj, finish_warmup = True):
        """
        会从in_obj中加载的对象: 
            state_manager_dict


        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        if isinstance(in_obj, str):
            state_dict = utils.load_pickle(in_obj)
        elif isinstance(in_obj, dict):
            state_dict = in_obj

        self.state_manager_dict = state_dict['state_manager_dict']

        self.selector.unfinish_list = state_dict['selector'][0]
        self.selector.template_history_benefit = state_dict['selector'][1]
        self.warm_uper.template_expl_dict = state_dict['warm_uper'][0]
        self.warm_uper.finish_num_dict = state_dict['warm_uper'][1]

        self.extra_info_list = state_dict['extra_info_list']
        self.time_list = state_dict['time_list']
        self.grid_plan_mapping = state_dict['grid_plan_mapping']
        # self.exploration_result = state_dict['exploration_result']

        self.state_manager_recouple()
        if finish_warmup == True:
            self.current_phase = "exploration"


    def state_manager_recouple(self,):
        """
        导入state_manager_dict，需要重新绑定explorer_ref
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        for k, v in self.state_manager_dict.items():
            local_manager: state_estimation.StateManager = v
            local_manager.explorer_ref = self

    def state_manager_decouple(self,):
        """
        导出state_manager_dict，需要消去explorer_ref
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        for k, v in self.state_manager_dict.items():
            local_manager: state_estimation.StateManager = v
            del local_manager.explorer_ref

    def dump_expt_state(self, out_path):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """ 
        # 仅仅导出state_manager_dict对象
        # utils.dump_pickle(self.state_manager_dict, out_path)

        self.state_manager_decouple()
        # 导出所有在StatefulExploration中的对象
        state_dict = {
            "state_manager_dict": self.state_manager_dict,
            "selector": (self.selector.unfinish_list, self.selector.template_history_benefit),
            "warm_uper": (self.warm_uper.template_expl_dict, self.warm_uper.finish_num_dict),
            "extra_info_list": self.extra_info_list,
            "time_list": self.time_list,
            "grid_plan_mapping": self.grid_plan_mapping,
            # "exploration_result": self.exploration_result
        }
        utils.dump_pickle(state_dict, out_path)
        self.state_manager_recouple()

    def reset_search_state(self,):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        super().reset_search_state()
        self.state_manager_dict = {}
        if hasattr(self, "warm_uper"):
            self.warm_uper.is_finish = False

    
    def assign_new_instance(self, schema_order: Any, query_meta: Any, card_dict: Any, 
        valid: Any, true_plan: Any | None = None, est_plan: Any | None = None, 
        true_cost: Any | None = None, est_cost: Any | None = None, p_error: Any | None = None,
        is_real = False, mode = "over-estimation"):
        """
        根据schema_order信息将探索实例划到其他state_manager中去
     
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        assert mode in ("over-estimation", "under-estimation")
        tmpl_id, order_key = self.join_order2tmpl_id(schema_order, mode)
        if tmpl_id is not None:
            # 已有对应的template
            try:
                local_manager: state_estimation.StateManager = self.state_manager_dict[tmpl_id]
            except KeyError as e:
                key_list = self.state_manager_dict.keys()
                print(f"assign_new_instance: meet KeyError. tmpl_id = {tmpl_id}. state_manager_dict.keys() = {key_list}.")
                raise e
            
            # if local_manager.mode == mode and is_real == True:
            if is_real == True:
                local_manager.add_new_instance(order_key, query_meta, card_dict, valid, true_plan, 
                    est_plan, true_cost, est_cost, p_error)
            else:
                # print(f"assign_new_instance: target_tmpl_id = {tmpl_id}. mode = {mode}. is_real = {is_real}.")
                local_manager.add_ref_case(query_meta, card_dict, p_error)
        else:
            # 未找到对应的template，先保存结果考虑后续创建
            prefix_num = common_config.template_table_num
            isolated_prefix = tuple(sorted(schema_order[:prefix_num]))
            isolated_key = isolated_prefix + tuple(schema_order[prefix_num:])
            # root_schema = tuple(sorted(self.external_info['query_instance'].query_meta[0]))
            # self.isolated_instance_dict[isolated_key].append((order_key, query_meta, card_dict, 
            #     valid, true_plan, est_plan, true_cost, est_cost, p_error))
            # print(f"assign_new_instance: order_key = {schema_order}. isolated_key = {isolated_key}. p_error = {p_error:.2f}. is_real = {is_real}.")

            # isolated_instance_dict需要包含over-estimation和under-estimation的信息，而不是只有template_key
            self.isolated_instance_dict[(isolated_prefix, mode)].append({
                    "isolated_key": isolated_key, 
                    "query_meta": query_meta, "card_dict": card_dict, 
                    "valid": valid, "true_plan": true_plan, "est_plan": est_plan, 
                    "true_cost": true_cost, "est_cost": est_cost, 
                    "p_error": p_error, "mode": mode, "is_real": is_real
                })
            order_key = isolated_key

        return tmpl_id, order_key

    def join_order2tmpl_id(self, join_order, mode):
        """
        给定join_order，找到特定的template_id，并且返回order_key
    
        Args:
            join_order:
            arg2:
        Returns:
            tmpl_id: 对应的template_id，如果没找到的话，返回None
            order_key: 对应在
        """
        tmpl_id_res, order_key = None, ()

        tmpl_id_dict = {}   # template_id到template_schema_key的映射
        # print(f"join_order2tmpl_id: template_id_list = {self.template_meta_dict.keys()}.")

        for tmpl_id in self.template_meta_dict.keys():
            template: plan_template.TemplatePlan = self.get_template_by_id(tmpl_id)
            tmpl_key = tuple(sorted(template.query_meta[0]))
            tmpl_id_dict[tmpl_id] = tmpl_key, template.mode

        for k, v in tmpl_id_dict.items():
            # print(f"join_order2tmpl_id: key = {k}. value = {v}.")
            tmpl_key, tmpl_mode = v
            key_len = len(tmpl_key)
            if set(join_order[:key_len]) == set(tmpl_key) and tmpl_mode == mode:
                tmpl_id_res = k
                order_key = tmpl_key + tuple(join_order[key_len:])
                break
        
        # print(f"join_order2tmpl_id: join_order = {join_order}. tmpl_id_res = {tmpl_id_res}. order_key = {order_key}.")
        return tmpl_id_res, order_key
    

    def transfer_case(self, src_tmpl_id, src_key, src_index, dst_tmpl_id, dst_key):
        """
        将某些实例从
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        # print(f"transfer_case: src_tmpl_id = {src_tmpl_id}. src_key = {src_key}. "\
        #       f"src_index = {src_index}. dst_tmpl_id = {dst_tmpl_id}. dst_key = {dst_key}.")

        src_manager: state_estimation.StateManager = self.state_manager_dict[src_tmpl_id]
        # 找出相关的case
        item = src_manager.state_dict[src_key].instance_list[src_index]

        if dst_tmpl_id is not None:
            dst_manager: state_estimation.StateManager = self.state_manager_dict[dst_tmpl_id]
            # 添加相关的case
            dst_manager.add_new_instance(dst_key, item['query_meta'], item['card_dict'], 
                item['valid'], item['true_plan'], item['est_plan'], item['true_cost'], 
                item['est_cost'], item['p_error'], reference_only=True)
        else:
            mode = src_manager.mode
            dst_prefix = dst_key[:common_config.template_table_num]
            signature = mv_management.meta_key_repr(item['query_meta'], self.workload)
            if signature not in self.reference_signature_dict[(dst_prefix, mode)]:
                self.reference_signature_dict[(dst_prefix, mode)].add(signature)
                self.reference_only_dict[(dst_prefix, mode)].append((dst_key, item['query_meta'], 
                    item['card_dict'], item['valid'], item['true_plan'], item['est_plan'], 
                    item['true_cost'], item['est_cost'], item['p_error']))


    def eval_isolated_status(self, topk = 3, num_limit = 2):
        """
        评测isolated查询的状态，考虑是否创建新的template
        考虑创建的条件:
            1. 和最优的template进行对比
            2. 和最差的template进行对比
            
        Args:
            topk: 
            create_num: 限制每次创建template的数目
        Returns:
            return1:
            return2:
        """
        if len(self.template_creation_info) > 0:
            # 2024-03-12: 说明此时系统还在创建模板，不再创建新的模板
            print(f"eval_isolated_status: len(template_creation_info) = {len(self.template_creation_info)}. \nkey_list = {self.template_creation_info.keys()}") 
            return {}
        
        tmpl_key_list = []
        error_list = []
        # 评估当前template下的收益情况
        for k, v in self.state_manager_dict.items():
            manager: state_estimation.StateManager = v
            max_p_error = manager.get_exploration_history()['max_p_error']
            error_list.append(max_p_error)
        
        error_list.sort(reverse=True)
        if len(error_list) < topk:
            topk = len(error_list)

        error_template = error_list[topk - 1]      # template探索的error状态
        # error_threshold = error_list[topk - 1]      # template探索的error状态
        max_error_list = [max([item["p_error"] for item in v]) for v in self.isolated_instance_dict.values()]
        max_error_list.sort(reverse=True)

        if num_limit <= len(max_error_list):
            error_instance = max_error_list[num_limit - 1]
        else:
            error_instance = 0.0

        error_threshold = max(error_template, error_instance)

        print(f"eval_isolated_status: topk = {topk}. template_threshold = {error_template:.2f}. "\
              f"instance_threshold = {error_instance:.2f}. error_list = {utils.list_round(max_error_list, 3)}")

        # 评估当前独立实例的收益情况
        for k, v in self.isolated_instance_dict.items():
            local_max_error = max([item["p_error"] for item in v])
            # if local_max_error >= error_threshold:
            # 2024-03-22: 考虑添加额外的条件
            if local_max_error >= error_threshold: 
                print(f"eval_isolated_status: try to create templates. k = {k}. max_error = {local_max_error:.2f}. threshold = {error_threshold:.2f}. "
                      f"len(isolated_case) = {len(v)}. len(ref_case) = {len(self.reference_only_dict[k])}.")
                
                if len(v) >= 5 and len(self.reference_only_dict[k]) > 5:
                    # 创建template时也要考虑
                    tmpl_key_list.append(k)

        creation_info = self.create_new_templates(tmpl_key_list)
        return creation_info
    

    def create_all_templates(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        tmpl_key_mode_list = list(self.isolated_instance_dict.keys())
        print(f"create_all_templates: template_num = {len(tmpl_key_mode_list)}")
        self.create_new_templates(tmpl_key_mode_list)

        
    def create_new_templates(self, tmpl_key_mode_list = [], proc_num = 5):
        """
        创建新的进程单独运行，构造新的模版
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        alias_inverse = workload_spec.get_alias_inverse(self.workload)
        alias_mapping = workload_spec.abbr_option[self.workload]

        def process_meta(in_meta, tmpl_key):
            table_list, filter_list = in_meta
            table_out, filter_out = list(tmpl_key), []
            
            for item in filter_list:
                alias_name, column_name, start_val, end_val = item
                table_name = alias_inverse[alias_name]
                if table_name in table_out:
                    filter_out.append(item)

            # print(f"create_new_templates.process_meta: tmpl_key = {tmpl_key}. table_out = {table_out}. filter_out = {filter_out}.")
            return table_out, filter_out
        
        def locate_columns(in_meta):
            # 
            _, filter_list = in_meta
            selected_columns = []
            alias_set = set()
            for alias_name, column_name, start_val, end_val in filter_list:
                if alias_name in alias_set:
                    continue
                table_name = alias_inverse[alias_name]
                selected_columns.append((table_name, column_name))
                alias_set.add(alias_name)
            return selected_columns
        
        def construct_template_meta(query_meta, selected_columns):
            # 
            schema_list, filter_list = query_meta
            schema_new, filter_new = deepcopy(schema_list), []

            for alias_name, column_name, start_val, end_val in filter_list:
                table_name = alias_inverse[alias_name]
                if (table_name, column_name) in selected_columns:
                    filter_new.append((alias_name, column_name, "placeholder", "placeholder"))
                else:
                    continue
            
            template_meta = schema_new, filter_new
            return template_meta
    
        # 2024-03-12: 从external_dict中取出bins_builder
        bins_builder: grid_preprocess.BinsBuilder = self.external_dict['bins_builder']
        iter_cond_builder = plan_init.IterationConditionBuilder(self.workload, 
            split_budget=self.split_budget, bins_builder=bins_builder)
        res_template_list, cond_bound_dict_list = [], []

        template_key_mapping = {}

        # for tmpl_key, mode in tmpl_key_mode_list:
        for k in tmpl_key_mode_list:
            tmpl_key, mode = k
            # fact_record_list中的query_meta, error
            fact_record_list = []

            # 找到相关的meta信息并进行裁剪
            meta_list, error_list = [item["query_meta"] for item in self.isolated_instance_dict[k]], \
                [item["p_error"] for item in self.isolated_instance_dict[k]]
            
            max_error_idx = np.argmax(error_list)
            meta_truncated = [process_meta(meta, tmpl_key) for meta in meta_list]
            print(f"create_new_templates: tmpl_key = {tmpl_key}.\nmeta_list = {meta_list}.\nerror_list = {utils.list_round(error_list, 2)}.")

            # for query_meta, error in zip(meta_truncated, error_list):
            # 2024-03-13: 只用一条记录作为依据，之后还需要优化
            # fact_record_list.append((meta_truncated[max_error_idx], error_list[max_error_idx]))
            fact_record = (meta_truncated[max_error_idx], error_list[max_error_idx])

            # query_meta, selected_columns, mode
            # mode = "over-estimation"      # 在这里mode需要动态决定

            query_meta = meta_truncated[max_error_idx]
            selected_columns = locate_columns(in_meta=query_meta)

            query_meta = construct_template_meta(query_meta, selected_columns)
            res_template_list.append((query_meta, selected_columns, mode))

            # 更新template_key_mapping
            template_key_mapping[plan_template.template_repr_key(\
                query_meta, selected_columns, alias_mapping)] = k

            # 选好fact_record_list，并获得对应的cond_bound_dict
            #
            # cond_bound_dict = iter_cond_builder.construct_condition_iteration_options(fact_record_list, selected_columns)

            # 
            cond_bound_dict = iter_cond_builder.construct_options_along_single_record(fact_record, selected_columns)
            cond_bound_dict['max_length'] = len(list(cond_bound_dict.values())[0])  # 设置其最大长度
            cond_bound_dict_list.append(cond_bound_dict)
        
        time_start = time.time()
        _, output_path_dict = self.template_manager.create_templates_under_cond_bound(
            res_template_list, cond_bound_dict_list, wait_complete=False)
        time_end = time.time()

        print(f"create_new_templates: create_templates_under_cond_bound. wait_complete = False. delta_time = {time_end - time_start:.2f}")
        # 
        # info_dict = {}
        # for k, v in output_path_dict.items():
        #     info_dict[k] = {
        #         "path": v,
        #         "info": template_dict[k].make_info_dict()
        #     }
        # return template_dict, info_dict

        # 2024-03-12:  利用output_path_dict来更新template_creation_info
        for k, v in output_path_dict.items():
            self.template_creation_info[k] = (v, template_key_mapping[k])

        return self.template_creation_info


    def eval_template_status(self,):
        """
        评估template的状态，如果有新的template就load进来
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        # template_meta_path = ""
        # new_meta_dict: dict = utils.load_json()
        template_info_local = {}
        for k, (out_path, tmpl_key) in self.template_creation_info.items():
            # k表示template key，v表示output_path
            if os.path.isfile(out_path) == True:
                local_template: plan_template.TemplatePlan = utils.load_pickle(out_path)
                self.template_manager.template_dict[k] = local_template
                template_info_local[k] = {
                    "path": out_path,
                    "info": local_template.make_info_dict()
                }
            else:
                pass
        
        if len(template_info_local) == 0:
            return
        
        print(f"eval_template_status: template_info_local = {template_info_local}")
        key_id_mapping = self.append_template_meta(template_info_local)

        print(f"eval_template_status: template_creation_info = {self.template_creation_info.items()}")
        new_id_list = []
        for k, (out_path, tmpl_key) in self.template_creation_info.items():
            if os.path.isfile(out_path) == False:
                print(f"eval_template_status: file_not_exist. k = {k}. out_path = {out_path}")
                continue

            # 构造新的state_manager，并且加入isolated_instance_dict的内容
            instance_list = self.isolated_instance_dict[tmpl_key]

            assert len(instance_list) > 0, f"eval_template_status: tmpl_key = {tmpl_key}. "\
                f"isolated_instance_dict.keys = {self.isolated_instance_dict.keys()}."

            try:
                tmpl_id = key_id_mapping[k]
            except KeyError as e:
                print(f"eval_template_status: meet KeyError. k = {k}. key_id_mapping = {key_id_mapping}.")
                raise e
            
            new_id_list.append(tmpl_id)

            # 这里不一定是over-estimation
            # local_manager = state_estimation.StateManager(self.workload, "over-estimation", tmpl_id, self)
            local_template = self.get_template_by_id(tmpl_id)
            local_template.bind_grid_plan()
            local_template.bind_template_plan()
            local_manager = state_estimation.StateManager(self.workload, local_template.mode, tmpl_id, self)

            for instance in instance_list:
                # 2024-03-12: 添加相关实例
                # order_key, query_meta, card_dict, valid, true_plan, \
                #     est_plan, true_cost, est_cost, p_error = instance
                if instance['is_real'] == True:
                    # 如果是真实case，添加到local_manager中
                    local_manager.add_new_instance(instance['isolated_key'], instance['query_meta'], 
                        instance['card_dict'], instance['valid'], instance['true_plan'], instance['est_plan'], 
                        instance['true_cost'], instance['est_cost'], instance['p_error'])
                else:
                    local_manager.add_ref_case(instance['query_meta'], instance['card_dict'], instance['p_error'])

                # 2024-03-20: 添加到warmup reference中
                # self.warm_uper.template_case_cache[tmpl_id].append((instance['query_meta'], instance['card_dict']))

            # 处理reference_only的问题
            ref_instance_list = self.reference_only_dict[tmpl_key]
            assert len(ref_instance_list) > 0, \
                f"eval_template_status: tmpl_key = {tmpl_key}. reference_only_dict.keys = {self.reference_only_dict.keys()}."

            for item in ref_instance_list:
                dst_key, query_meta, card_dict, valid, true_plan, est_plan, \
                    true_cost, est_cost, p_error = item
                local_manager.add_new_instance(dst_key, query_meta, card_dict, True, 
                    true_plan, est_plan, true_cost, est_cost, p_error, reference_only=True)

            # 2024-03-12: 将local_manager添加到self.state_manager_dict
            print(f"eval_template_status: add_new_manager. tmpl_id = {tmpl_id}. key = {k}.")
            self.state_manager_dict[tmpl_id] = local_manager
            self.root_id_dict[tmpl_id] = {}
            # 删除isolated_instance_dict内容
            del self.isolated_instance_dict[tmpl_key]

        self.template_id_list.extend(new_id_list)
        print(f"eval_template_status: new_id_list = {new_id_list}. template_id_list = {self.template_id_list}.")

        self.warm_uper.append_new_templates(new_id_list)
        # 2024-03-12: 删除已经加载成功的项
        for k in template_info_local.keys():
            del self.template_creation_info[k]

        return self.template_manager
    

    def append_template_meta(self, template_info_dict: dict):
        """
        在已有的基础上添加新的template元信息
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        print("update_template_meta: template_info_dict = {}.".format(template_info_dict))

        # 2024-03-12: 返回
        key_id_mapping = {}
        for k, v in template_info_dict.items():
            if self.template_meta_dict == {}:
                next_template_id = 0
            else:
                next_template_id = max([int(k) for k in self.template_meta_dict.keys()]) + 1    # 需要先转化成数字
            
            key_id_mapping[k] = str(next_template_id) 
            self.template_meta_dict[str(next_template_id)] = {
                "template_key": k, 
                "info": v
            }

        print(f"append_template_meta: key_id_mapping = {key_id_mapping}.")
        # 导出结果
        utils.dump_json(self.template_meta_dict, self.template_meta_path)
        return key_id_mapping


    def set_search_config(self, template_id_list = "all"):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        super().set_search_config(template_id_list)

        for tmpl_id in self.template_id_list:
            local_template_plan = self.get_template_by_id(tmpl_id)
            self.state_manager_dict[tmpl_id] = state_estimation.StateManager(\
                self.workload, local_template_plan.mode, tmpl_id, self)
        
        try:
            self.warm_uper.reset_template_info(self.warm_up_num)
            # self.selector.refresh_state()       # 刷新state
        except AttributeError as e:
            # continue
            pass


    def launch_short_task(self, ):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        if self.warm_uper.is_finish == True:
            # 判断warm_uper是否结束所有任务，更新状态
            self.current_phase = "exploration"
            if common_config.enable_new_template == True:
                self.eval_isolated_status()
                self.eval_template_status()
        else:
            self.current_phase = "warm_up"

        #     # 下面的代码用于直接退出
        #     print("launch_short_task: exit program!")
        #     # 
        #     return self.task_info_dict

        total_task_num = 5      # 默认设置为5个任务，之后考虑根据探索状态调整
        # total_task_num = self.start_task_num         # 
        # print(f"launch_short_task: total_task_old = {total_task_num}.  total_task_new = {self.start_task_num}.")

        if self.current_phase == "warm_up":
            task_list = self.warm_uper.construct_new_tasks(total_task_num)
        else:
            task_list = []

        left_task_num = total_task_num - len(task_list)
        # print(f"launch_short_task: left_task_num = {left_task_num}. total_task_num = {total_task_num}.")

        if (left_task_num > 0 and common_config.warmup_barrier == False) or self.current_phase == "exploration":
            # task_list = self.selector.construct_new_tasks(left_task_num)
            task_list.extend(self.selector.construct_new_tasks(left_task_num))

        for node_signature, estimate_benefit, new_node, ref_tree in task_list:
            if node_signature == "":
                continue
            
            new_node: stateful_search.StatefulNode = new_node
            ref_tree: stateful_search.StatefulTree = ref_tree

            tree_sig = ref_tree.tree_signature
            if tree_sig not in self.tree_info_dict['tree_ref'].keys():
                self.tree_info_dict['tree_ref'][tree_sig] = ref_tree
                self.tree_info_dict['tree2node'][tree_sig] = set()

            self.tree_info_dict['tree2node'][tree_sig].add(node_signature)
            self.tree_info_dict['node2tree'][node_signature] = tree_sig

            try:
                expected_cost, actual_cost = new_node.expected_cost, new_node.actual_cost
            except AttributeError as e:
                print("start_single_task: meet error. tree_id = {}. node_id = {}.".\
                    format(tree_sig, new_node.node_id))
                raise(e)
            
            p_error = actual_cost / expected_cost

            # 2024-03-27: 根据exploration_info选择是否有timeout
            # if ref_tree.exploration_info != {}:
            if ref_tree.exploration_info != {} and len(ref_tree.exploration_info) > 1:
                # 启动任务
                print(f"launch_short_task: ref_tree = exploration mode. exploration_info = {ref_tree.exploration_info}. "\
                      f"root_id = {ref_tree.root_id}. template_id = {ref_tree.template_id}.")
                subquery_res, single_table_res = new_node.extension_ref.\
                    true_card_plan_async_under_constaint(proc_num=5, with_card_dict=True)
            else:
                print(f"launch_short_task: ref_tree = warmup mode. exploration_info = {ref_tree.exploration_info}. "\
                      f"root_id = {ref_tree.root_id}. template_id = {ref_tree.template_id}.")
                subquery_res, single_table_res = new_node.extension_ref.true_card_plan_async_under_constaint(
                    proc_num=5, timeout=common_config.warmup_timeout, with_card_dict=True)
            
            subquery_cost, single_table_cost = utils.dict_apply(subquery_res, \
                lambda a: a['cost']), utils.dict_apply(single_table_res, lambda a: a['cost'])
            total_cost = sum(subquery_cost.values()) + sum(single_table_cost.values())

            # 获得签名，用于解析路径
            signature = new_node.extension_ref.get_extension_signature()    
            self.inspector.load_card_info(signature=signature, subquery_dict=\
                subquery_res, single_table_dict=single_table_res)

            # self.task_info_dict[node_signature] = (estimate_benefit, new_node)

            curr_task_info = {
                "task_signature": signature, "state": "activate",
                "process": {}, "query_info": {}, "estimation": {},
                "query_cost": (subquery_cost, single_table_cost),   # 每个查询的预期cost
                "total_cost": total_cost,                           # 预期总的cost
                "total_time": total_cost * self.expl_estimator.get_learned_factor(),       # 预期总的执行时间
                "elapsed_time": 0.0,
                "start_time": time.time(),   # 增加任务的开始时间
                "end_node": new_node.is_end()
            }

            curr_task_info['estimation'] = {
                "benefit": estimate_benefit,
                "expected_cost": expected_cost,
                "actual_cost": actual_cost,
                "p_error": p_error
            }

            self.node_ref_dict[node_signature] = new_node
            self.task_info_dict[node_signature] = curr_task_info

        return self.task_info_dict

    def exit_func(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        # 用于测试warm_up的逻辑，结束后直接退出
        if self.warm_uper.is_finish == True:
            return True
        else:
            return False

    def exec_before_generation(self,):
        """
        在生成workload前需要执行的内容
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        print(f"exec_before_generation: card_est_input = {self.card_est_input}.")
        assert self.card_est_input in common_config.option_collections

        # 收益估计初始化
        if self.card_est_input == "external":
            config = common_config.option_collections[self.card_est_input]
            external_card_estimation.start_estimation(config['url'], \
                self.workload, self.ce_str, config['model_type'], config['sample_num'])
    
        estimation_interface.set_global_card_estimator(self.card_est_input)

        # 中间节点策略选择初始化
        stateful_analysis.set_global_model(self.action_selection_mode)
        # 根结点探索策略初始化
        common_config.set_tree_mode(self.root_selection_mode)

        # 2024-03-16: 更新函数位置
        if self.noise_parameters is not None:
            # case_based_estimation.set_noise_parameters(*self.noise_parameters)
            common_config.set_noise_parameters(*self.noise_parameters)

        # 查询时间拟合
        # if self.expl_estimator.get_estimator_name == "Linear":
        #     self.expl_estimator.load_query_info()

    def exec_after_generation(self,):
        """
        在生成workload后需要执行的内容
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        print(f"exec_after_generation: card_est_input = {self.card_est_input}.")
        if self.card_est_input == "external":
            config = common_config.option_collections[self.card_est_input]
            external_card_estimation.end_estimation(config['url'])

    def add_historical_data(self, time_list, query_list, meta_list, result_list, card_dict_list):
        """
        每次迭代之后添加历史数据
    
        Args:
            time_list:
            query_list:
            meta_list:
            card_dict_list:
        Returns:
            return1:
            return2:
        """
        for info in time_list:
            self.expl_estimator.load_query_info(info['query'], info['cost'], info['time'])

        for (query, meta, result, card_dict) in \
            zip(query_list, meta_list, result_list, card_dict_list):

            if self.card_est_input == "external":
                # 
                estimation_interface.upload_complete_instance(self.workload, query, meta, card_dict)
            self.expl_estimator.load_task_info(result[0])

    def update_exploration_state(self,):
        """
        更新探索状态
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        state_dict, query_list, meta_list, result_list, card_dict_list = \
            self.evaluate_running_tasks()
        
        self.add_historical_data(state_dict['new_complete_queries'], \
            query_list, meta_list, result_list, card_dict_list)
        self.resource_allocation(state_dict=state_dict)     # 调配资源

        return query_list, meta_list, result_list, card_dict_list

    def update_grid_plan_benefit(self, template_id, grid_plan_id, benefit):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        print(f"update_grid_plan_benefit: template_id = {template_id}. "\
              f"grid_plan_id = {grid_plan_id}. benefit = {benefit:.2f}.")
        
        self.selector.template_history_benefit[template_id].append(benefit)
        target_template = self.get_template_by_id(template_id)
        target_template.set_grid_plan_benefit(grid_plan_id, benefit)

    @utils.timing_decorator
    def update_tree_state(self, node_signature, subquery_new, single_table_new):
        """
        更新树的状态
        
        Args:
            node_signature:
            subquery_new:
            single_table_new:
        Returns:
            flag: True代表探索完成，False代表探索未完成
            benefit: 探索收益
            cost_true: 
            cost_estimation:
        """
        tree_signature = self.tree_info_dict['node2tree'][node_signature]
        tree_instance: stateful_search.StatefulTree = self.tree_info_dict['tree_ref'][tree_signature]

        external_dict = {
            "subquery": subquery_new,
            "single_table": single_table_new
        }

        # 更新节点的状态
        expl_flag, benefit, cost_true, cost_estimation = tree_instance.\
            update_node_state(node_signature=node_signature, external_info=external_dict)

        template_id, root_id = tree_instance.template_id, tree_instance.root_id

        if expl_flag == True and benefit > 0 and cost_estimation > 1.0:
            try:
                grid_plan_id = self.grid_plan_mapping[(template_id, root_id)]
            except KeyError as e:
                print(f"update_tree_state: meet KeyError. grid_plan_mapping = {self.grid_plan_mapping.keys()}.")
                raise e
            self.update_grid_plan_benefit(template_id, grid_plan_id, cost_estimation / cost_true)

        if expl_flag == True and tree_instance.is_blocked == True:
            # 考虑探索完全的情况，可能从block转换成active
            if tree_instance.root.selectable == True:
                # 根节点重新可选择的状态
                tree_instance.is_blocked = False
                # 允许再次探索
                self.tree_state_transition(template_id, root_id, 'block', 'active')
            else:
                # 如果所有节点探索完毕，进入finish状态
                flag = tree_instance.has_explored_all_nodes()
                if flag == True:
                    self.tree_state_transition(template_id, root_id, 'block', 'finish')

        return expl_flag, benefit, cost_true, cost_estimation


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
        self.set_search_config()

        # 打印更新完模版信息后exploration类的状态
        print(f"update_template_info: template_id_list = {self.template_id_list}. "\
              f"finish_num_dict = {self.warm_uper.finish_num_dict.keys()}. template_expl_dict = {self.warm_uper.template_expl_dict.keys()}.")

    def evaluate_warmup(self, total_time, out_path = "state_manager_dict.pkl"):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        time_start = time.time()
        common_config.set_warmup_barrier(True)      # 设置预热屏障
        self.agent.load_external_func(self.exit_func, "exit")

        query_list, meta_list, result_list, card_dict_list = \
            self.parallel_workload_generation(total_time)
        time_end = time.time()

        print(f"evaluate_warmup: delta_time = {time_end - time_start: .2f}.")
        self.agent.load_external_func(lambda: False, "exit")
        # self.dump_expt_state(out_path)

        return query_list, meta_list, result_list, card_dict_list


    def save_template_state(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        template_dir_path = os.path.dirname(self.template_meta_path)
        dir_name = os.path.basename(template_dir_path)
        parent_path = os.path.dirname(template_dir_path)
        template_zip_path = p_join(parent_path, f"{dir_name}.zip")

        print(f"save_template_state: template_dir_path = {template_dir_path}. template_zip_path = {template_zip_path}.")

        if os.path.isfile(template_zip_path):
            os.remove(template_zip_path)    # 删除原来的zip
        shutil.make_archive(p_join(parent_path, f"{dir_name}"), format="zip", base_dir=template_dir_path)

        return template_dir_path

    def restore_template_state(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        template_dir_path = os.path.dirname(self.template_meta_path)
        dir_name = os.path.basename(template_dir_path)
        parent_path = os.path.dirname(template_dir_path)
        template_zip_path = p_join(parent_path, f"{dir_name}.zip")

        print(f"restore_template_state: template_dir_path = {template_dir_path}. template_zip_path = {template_zip_path}.")

        shutil.rmtree(template_dir_path)            # 删除当前的目录
        shutil.unpack_archive(template_zip_path, extract_dir="/")
        # shutil.unpack_archive(template_zip_path)    # 用zipfile还原目录
        

    def evaluate_exploration(self, total_time, in_path = "state_manager_dict.pkl"):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        # 2024-03-18: 考虑到template创建的情况，探索前保存template_obj的状态，探索后再restore回去
        self.warm_uper.is_finish = True
        self.load_expt_state(in_path)

        # 手动加一下manager_ref
        for k, v in self.state_manager_dict.items():
            local_manager: state_estimation.StateManager = v
            local_manager.add_manager_ref()

        self.restore_template_state()
        self.save_template_state()
        query_list, meta_list, result_list, card_dict_list = \
            self.parallel_workload_generation(total_time)
        
        # self.restore_template_state()
        self.reset_search_state()

        return query_list, meta_list, result_list, card_dict_list

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
        # 2024-03-22: 调整template构建的时间限制
        common_config.merge_time_limit = 30

        self.exec_before_generation()
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
        #
        print(f"parallel_workload_generation: template_id_list = {self.template_id_list}.")
        self.get_forest_state()
        self.exec_after_generation()
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

        # 调用父类函数
        query, meta, result, card_dict = \
            super().add_complete_instance(node_sig, cost_true, cost_estimation)
        
        self.time_list.append(self.task_info_dict[node_sig]['start_time'])
        curr_node: stateful_search.StatefulNode = self.node_ref_dict[node_sig]
        self.extra_info_list.append((curr_node.template_id, \
            curr_node.root_id, curr_node.extend_table))
        
        return query, meta, result, card_dict
    

    def get_extra_info(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        return self.extra_info_list

    def stateful_workload_generation(self, template_id_list = None, root_config = {}, \
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
        result = self.parallel_workload_generation(total_time)
        return result

    def create_query_on_template(self, template: plan_template.TemplatePlan, init_query_config):
        """
        {Description}
    
        Args:
            template:
            init_query_config:
        Returns:
            query_text: 
            query_meta: 
            card_dict:
        """
        t1 = time.time()
        template.set_ce_handler(self.ce_handler)
        # selected_query, selected_meta, true_card, estimation_card = \
        #     template.explore_init_query(external_info=init_query_config)        # 获得新的目标查询
        selected_query, selected_meta, true_card, estimation_card = \
            template.explore_init_query_multiround(external_info=init_query_config, update=True)        # 采用新的查询探索模式

        if self.q_error_evaluation(template.mode, true_card, estimation_card) == False:
            # 探索到的查询q_error过小
            print(f"create_query_on_template: q_error doesn't reach threshold. mode = {template.mode}. true_card = {true_card}. est_card = {estimation_card}.")
            # 返回一个无效的查询
            return "", ([], []), {}

        t2 = time.time()
        subquery_true, single_table_true = \
            template.get_plan_cardinalities(in_meta=selected_meta, query_text=selected_query)
        
        t3 = time.time()
        root_query = node_query.get_query_instance(workload=self.workload, query_meta=selected_meta, \
            ce_handler = self.ce_handler, external_dict=self.external_dict)
        
        t4 = time.time()
        root_query.add_true_card(subquery_true, mode = "subquery")
        root_query.add_true_card(single_table_true, mode = "single_table")

        # print(f"create_query_on_template: t2 - t1 = {t2 - t1:.2f}. t3 - t2 = {t3 - t2:.2f}. t4 - t3 = {t4 - t3:.2f}.")
        print(f"create_query_on_template: true_card = {true_card}. estimation_card = {estimation_card}. "
              f"q_error = {max(true_card / (estimation_card + 1.0), estimation_card / (true_card + 1.0)): .2f}.")

        return node_query.construct_instance_element(root_query)
    

    def explore_query_by_state_manager(self, template_id, template: plan_template.TemplatePlan, \
            init_query_config, mode, eval_num):
        """
        根据state_manager来探索收益
    
        Args:
            template_id:
            template:
            init_query_config:
            mode:
            eval_num:
        Returns:
            query_instance: 
            exploration_dict:
        """
        state_manager: state_estimation.StateManager = self.state_manager_dict[template_id]
        curr_max_error = state_manager.get_exploration_history()['max_p_error']  # 当前最大的误差

        def score_func(query_meta, card_dict):
            # print(f"score_func: meta = {query_meta}")
            # print(f"score_func: card_dict = {card_dict}")

            # 利用state_manager计算当前实例的预期分数
            tree_result = state_manager.infer_new_init_benefit(query_meta, card_dict)   # 获得基于历史数据的匹配结果
            if self.selector.estimation_valid_check(tree_result) == False:
                # 引用无效
                # print(f"explorate_query_on_template.score_func: estimation_valid_check is False. tree_result = {tree_result}.")
                print(f"explorate_query_on_template.score_func: estimation_valid_check is False.")
                return 0.0, {}

            result_list, max_error, max_path = state_estimation.tree_result_filter(tree_result, curr_max_error * 0.6)
            # return result_list, max_error, max_path
            if len(result_list) > 0:
                path_list = self.selector.extract_path_list(result_list)
            else:
                path_list = [max_path[0], ]
            
            # 添加额外的信息
            ref_index_dict = {}
            for schema_order in path_list:
                if schema_order in tree_result:
                    ref_index_dict[schema_order] = list(set(tree_result[schema_order][1]))
                else:
                    print(f"schema_order = {schema_order}. tree_result.keys = {tree_result.keys()}.")
                    
            info_dict = { "path_list": path_list, "ref_index_dict": ref_index_dict }
            print(f"score_func: info_dict = {info_dict}")

            return max_error, info_dict

        return self.explorate_query_on_template(template_id, \
            template, init_query_config, mode, eval_num, score_func)
    

    # def explorate_query_by_case_cache(self, template_id, template: plan_template.TemplatePlan, \
    #         init_query_config, mode, eval_num, case_cache):
    #     """
    #     {Description}
    
    #     Args:
    #         arg1:
    #         arg2:
    #     Returns:
    #         return1:
    #         return2:
    #     """
    #     benefit_estimator = case_based_estimation.CaseBasedEstimator(self.workload)
    #     benefit_estimator.instance_list = case_cache
    #     alias_inverse = workload_spec.get_alias_inverse(self.workload)
    #     schema_list = template.query_meta[0]
    #     schema_num = len(schema_list)

    #     def score_func(query_meta, card_dict):
    #         try:
    #             result_list = benefit_estimator.eval_new_init(query_meta, card_dict, with_noise = False)
    #         except TypeError as e:
    #             print(f"explorate_query_by_case_cache.score_func: meet TypeError. query_meta = {query_meta}.\ncard_dict = {card_dict}.")
    #             raise e
            
    #         max_error = 0.0
    #         for (query_meta, card_dict), idx in result_list:
    #             # 2024-03-20: 确保schema order匹配的情况下选择error最大的
    #             local_analyzer = case_analysis.CaseAnalyzer(\
    #                 "", query_meta, (), card_dict, self.workload)

    #             # 2024-03-20: 采用analyze_result相近的逻辑
    #             if mode == "over-estimation":
    #                 flag, jo_list = local_analyzer.get_plan_join_order(mode="true")
    #             else:
    #                 flag, jo_list = local_analyzer.get_plan_join_order(mode="estimation")

    #             jo_list = [alias_inverse[alias] for alias in jo_list]
    #             if flag == False:
    #                 continue
    #             else:
    #                 if set(jo_list[:schema_num]) != set(schema_list):
    #                     print(f"explorate_query_by_case_cache: set1 = {set(jo_list[:schema_num])}. set2 = {set(schema_list)}.")
    #                     # join_prefix不匹配
    #                     continue
    #                 else:
    #                     # 更新max_error
    #                     max_error = max(max_error, local_analyzer.p_error)
    #         info_dict = {}    # 不设置info_dict
    #         return 0.0, info_dict

    #     return self.explorate_query_on_template(template_id, \
    #         template, init_query_config, mode, eval_num, score_func)
    
    @utils.timing_decorator
    def explorate_query_on_template(self, template_id, template: plan_template.TemplatePlan, \
            init_query_config, mode, eval_num, score_func):
        """
        在单个模版上考虑不同instance收益，最后选择最优的
    
        Args:
            template_id:
            template: 
            init_query_config:
            mode: 
            eval_num:
            score_func:
        Returns:
            query_instance: 
            exploration_dict: 
        """
        # state_manager: state_estimation.StateManager = self.state_manager_dict[template_id]
        template.set_ce_handler(self.ce_handler)

        inf = 1e11
        lower_bound = init_query_config.get("min_card", -inf)
        upper_bound = init_query_config.get("max_card", inf)
        test_num = init_query_config.get("num", 20)

        query_list, meta_list, label_list, estimation_list = \
            template.generate_random_samples(lower_bound, upper_bound, test_num)

        
        assert template.mode in ("over-estimation", "under-estimation")
        if template.mode == "over-estimation":
            cmp_func = lambda a, b: b / (a + 1.0)
        elif template.mode == "under-estimation":
            cmp_func = lambda a, b: a / (b + 1.0)

        error_list = [cmp_func(true_card, est_card) for \
            true_card, est_card in zip(label_list, estimation_list)]

        # 获得当前的根结点评测器
        root_evaluator = self.get_root_evaluator(template_id, template.curr_plan)
        root_evaluator.num_limit = eval_num
        assert mode in ("random", "greedy", "card-split", "meta-split", "hybrid-split", "feedback-based")

        if mode == "random":
            query_candidates, meta_candidates, error_candidates = \
                root_evaluator.random_split(query_list, meta_list, label_list, error_list)
        elif mode == "greedy":
            query_candidates, meta_candidates, error_candidates = \
                root_evaluator.naive_greedy_split(query_list, meta_list, label_list, error_list)
        elif mode == "card-split":
            # 基于基数进行划分
            query_candidates, meta_candidates, error_candidates = root_evaluator.card_based_split(\
                query_list, meta_list, label_list, error_list, common_config.card_distance)
        elif mode == "meta-split":
            # 基于meta进行划分
            query_candidates, meta_candidates, error_candidates = root_evaluator.meta_based_split(\
                query_list, meta_list, label_list, error_list, common_config.meta_distance)
        elif mode == "hybrid-split":
            # 综合前两者信息划分
            query_candidates, meta_candidates, error_candidates = root_evaluator.hybrid_split(\
                query_list, meta_list, label_list, error_list, common_config.card_distance,
                common_config.meta_distance, common_config.hybrid_alpha)
        elif mode == "feedback-based":
            # 基于历史feedback进行选择
            raise NotImplementedError("explorate_query_on_template: feedback-based has not beed implemented")

        # 考虑ExternalCaseMatcher生成额外的case
        state_manager: state_estimation.StateManager = self.state_manager_dict[template_id]

        card_dict_list = [template.get_plan_cardinalities(in_meta=query_meta, query_text=query_text) \
            for query_text, query_meta in zip(query_list, meta_list)]
        
        instance_input = list(zip(meta_list, card_dict_list))
        selected_index = state_manager.case_matcher.case_candidates_match(instance_input, out_num = 3)

        query_extra, meta_extra, error_extra = utils.list_index_batch([query_list, meta_list, error_list], selected_index)
        query_candidates.extend(query_extra)
        meta_candidates.extend(meta_extra)
        error_candidates.extend(error_extra)

        def construct_root(query_text, query_meta):
            # 这一步有并行优化的空间
            query_instance = node_query.get_query_instance(workload=self.workload, query_meta=query_meta, \
                ce_handler = self.ce_handler, external_dict=self.external_dict)
            
            subquery_true, single_table_true = template.get_plan_cardinalities(\
                in_meta=query_meta, query_text=query_text)

            # 添加真实基数
            query_instance.add_true_card(subquery_true, mode="subquery")
            query_instance.add_true_card(single_table_true, mode="single_table")

            return query_instance
        
        root_candidates = [construct_root(query_text, query_meta) for \
            query_text, query_meta in zip(query_candidates, meta_candidates)]
        card_dict_candidates = [node_query.construct_instance_element(item)[2] for item in root_candidates]

        assert len(meta_candidates) == len(card_dict_candidates), \
            f"explorate_query_on_template: len(meta_candidates) = {len(meta_candidates)}. len(card_dict_candidates) = {len(card_dict_candidates)}."
        
        if len(meta_candidates) == 0:
            # 如果结果为空，则直接退出
            print(f"explorate_query_on_template: failed! cannot find valid queries. template_id = {template_id}.")
            return None, {}
        
        try:
            composite_list = [score_func(meta, card_dict) for \
                meta, card_dict in zip(meta_candidates, card_dict_candidates)]
            score_candidates, info_candidates = zip(*composite_list)
        except TypeError as e:
            print(f"explorate_query_on_template: meet TypeError. composite_list = {composite_list}.")
            raise e

        assert len(query_candidates) == len(meta_candidates) == len(root_candidates) == \
            len(error_candidates) == len(score_candidates) == len(info_candidates), \
            f"len1 = {len(query_candidates)}. len2 = {len(meta_candidates)}. len3 = {len(root_candidates)}. "\
            f"len4 = {len(error_candidates)}. len5 = {len(score_candidates)}. len6 = {len(info_candidates)}."
        
        instance_candidates = list(zip(query_candidates, meta_candidates, \
            root_candidates, error_candidates, score_candidates, info_candidates))
        
        # 
        # print(f"explorate_query_on_template: template_id = {template_id}. grid_id = "\
        #       f"{template.curr_plan.grid_plan_id}. max_error = {curr_max_error:.2f}.")
        
        for instance in instance_candidates:
            print(f"explorate_query_on_template: error = {instance[3]:.2f}. score = {instance[4]:.2f}.")

        # 打印
        instance_sorted = sorted(instance_candidates, key=lambda a: a[-2], reverse=True)
        best_case = instance_sorted[0]
        
        # 选择最好的样本输出
        # return query_text, query_meta, card_dict
        return best_case[2], best_case[5]

    
    def create_new_root(self, tmpl_id, init_query_config, tree_config, max_try_times = 1):
        """
        {Description}
        20231111: 添加时间打印的具体信息，检查问题到底出在哪了
    
        Args:
            tmpl_id: 
            init_query_config: 
            tree_config: 树的相关配置
            max_try_times: 
        Returns:
            root_id:
            flag:
        """
        t1 = time.time()
        # print(f"create_new_root: tmpl_id = {tmpl_id}. init_query_config = {init_query_config}. tree_config = {tree_config}.")
        new_root_id = len(self.root_id_dict[tmpl_id]) + 1
        workload = self.workload

        curr_template_plan: plan_template.TemplatePlan = self.get_template_by_id(id=tmpl_id)
        
        # 20240223新增
        curr_grid_plan_id = curr_template_plan.select_grid_plan(mode="history")
        curr_template_plan.bind_grid_plan(curr_grid_plan_id)

        curr_template_plan.grid_info_adjust()                                   # 调整grid的信息
        curr_template_plan.set_ce_handler(external_handler=self.ce_handler)     # 设置基数估计器

        t2 = time.time()
        # 最多建n次，失败了则代表创建节点失败，
        # 但是考虑到多轮探索的情况，实际感觉不需要retry，直接失败得了
        flag = False
        
        # TODO: 优化root选择的策略
        for i in range(max_try_times):
            t3 = time.time()
            selected_query, selected_meta, true_card, estimation_card = \
                curr_template_plan.explore_init_query(external_info=init_query_config)        # 获得新的目标查询
            
            # 屏蔽Q_Error的影响，用做测试
            # 2024-03-19: threshold设为0，放宽init_query的条件
            threshold = 1.0
            if self.q_error_evaluation(curr_template_plan.mode, true_card, estimation_card, threshold) == False:
                # 探索到的查询q_error过小
                print(f"create_new_root: q_error doesn't reach threshold. template_id = {tmpl_id}. threshold = {threshold:.2f}. mode = {curr_template_plan.mode}. true_card = {true_card}. est_card = {estimation_card}.")
                t4 = t5 = t6 = t7 = time.time()
                continue
            
            if selected_meta == ([], []) or len(selected_meta[0]) < 1:
                print(f"create_new_root: cannot find valid queries. template_id = {tmpl_id}. query_meta = {selected_meta}.")
                continue

            t4 = time.time()
            try:
                subquery_dict, single_table_dict = curr_template_plan.get_plan_cardinalities(\
                    in_meta=selected_meta, query_text=selected_query)
            except KeyError as e:
                print(f"create_new_root: meet KeyError. query_meta = {selected_meta}. true_card = {true_card}. est_card = {estimation_card}.")
                raise e

            
            t5 = time.time()
            root_query = node_query.get_query_instance(workload=workload, query_meta=selected_meta, \
                ce_handler = self.ce_handler, external_dict=self.external_dict)

            # query_instance导入真实基数
            root_query.add_true_card(subquery_dict, mode="subquery")
            root_query.add_true_card(single_table_dict, mode="single_table")

            external_info = {
                "query_instance": root_query,
                "selected_tables": self.selected_tables,
                "max_depth": tree_config['max_depth'],     # 最大深度hard-code进去,
                "timeout": tree_config['timeout']          # 查询时间限制在1min
            }
            t6 = time.time()

            # 使用高级搜索树，并且设置template_id
            new_search_tree = stateful_search.StatefulTree(workload=self.workload, \
                external_info=external_info, max_step = self.max_expl_step, template_id=tmpl_id, \
                mode=curr_template_plan.mode, init_strategy=self.init_strategy, \
                state_manager_ref = self.state_manager_dict[tmpl_id])

            t7 = time.time()
            new_search_tree.set_root_id(root_id=new_root_id)    # 设置新的根节点ID
            self.root_id_dict[tmpl_id][new_root_id] = new_search_tree

            # 20240223: 更新设置grid_plan_id
            self.grid_plan_mapping[(tmpl_id, new_root_id)] = curr_grid_plan_id

            flag = True
            break
            # self.latest_root_dict[tmpl_id] = new_root_id
            # return new_root_id, flag

        # time_list = [t1, t2, t3, t4, t5, t6, t7]
        # time_diff = np.diff(time_list)
        # print(f"create_new_root: t2 - t1 = {time_diff[0]:.2f}. t3 - t2 = {time_diff[1]:.2f}. t4 - t3 = {time_diff[2]:.2f}. "\
        #       f"t5 - t4 = {time_diff[3]:.2f}. t6 - t5 = {time_diff[4]:.2f}. t7 - t6 = {time_diff[5]:.2f}.")
        
        self.latest_root_dict[tmpl_id] = new_root_id
        return new_root_id, flag
    
    def create_new_root_by_case(self, tmpl_id, init_query_config, tree_config, max_try_times = 1):
        """
        参考case_cache创建root
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        t1 = time.time()
        # print(f"create_new_root: tmpl_id = {tmpl_id}. init_query_config = {init_query_config}. tree_config = {tree_config}.")
        new_root_id = len(self.root_id_dict[tmpl_id]) + 1
        curr_template_plan: plan_template.TemplatePlan = self.get_template_by_id(id=tmpl_id)
        
        # 20240223新增
        curr_grid_plan_id = curr_template_plan.select_grid_plan(mode="history")
        curr_template_plan.bind_grid_plan(curr_grid_plan_id)

        curr_template_plan.grid_info_adjust()                                   # 调整grid的信息
        curr_template_plan.set_ce_handler(external_handler=self.ce_handler)     # 设置基数估计器

        t2 = time.time()
        # 最多建n次，失败了则代表创建节点失败，
        # 但是考虑到多轮探索的情况，实际感觉不需要retry，直接失败得了
        flag = False
        
        # TODO: 优化root选择的策略
        for i in range(max_try_times):
            # mode, eval_num = "card-split", 5    # 改为从全局配置中传入
            mode, eval_num = common_config.exploration_mode, 0

            # 2024-03-20: 更新函数调用方法
            root_query, exploration_info = self.explore_query_by_state_manager(\
                 tmpl_id, curr_template_plan, init_query_config, mode, eval_num)

            if root_query is None:
                continue

            external_info = {
                "query_instance": root_query,
                "selected_tables": self.selected_tables,
                "max_depth": tree_config['max_depth'],     # 最大深度hard-code进去,
                "timeout": tree_config['timeout']          # 查询时间限制在1min
            }

            # 使用高级搜索树，并且设置template_id
            new_search_tree = stateful_search.StatefulTree(workload = self.workload, \
                external_info=external_info, max_step = self.max_expl_step, template_id=tmpl_id, \
                mode=curr_template_plan.mode, init_strategy=self.init_strategy, \
                state_manager_ref=self.state_manager_dict[tmpl_id], exploration_info=exploration_info)

            t8 = time.time()

            # 设置新的根节点ID
            new_search_tree.set_root_id(root_id=new_root_id)    
            self.root_id_dict[tmpl_id][new_root_id] = new_search_tree

            # 20240223: 设置new_root_id，tmpl_id和grid_plan_id的对应关系
            print(f"TaskSelector.construct_new_tree: root_id = {new_root_id}. "\
                  f"tmpl_id = {tmpl_id}. grid_plan_id = {curr_grid_plan_id}.")
            self.grid_plan_mapping[(tmpl_id, new_root_id)] = curr_grid_plan_id

            self.latest_root_dict[tmpl_id] = new_root_id
            res_task = new_search_tree.one_step_search()    # 进行单步探索
            res_task = res_task + (new_search_tree,)

            t9 = time.time()

            flag = res_task[0] != ""

            if flag == True:
                break

        self.latest_root_dict[tmpl_id] = new_root_id
        return new_root_id, flag
    

# %%

class TemplateWarmUp(object):
    """
    模版探索的预热模块，每一次都尽可能的启动新任务，并保存结果作为参照

    Members:
        field1:
        field2:
    """

    def __init__(self, explorer: StatefulExploration, max_tree_num = 3):
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

        self.is_finish = False

    def reset_template_info(self, max_tree_num = 3):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        self.template_expl_dict = {}
        for tmpl_id in self.explorer.template_id_list:
            self.template_expl_dict[tmpl_id] = list()
            self.finish_num_dict[tmpl_id] = 0
            self.max_num_dict[tmpl_id] = max_tree_num

    def append_new_templates(self, new_id_list: list, max_tree_num = 5):
        """
        {Description}
    
        Args:
            new_id_list:
            arg2:
        Returns:
            return1:
            return2:
        """
        print(f"append_new_templates: new_id_list = {new_id_list}. max_tree_num = {max_tree_num}.")

        for tmpl_id in new_id_list:
            self.template_expl_dict[tmpl_id] = list()
            self.finish_num_dict[tmpl_id] = 0
            self.max_num_dict[tmpl_id] = max_tree_num

        self.is_finish = False

    def eval_warm_up_state(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        flag = True
        num_list = []
        for tmpl_id, num in self.finish_num_dict.items():
            # flag = flag and (num >= self.max_tree_num)
            flag = flag and (num >= self.max_num_dict[tmpl_id])
            tree_num = len(self.template_expl_dict[tmpl_id])
            num_list.append((tmpl_id, num, tree_num, self.max_num_dict[tmpl_id]))
        
        print(f"eval_warm_up_state: (tmpl_id, finish_num, tree_num, limit_num) = {num_list}")

        self.is_finish = flag
        return self.is_finish

    def construct_new_tasks(self, task_num = None, time_limit = 20):
        """
        {Description}

        Args:
            task_num: 
            time_limit: 
        Returns:
            task_list: 包含(node_signature, estimate_benefit, new_node, tree_reference)这些成员
            return2:
        """
        if self.eval_warm_up_state() == True:
            return []

        if task_num is None:
            task_num = 1000

        task_list = []

        time_start = time.time()

        # 遍历整一个template_expl_dict
        for tmpl_id, v in self.template_expl_dict.items():
            last_tree = None
            tree_list: list = v

            # print(f"construct_new_tasks: tmpl_id = {tmpl_id}. len(v) = {len(v)}. finish_num = {self.finish_num_dict[tmpl_id]}. max_num = {self.max_num_dict[tmpl_id]}.")            
            # if self.finish_num_dict[tmpl_id] >= self.max_tree_num:
            if self.finish_num_dict[tmpl_id] >= self.max_num_dict[tmpl_id]:
                continue

            if len(v) == 0:
                last_tree = self.create_new_tree(tmpl_id)
                if last_tree is not None:
                    tree_list.append(last_tree)
            else:
                # 选择最后一棵树
                last_tree = v[-1]
                flag = last_tree.has_explored_all_nodes()
                if flag == True:
                    # if self.finish_num_dict[tmpl_id] + 1 < self.max_tree_num:
                    if self.finish_num_dict[tmpl_id] + 1 < self.max_num_dict[tmpl_id]:
                        # 树已经被探索完了
                        self.finish_num_dict[tmpl_id] += 1 # 更新已完成模版个数
                        last_tree = self.create_new_tree(tmpl_id)
                        if last_tree is not None:
                            tree_list.append(last_tree)
                    else:
                        last_tree = None

            if last_tree is not None:
                last_tree: stateful_search.StatefulTree = last_tree
                local_list = last_tree.enumerate_all_actions()
                # print(f"construct_new_tasks: enumerate_all_actions. len(local_list) = {len(local_list)}.")
                if len(local_list) > 0:
                    task_list.extend(local_list)
                else:
                    # 感觉这里不应该更新finish_num
                    # if self.finish_num_dict[tmpl_id] + 1 < self.max_num_dict[tmpl_id]:
                    #     self.finish_num_dict[tmpl_id] += 1
                    flag = last_tree.has_explored_all_nodes()
                    print(f"construct_new_tasks: tmpl_id = {last_tree.template_id}. root_id = {last_tree.root_id}. "\
                        f"has_explored_all_nodes = {flag}. len(local_list) = {len(local_list)}.")
            else:
                # 
                # print(f"construct_new_tasks: create_new_tree fails. tmpl_id = {tmpl_id}")
                # if self.finish_num_dict[tmpl_id] < self.max_tree_num:
                if self.finish_num_dict[tmpl_id] < self.max_num_dict[tmpl_id]:
                    self.finish_num_dict[tmpl_id] += 1

            time_end = time.time()
            # early stopping
            if len(task_list) >= task_num or (time_end - time_start > time_limit and len(task_list) >= 1):
                break

        print(f"TemplateWarmUp.construct_new_tasks: delta_time = {time_end - time_start:.2f}. task_num = {task_num}. len(task_list) = {len(task_list)}")
        return task_list
    
    @utils.timing_decorator
    def create_new_tree(self, tmpl_id):
        """
        {Description}
    
        Args:
            tmpl_id:
            arg2:
        Returns:
            return1:
            return2:
        """
        # time_start = time.time()
        # expl = self.explorer

        # # warm up阶段不需要额外的探索信息
        # new_root_id, flag = expl.create_new_root(tmpl_id, \
        #     expl.init_query_config, expl.tree_config)

        # time_end = time.time()
        # print(f"TemplateWarmUp.create_new_tree: delta_time = {time_end-time_start:.2f}. "\
        #       f"template_id = {tmpl_id}. root_id = {new_root_id}. flag = {flag}.")
        # if flag == False:
        #     return None
        
        # new_tree = expl.root_id_dict[tmpl_id][new_root_id]
        # return new_tree
        # if len(self.template_case_cache[tmpl_id]) == 0:
        #     return self.create_new_tree_by_error(tmpl_id)
        # else:
        #     return self.create_new_tree_by_case(tmpl_id)
        return self.create_new_tree_by_error(tmpl_id)
        
    def create_new_tree_by_error(self, tmpl_id):
        """
        {Description}
    
        Args:
            tmpl_id:
            arg2:
        Returns:
            return1:
            return2:
        """
        time_start = time.time()
        expl = self.explorer

        # warm up阶段不需要额外的探索信息
        new_root_id, flag = expl.create_new_root(tmpl_id, \
            expl.init_query_config, expl.tree_config)

        time_end = time.time()
        print(f"TemplateWarmUp.create_new_tree: delta_time = {time_end-time_start:.2f}. "\
              f"template_id = {tmpl_id}. root_id = {new_root_id}. flag = {flag}.")
        if flag == False:
            return None
        
        new_tree = expl.root_id_dict[tmpl_id][new_root_id]
        return new_tree

    def create_new_tree_by_case(self, tmpl_id):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        time_start = time.time()
        expl = self.explorer

        # warm up阶段不需要额外的探索信息
        new_root_id, flag = expl.create_new_root_by_case(tmpl_id, expl.init_query_config, 
            expl.tree_config)
        # expl.create_new_root()
        time_end = time.time()
        print(f"TemplateWarmUp.create_new_tree: delta_time = {time_end-time_start:.2f}. "\
              f"template_id = {tmpl_id}. root_id = {new_root_id}. flag = {flag}.")
        if flag == False:
            return None
        
        new_tree = expl.root_id_dict[tmpl_id][new_root_id]
        return new_tree



# %%

class TaskSelector(object):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, explorer: StatefulExploration):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        self.explorer = explorer
        self.unfinish_list = []
        # self.template_history_benefit = {}
        self.template_history_benefit = defaultdict(list)

        # 2024-03-16: 初始化时不刷新状态
        # self.refresh_state()

    @utils.timing_decorator
    def construct_new_tasks(self, task_num):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            task_list: 包含(node_signature, estimate_benefit, new_node, ref_tree)这些成员
            return2:
        """
        time_start = time.time()

        # print(f"construct_new_tasks: task_num = {task_num}.")
        task_list = []
        expl = self.explorer

        for task_id in range(task_num):
            if self.unfinish_tree_available() == True:
                # 优先处理未结束的任务
                flag = True
                res_task = self.process_unfinish_task()
            else:
                # 尝试构造新的任务
                flag, res_task = self.construct_new_tree(tree_config = expl.tree_config)

            if flag:
                task_list.append(res_task)

        time_end = time.time()
        print(f"TaskSelector.construct_new_tasks: delta_time = {time_end - time_start:.2f}. task_num = {task_num}. len(task_list) = {len(task_list)}")
        return task_list
    

    # @utils.timing_decorator
    def unfinish_tree_available(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            flag:
            return2:
        """
        if len(self.unfinish_list) == 0:
            return False
        else:
            for stateful_tree in self.unfinish_list:
                stateful_tree: stateful_search.StatefulTree = stateful_tree
                if stateful_tree.is_blocked == False and stateful_tree.is_terminate == False:
                    return True
            return False
    
    @utils.timing_decorator
    def process_unfinish_task(self,):
        """
        处理未完成的任务
    
        Args:
            arg1:
            arg2:
        Returns:
            node_signature: 
            estimate_benefit: 
            new_node: 
            stateful_tree:
        """
        for stateful_tree in self.unfinish_list:
            stateful_tree: stateful_search.StatefulTree = stateful_tree
            if stateful_tree.is_blocked == False and stateful_tree.is_terminate == False:
                # 探索到未被阻塞的树
                node_signature, estimate_benefit, new_node = \
                    stateful_tree.one_step_search()
                break

        return node_signature, estimate_benefit, new_node, stateful_tree

    def result_aggregation(self, tree_result: dict):
        """
        {Description}
        
        Args:
            tree_result:
            arg2:
        Returns:
            out_p_error:
            res2:
        """
        out_p_error = 0.0
        for schema_order, candidate_list in tree_result.items():
            for item in candidate_list:
                if item[0] == "invalid_extend":
                    # 
                    out_p_error = 10.0
                    # out_p_error = 100.0
                else:
                    if item[2] > out_p_error:
                        out_p_error = item[2]
        return out_p_error
    
    
    def extract_path_list(self, result_list):
        """
        提取合法的路径列表
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        path_set = set()
        
        for schema_order, _ in result_list:
            path_set.add(schema_order)

        return list(path_set)

    
    def estimation_valid_check(self, tree_result: dict):
        """
        判断当前的任务是否成立，如果flag为False，说明没有任何新的增益
    
        Args:
            tree_result: 
            arg2:
        Returns:
            flag: 是否有效
        """
        flag = False

        for schema_order, (candidate_list, _) in tree_result.items():
            # print("estimation_valid_check: ")
            if len(candidate_list) > 0:
                flag = True
        return flag
    

    @utils.timing_decorator
    def exploit_template(self, tmpl_id, max_try_times = 3):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        expl = self.explorer
        curr_template_plan = expl.get_template_by_id(tmpl_id)
        curr_template_plan.grid_info_adjust()                                   # 调整grid的信息
        curr_template_plan.set_ce_handler(external_handler=expl.ce_handler)     # 设置基数估计器

        state_manager: state_estimation.StateManager = expl.state_manager_dict[tmpl_id]
        result_list = []

        for idx in range(max_try_times):
            query_text, query_meta, card_dict = expl.create_query_on_template(\
                curr_template_plan, expl.init_query_config)

            tree_result = state_manager.infer_new_init_benefit(query_meta, card_dict)
            print(f"exploit_template: tmpl_id = {tmpl_id}. idx = {idx}. tree_result = {tree_result}")
            result_list.append(tree_result)
            
        return result_list

    @utils.timing_decorator
    def construct_new_tree(self, max_try_times = 3, tree_config = {}):
        """
        {Description}
        
        Args:
            max_try_times:
            tree_config:
        Returns:
            flag: 表示是否构建成功
            res_task: 包含(node_signature, estimate_benefit, new_node, tree_reference)的元组
        """
        tree_mode = common_config.tree_mode
        assert tree_mode in ("advance", "normal")
        if tree_mode == "advance":
            return self.construct_new_tree_advance(max_try_times, tree_config)
        elif tree_mode == "normal":
            return self.construct_new_tree_normal(max_try_times, tree_config)
        
    @utils.timing_decorator
    def construct_new_tree_advance(self, max_try_times = 3, tree_config = {}):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            flag: 表示是否构建成功
            res_task: 包含(node_signature, estimate_benefit, new_node, tree_reference)的元组
        """
        t1 = time.time()
        tmpl_id = self.select_template()
        flag, expl = False, self.explorer
        curr_template_plan = expl.get_template_by_id(tmpl_id)

        # 在template_plan中选择新的grid_plan_id，执行更加多样化的探索
        curr_grid_plan_id = curr_template_plan.select_grid_plan(mode="history")
        curr_template_plan.bind_grid_plan(curr_grid_plan_id)
        curr_template_plan.grid_info_adjust()                                   # 调整grid的信息
        curr_template_plan.set_ce_handler(external_handler=self.explorer.ce_handler)     # 设置基数估计器
        try:
            new_root_id = len(expl.root_id_dict[tmpl_id]) + 1
        except KeyError as e:
            print(f"construct_new_tree: meet KeyError. root_id_dict = {expl.root_id_dict.keys()}. template_meta_dict = {expl.template_meta_dict.keys()}")
            raise e
        
        res_task = "", 0.0, None, None
        state_manager: state_estimation.StateManager = expl.state_manager_dict[tmpl_id]
        t2 = time.time()

        time_list = []

        for idx in range(max_try_times):
            # mode, eval_num = "card-split", 5    # 改为从全局配置中传入
            mode, eval_num = common_config.exploration_mode, common_config.exploration_eval_num

            # exploration_info = { "path_list": path_list }
            # 2024-03-20: 更新函数调用方法
            # root_query, exploration_info = expl.explorate_query_on_template(\
            #     tmpl_id, curr_template_plan, expl.init_query_config, mode, eval_num)
            root_query, exploration_info = expl.explore_query_by_state_manager(\
                 tmpl_id, curr_template_plan, expl.init_query_config, mode, eval_num)
            
            if root_query is None:
                continue

            external_info = {
                "query_instance": root_query,
                "selected_tables": expl.selected_tables,
                "max_depth": tree_config['max_depth'],     # 最大深度hard-code进去,
                "timeout": tree_config['timeout']          # 查询时间限制在1min
            }

            # 使用高级搜索树，并且设置template_id
            # new_search_tree = stateful_search.StatefulTree(workload = expl.workload, \
            #     external_info=external_info, max_step = expl.max_expl_step, template_id=tmpl_id, \
            #     mode=curr_template_plan.mode, init_strategy=expl.init_strategy, \
            #     state_manager_ref=state_manager, exploration_info=exploration_info)

            new_search_tree = stateful_search.StatefulTree(workload = expl.workload, \
                external_info=external_info, max_step = 100, template_id=tmpl_id, \
                mode=curr_template_plan.mode, init_strategy=expl.init_strategy, \
                state_manager_ref=state_manager, exploration_info=exploration_info)
            

            self.unfinish_list.append(new_search_tree)  # 添加新的树到未完成列表中

            t8 = time.time()

            # 设置新的根节点ID
            new_search_tree.set_root_id(root_id=new_root_id)    
            expl.root_id_dict[tmpl_id][new_root_id] = new_search_tree

            # 20240223: 设置new_root_id，tmpl_id和grid_plan_id的对应关系
            print(f"TaskSelector.construct_new_tree: root_id = {new_root_id}. "\
                  f"tmpl_id = {tmpl_id}. grid_plan_id = {curr_grid_plan_id}.")
            self.explorer.grid_plan_mapping[(tmpl_id, new_root_id)] = curr_grid_plan_id

            expl.latest_root_dict[tmpl_id] = new_root_id
            res_task = new_search_tree.one_step_search()    # 进行单步探索
            res_task = res_task + (new_search_tree,)

            t9 = time.time()

            flag = res_task[0] != ""

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
        return flag, res_task

    @utils.timing_decorator
    def construct_new_tree_normal(self, max_try_times = 3, tree_config = {}):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            flag: 表示是否构建成功
            res_task: 包含(node_signature, estimate_benefit, new_node, tree_reference)的元组
        """

        t1 = time.time()
        tmpl_id = self.select_template()
        flag, expl = False, self.explorer
        curr_template_plan = expl.get_template_by_id(tmpl_id)

        # 在template_plan中选择新的grid_plan_id，执行更加多样化的探索
        curr_grid_plan_id = curr_template_plan.select_grid_plan(mode="history")
        curr_template_plan.bind_grid_plan(curr_grid_plan_id)
        curr_template_plan.grid_info_adjust()                                   # 调整grid的信息
        curr_template_plan.set_ce_handler(external_handler=self.explorer.ce_handler)     # 设置基数估计器
        try:
            new_root_id = len(expl.root_id_dict[tmpl_id]) + 1
        except KeyError as e:
            print(f"construct_new_tree: meet KeyError. root_id_dict = {expl.root_id_dict.keys()}. template_meta_dict = {expl.template_meta_dict.keys()}")
            raise e
        
        res_task = "", 0.0, None, None
        state_manager: state_estimation.StateManager = expl.state_manager_dict[tmpl_id]
        curr_max_error = state_manager.get_exploration_history()['max_p_error']  # 当前最大的误差

        t2 = time.time()

        time_list, instance_cache = [], []
        for idx in range(max_try_times):
            t3 = time.time()
            query_text, query_meta, card_dict = expl.create_query_on_template(\
                curr_template_plan, expl.init_query_config)
            
            if (query_text.startswith("SELECT") or query_text.startswith("select")) == False:
                continue
            t4 = time.time()

            tree_result = state_manager.infer_new_init_benefit(query_meta, card_dict)   # 获得基于历史数据的匹配结果
            if self.estimation_valid_check(tree_result) == True:
                # 通过当前最大的error来过滤结果
                # result_list = self.result_filter(tree_result, curr_max_error)
                # 放宽要求(即使放宽了有时候也不一定满足条件，所以考虑都失败的情况下选最优的)
                result_list, max_error, max_path = state_estimation.tree_result_filter(tree_result, curr_max_error * 0.6)

                if len(result_list) > 0:
                    path_list = self.extract_path_list(result_list)

                    # 打印path_list的信息
                    workload = self.explorer.workload
                    print(f"TaskSelector.construct_new_tree: tmpl_id = {tmpl_id}. root_id = {new_root_id}. "\
                          f"path_list = {[utils.construct_alias_list(path, workload) for path in path_list]}")
                    flag = True
                    # 2024-03-30: 结果去重
                    exploration_info = { "path_list": path_list, 
                        "ref_index_dict": {schema_order: list(set(tree_result[schema_order][1])) \
                            for schema_order in path_list if schema_order in tree_result.keys()}}
                    print(f"construct_new_tree_normal: info_dict = {exploration_info}")
                    # 只有结果大于0的时候才是合法的
                    break
                else:
                    # init_root不符合要求的情况
                    instance_cache.append((query_text, query_meta, card_dict, max_error, max_path))     # 临时保存结果
                    print(f"TaskSelector.construct_new_tree: instance_max_error = {max_error:.2f}. curr_max_error = {curr_max_error:.2f}.")
            else:
                print(f"TaskSelector.construct_new_tree: estimation_valid_check = False. tree_result = {tree_result}")
                # raise ValueError(f"TaskSelector.construct_new_tree: estimation_valid_check = False. tree_result = {tree_result}")
            
            t5 = time.time()
            time_list.append((t3, t4, t5))

        # 如果找不到比当前error大的结果，从cache中选取生成结果中最好的
        if flag == False and len(instance_cache) > 0:
            flag = True
            instance_cache.sort(key=lambda item: item[3], reverse=True)
            query_text, query_meta, card_dict, max_error, max_path = instance_cache[0]
            exploration_info = { "path_list": [max_path[0], ] }
            print("TaskSelector.construct_new_tree: select best one since no valid case. "\
                  f"curr_max_error = {curr_max_error:.2f}. instance_error = {max_error:.2f}.")
        
        if flag == True:
            t6 = time.time()
            root_query = node_query.get_query_instance(workload=expl.workload, query_meta=query_meta, \
                ce_handler = expl.ce_handler, external_dict=expl.external_dict)
            subquery_dict, single_table_dict, _, _ = utils.extract_card_info(card_dict)

            # query_instance导入真实基数
            root_query.add_true_card(subquery_dict, mode="subquery")
            root_query.add_true_card(single_table_dict, mode="single_table")

            t7 = time.time()

            external_info = {
                "query_instance": root_query,
                "selected_tables": expl.selected_tables,
                "max_depth": tree_config['max_depth'],     # 最大深度hard-code进去,
                "timeout": tree_config['timeout']          # 查询时间限制在1min
            }

            # 使用高级搜索树，并且设置template_id
            # new_search_tree = stateful_search.StatefulTree(workload = expl.workload, \
            #     external_info=external_info, max_step = expl.max_expl_step, template_id=tmpl_id, \
            #     mode=curr_template_plan.mode, init_strategy=expl.init_strategy, \
            #     state_manager_ref=state_manager, exploration_info=exploration_info)

            new_search_tree = stateful_search.StatefulTree(workload = expl.workload, \
                external_info=external_info, max_step = 100, template_id=tmpl_id, \
                mode=curr_template_plan.mode, init_strategy=expl.init_strategy, \
                state_manager_ref=state_manager, exploration_info=exploration_info)
            
            self.unfinish_list.append(new_search_tree)  # 添加新的树到未完成列表中

            t8 = time.time()

            # 设置新的根节点ID
            new_search_tree.set_root_id(root_id=new_root_id)    
            expl.root_id_dict[tmpl_id][new_root_id] = new_search_tree

            # 20240223: 设置new_root_id，tmpl_id和grid_plan_id的对应关系
            print(f"TaskSelector.construct_new_tree: root_id = {new_root_id}. "\
                  f"tmpl_id = {tmpl_id}. grid_plan_id = {curr_grid_plan_id}.")
            self.explorer.grid_plan_mapping[(tmpl_id, new_root_id)] = curr_grid_plan_id

            expl.latest_root_dict[tmpl_id] = new_root_id
            res_task = new_search_tree.one_step_search()
            res_task = res_task + (new_search_tree,)

            t9 = time.time()

            flag = res_task[0] != ""
        else:
            t6 = t7 = t8 = t9 = time.time()

        # # 时间处理
        # # 打印循环前的时间
        # print(f"construct_new_tree: t2 - t1 = {t2 - t1:.2f}")
        # # 打印循环时间
        # for idx, (t3, t4, t5) in enumerate(time_list):
        #     print(f"construct_new_tree: idx = {idx}. t4 - t3 = {t4 - t3:.2f}. t5 - t4 = {t5 - t4:.2f}.")
        # # 打印循环后的时间
        # print(f"construct_new_tree: t7 - t6 = {t7 - t6:.2f}. t8 - t7 = {t8 - t7:.2f}. t9 - t8 = {t9 - t8:.2f}.")
        return flag, res_task

    # @utils.timing_decorator
    def refresh_state(self,):
        """
        刷新当前的状态
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        print(f"TaskSelector.refresh_state: tmpl_id_list = {self.explorer.state_manager_dict.keys()}")
        # for tmpl_id, state_manager in self.explorer.state_manager_dict.items():
        #     state_manager: state_estimation.StateManager = state_manager
        #     # self.template_history_benefit[tmpl_id] = {
        #     #     "max_p_error": 0.0,
        #     #     "instance_num": 0,
        #     #     "state_num": 0
        #     # }
        #     self.template_history_benefit[tmpl_id] = state_manager.get_exploration_history()

    # @utils.timing_decorator
    def select_template(self,):
        """
        选择模版，根据历史信息选择探索的模版，考虑多种策略

        1. 根据max_p_error设置对应概率，再进行模版选取

        Args:
            arg1:
            arg2:
        Returns:
            tmpl_id:
        """
        # 2024-03-19: 更新template_selection的策略
        # self.refresh_state()
        # p_error_dict = { tmpl_id: item["max_p_error"] for \
        #     tmpl_id, item in self.template_history_benefit.items() }
        def calc_improvement(error_list):
            # 
            window_size = 5
            diff_val = 1.0
            if len(error_list) <= window_size:
                diff_val = max(error_list) - 1.0
            else:
                diff_val = max(error_list[-1 * window_size:]) - max(error_list[:-1 * window_size]) 
            return max(diff_val, 1.0)
    
        def merge_info(score_dict1: dict, score_dict2: dict, alpha: float = 0.75):
            # 合并两个score_dict
            assert set(score_dict1.keys()) == set(score_dict2.keys())

            score_dict1 = utils.prob_dict_normalization(score_dict1)
            score_dict2 = utils.prob_dict_normalization(score_dict2)
            key_list = score_dict1.keys()
            merge_dict = {k: (alpha * score_dict1[k] + (1 - alpha) * score_dict2[k]) for k in key_list}

            # print(f"merge_info: merge_dict = {utils.dict_round(merge_dict, 2)}\n"
            #       f"score_dict1 = {utils.dict_round(score_dict1, 2)}\nscore_dict2 = {utils.dict_round(score_dict2, 2)}")
            return merge_dict

        p_error_dict = { tmpl_id: (max(1.0, max(item)) - 1.0) for \
            tmpl_id, item in self.template_history_benefit.items() }
        
        recent_dict = {
            tmpl_id: calc_improvement(item) for tmpl_id, item in self.template_history_benefit.items()
        }

        merge_dict = merge_info(p_error_dict, recent_dict)
        
        if len(merge_dict) == 0:
            # 若merge_dict中没有内容，直接从id_list随机选一个
            target_tmpl_id = np.random.choice(self.explorer.template_id_list)
        else:
            target_tmpl_id = utils.prob_dict_choice(merge_dict)

        # target_tmpl_id = utils.prob_dict_infer(p_error_dict, out_num=1)
        # print(f"select_template: p_error_dict = {utils.dict_round(p_error_dict, 3)}. target_tmpl_id = {target_tmpl_id}.")

        print(f"select_template: merge_dict = {utils.dict_round(merge_dict, 2)}. target_tmpl_id = {target_tmpl_id}.")
        return target_tmpl_id
    
# %%
