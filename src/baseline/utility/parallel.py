#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from os.path import join as p_join
from matplotlib import pyplot as plt
import json, os, time, re, math
import pickle
from copy import deepcopy

from utility import utils
import psycopg2 as pg
from asynchronous import construct_input, task_management, state_inspection
from plan import node_extension
# %%


class ParallelSearcher(object):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, workload, total_task_num):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        self.workload = workload
        self.total_task_num = total_task_num
        
        self.db_conn = pg.connect(**construct_input.conn_dict(workload))
        self.inspector = state_inspection.WorkloadInspector(workload=workload, db_conn=self.db_conn)
        self.agent = task_management.TaskAgent(workload=workload, inspector=self.inspector)

        self.complete_order = []    # 任务完成顺序

        # 注册相关的函数信息
        self.agent.load_external_func(self.update_exploration_state, "eval_state")
        self.agent.load_external_func(self.launch_short_task, "short_task")
        self.agent.load_external_func(self.adjust_long_task, "long_task")

        self.reset_search_state()

    # def construct_start_time_list(self,):
    #     """
    #     构造开始时间的列表
    
    #     Args:
    #         arg1:
    #         arg2:
    #     Returns:
    #         time_list:
    #         return2:
    #     """
    #     query_list = self.agent.query_global
    #     assert len(query_list) == len(self.complete_order), f"construct_start_time_list: "\
    #         f"len(query_list) = {len(query_list)}. len(order_list) = {len(self.complete_order)}."

    #     # time_list = [self.task_info_dict[node_sig]['start_time'] for node_sig in self.complete_order]
    #     time_list = []
    #     for node_sig in self.complete_order:
    #         if node_sig in self.task_info_dict.keys():
    #             time_list.append(self.task_info_dict[node_sig]['start_time'])
    #         else:
    #             time_list.append(None)

    #     return time_list

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
        self.inspector.reset_monitor_state()    # 更新观测者的信息

        # 模板探索的信息
        self.template_explore_info = []
        self.complete_order, self.time_list = [], []

        # 任务的保存信息
        self.task_info_dict = {}

        # extension_signature到extension_instance实例引用的字典
        self.extension_ref_dict = {}


    def launch_search_process(self, total_time, with_start_time = False):
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
            time_info: (optional)
        """
        print("call ParallelSearcher.launch_search_process")
        self.start_time = time.time()
        query_list, meta_list, result_list, card_dict_list = \
            self.agent.main_process(total_time = total_time)
        self.agent.terminate_all_instances()
        self.end_time = time.time()

        if with_start_time == False:
            return query_list, meta_list, result_list, card_dict_list
        else:
            assert len(self.time_list) == len(query_list), f"launch_search_process: "\
                f"len(query_list) = {len(query_list)}. len(time_list) = {len(self.time_list)}."
            
            time_info = self.time_list, self.start_time, self.end_time
            return query_list, meta_list, result_list, \
                card_dict_list, time_info

    def update_exploration_state(self,):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        state_dict, query_list, meta_list, result_list, card_dict_list = \
            self.evaluate_running_tasks()
        
        self.start_task_num = self.total_task_num - state_dict['running_num']
        return query_list, meta_list, result_list, card_dict_list

        
    @utils.timing_decorator
    def evaluate_running_tasks(self,):
        """
        更新森林的状态，针对每一个task，需要包含以下的成员信息:

        "node_signature": {
            "task_signature": "",
            "start_time": "",
            "end_time": ""
            "subquery_num":
            "single_table_num":
        }
        
        Args:
            None
        Returns:
            state_dict:
            query_list:
            meta_list:
            result_list:
            card_dict_list:
        """
        # 总的信息
        result_dict = self.inspector.get_workload_state()

        state_dict = {}         
        state_dict['running_num'] = 0   # 还在运行的任务数目

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
            
            # 完成query的update
            subquery_num, single_table_num = len(local_dict['subquery_dict']), len(local_dict['single_table_dict'])

            subquery_target, single_table_target = \
                self.task_info_dict[node_sig]['subquery_num'], self.task_info_dict[node_sig]['single_table_num']
            
            if subquery_num == subquery_target and single_table_num == single_table_target:
                valid_flag, cost_true, cost_estimation, p_error = self.load_async_card_dict(extension_sig = node_sig, \
                    subquery_true = local_dict['subquery_dict'], single_table_true = local_dict['single_table_dict'])

                if valid_flag == True:
                    # 探索完全，添加相应的结果
                    query, meta, result, card_dict = self.add_complete_instance(node_sig, cost_true, cost_estimation)
                    query_list.append(query), meta_list.append(meta)
                    result_list.append(result), card_dict_list.append(card_dict)
                else:
                    self.add_incomplete_instance(node_sig, cost_true, cost_estimation)

                finish_num += 1
                delete_list.append(node_sig)
            else:
                state_dict['running_num'] += 1

        for node_sig in delete_list:
            del self.task_info_dict[node_sig]

        return state_dict, query_list, meta_list, result_list, card_dict_list
    
    
    def add_complete_instance(self, node_sig, cost_true, cost_estimation):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        self.complete_order.append(node_sig)
        self.time_list.append(self.task_info_dict[node_sig]['start_time'])

        curr_ext: node_extension.ExtensionInstance = self.extension_ref_dict[node_sig]
        query = curr_ext.query_text
        meta = curr_ext.query_meta
        result = cost_estimation / cost_true, cost_estimation, cost_true
        card_dict = {
            "true": {
                "subquery": curr_ext.subquery_true,
                "single_table": curr_ext.single_table_true
            },
            "estimation": {
                "subquery": curr_ext.subquery_estimation,
                "single_table": curr_ext.single_table_estimation
            }
        }

        return query, meta, result, card_dict
    

    def generate_single_query(self,) -> (str, tuple):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        raise NotImplementedError("generate_single_query has not implimented")
    

    def start_single_task(self, timeout = 60000):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        # 根据不同方法获得查询
        query_text, query_meta = self.generate_single_query()   
        if query_meta is None or query_text is None:
            print("start_single_task: warning! query is invalid.")
            return
        
        ext_instance = self.create_extension_instance(query_text, query_meta)
        
        subquery_res, single_table_res = ext_instance.\
            true_card_plan_async_under_constaint(proc_num=5, timeout=timeout)
        
        signagure = ext_instance.get_extension_signature()
        self.inspector.load_card_info(signagure, subquery_res, single_table_res)

        # 更新全局信息
        self.extension_ref_dict[signagure] = ext_instance
        self.task_info_dict[signagure] = {
            "task_signature": signagure,
            "subquery_num": len(subquery_res),
            "single_table_num": len(single_table_res),
            "start_time": time.time()
        }
        return ext_instance

    def card_dict_valid_check(self, subquery_dict: dict, single_table_dict: dict):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        for val in subquery_dict.values():
            if val < -1e-6:
                print("card_dict_valid_check: 出现超时的情况! 接下来会直接删除任务")
                return False
        
        for val in single_table_dict.values():
            if val < -1e-6:
                print("card_dict_valid_check: 出现超时的情况! 接下来会直接删除任务")
                return False
            
        return True


    def load_async_card_dict(self, extension_sig, subquery_true, single_table_true):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        # 判断基数字典是否有效
        valid_flag = self.card_dict_valid_check(subquery_true, single_table_true)

        if valid_flag == False:
            return valid_flag, -1, -1, 1.0
        
        extension_instance:node_extension.ExtensionInstance = self.extension_ref_dict[extension_sig]
        flag, cost1, cost2 = extension_instance.\
            load_external_missing_card(subquery_true, single_table_true)

        # plan1, plan2, cost1, cost2 = extension_instance.\
        #     load_external_card(subquery_true, single_table_true)
        return valid_flag, cost1, cost2, cost2 / cost1
    
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


    def add_incomplete_instance(self, node_sig, cost_true, cost_estimation):
        """
        {Description}
    
        Args:
            node_sig:
            cost_true:
            cost_estimation:
        Returns:
            None
        """
        print(f"add_incomplete_instance: node_sig = {node_sig}. It may exist problem.")
    


    def adjust_long_task(self,):
        """
        调整长任务，默认不进行处理
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        pass
# %%
