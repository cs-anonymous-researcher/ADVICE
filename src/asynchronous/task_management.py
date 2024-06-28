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

from asynchronous import state_inspection, construct_input
from utility import workload_spec
import signal
import psycopg2 as pg
from utility import common_config


# %%
default_func = lambda: None
false_func = lambda: False

# %%

class TaskAgent(object):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, workload, inspector: state_inspection.WorkloadInspector = None):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        self.workload = workload
        self.inspector = inspector
        self.controller = pg.connect(**construct_input.conn_dict(workload))
        self.state_dict = {}

        self.query_global, self.meta_global, self.result_global, \
            self.card_dict_global = [], [], [], []
        
        # 更新探索的状态
        self.update_exploration_state = default_func
        # 
        self.launch_short_task = default_func
        # 调整长任务的状态
        self.adjust_long_task = default_func

        self.function_load_set = set()
        self.exit_func = false_func

        # 20231226新增
        self.suspended_set = set()

    def suspend_instance(self, signature):
        """
        停止某个实例

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        # 
        self.suspended_set.add(signature)
        pid_list = self.inspector.get_correlated_processes(\
            info = signature, mode = "all")
        
        pid_finished = self.inspector.get_finished_proc(signature)
        pid_left = list(set(pid_list).difference(pid_finished))

        for pid in pid_left:
            try:
                os.kill(pid, signal.SIGSTOP)
            except ProcessLookupError as e:
                print(f"suspend_instance: meet ProcessLookupError. pid = {pid}. "\
                      f"signature = {signature}. pid_list = {pid_list}.")
                # raise e
    
    def terminate_suspended_process(self,):
        """
        处理所有被暂停的进程，唤醒并终止它们
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        left_sig_list = list(self.suspended_set)
        for signature in left_sig_list:
            self.restore_instance(signature)


    def terminate_instance(self, signature):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        pid = self.inspector.get_correlated_processes(\
            info=signature, mode = "own")

        with self.controller.cursor() as cursor:
            cursor.execute("SELECT pg_terminate_backend({pid});".format(pid=pid))

    def terminate_all_instances(self,):
        """
        中止所有运行的数据库查询
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        db_name = workload_spec.workload2database[self.workload]

        get_pid_query = """SELECT pg_stat_activity.pid FROM pg_stat_activity WHERE \
            pg_stat_activity.datname = '{}' AND pid <> pg_backend_pid();""".format(db_name)
        
        try:
            with self.controller.cursor() as cursor:
                cursor.execute(get_pid_query)
                result = cursor.fetchall()
                pid_list = [item[0] for item in result]
        except pg.OperationalError as e:
            pass

        print(f"terminate_all_instances: pid_list = {pid_list}")

        # 中止所有活跃的数据库进程
        terminate_query = """SELECT pg_terminate_backend(pg_stat_activity.pid) FROM pg_stat_activity WHERE \
            pg_stat_activity.datname = '{}' AND pid <> pg_backend_pid();""".format(db_name)
        
        with self.controller.cursor() as cursor:
            cursor.execute(terminate_query)
        self.controller.commit()

    def restore_instance(self, signature):
        """
        恢复实例的运行
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        self.suspended_set.remove(signature)
        pid_list = self.inspector.get_correlated_processes(\
            info = signature, mode = "all")

        # 剔除已经完成的进程
        pid_finished = self.inspector.get_finished_proc(signature)
        pid_left = list(set(pid_list).difference(pid_finished))

        for pid in pid_left:
            try:
                os.kill(pid, signal.SIGCONT)
            except ProcessLookupError as e:
                print(f"restore_instance: meet ProcessLookupError. pid = {pid}. "\
                      f"signature = {signature}. pid_list = {pid_list}.")
                # raise e

    def load_external_func(self, in_func, mode):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        # print(f"load_external_func: in_func = {in_func}. mode = {mode}.")

        if mode in ["short_task", "long_task", "eval_state"]:
            self.function_load_set.add(mode)

        if mode == "short_task":
            self.launch_short_task = in_func
        elif mode == "long_task":
            self.adjust_long_task = in_func
        elif mode == "eval_state":
            self.update_exploration_state = in_func
        elif mode == "exit":
            self.exit_func = in_func
        else:
            raise ValueError("load_external_func: Unsupported mode ({}). \
                available_mode = ['short_task', 'long_task', 'eval_state']".format(mode))

    def update_and_fetch_state(self,):
        """
        更新并获取当前状态
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        task_list = self.inspector.get_all_unfinished_tasks()
        result_dict = {}

        # 获取当前任务的Process Number
        

        # 获得所有任务的CPU Time
        for i in task_list:
            cpu_time = self.inspector.get_cpu_time(i)
            result_dict[i]['cpu_time'] = cpu_time

        # 获得未完成task的部分cardinality
        for i in task_list:
            card_dict = self.inspector.get_card_dict(i)
            result_dict[i]['card_dict'] = card_dict

        return result_dict

    def main_process(self, total_time):
        """
        主过程
        
        Args:
            total_time:
            arg2:
        Returns:
            query_global:
            meta_global: 
            result_global: 
            card_dict_global:
        """
        interval = 5        # 每隔5s处理一次信息
        init_time = time.time()
        global_cnt = 0

        assert type(total_time) in (int, float), f"main_process: time TypeError. total_time = {total_time}"
        assert len(self.function_load_set) >= 3, \
            f"main_process: has not load enough function. function_load_set = {self.function_load_set}"
        while True:
            global_cnt += 1
            epoch_start_time = time.time()    # 开始时间

            # 
            conn_num = self.inspector.get_database_conn_state()
            total_num, total_cpu = self.inspector.get_server_cpu_state()
            free, used, buffers = self.inspector.get_server_memory_usage()
            
            # print("main_process: global_cnt = {}.".format(global_cnt))
            # print("main_process: elasped_time = {:.2f}. conn_num = {}. total_proc_num = {}. total_proc_cpu = {:.2f}. "\
            #       "memory_free = {:.2f}. memory_used = {:.2f}. memory_buffers = {:.2f}".\
            #       format(epoch_start_time - init_time, conn_num, total_num, total_cpu, free, used, buffers))

            # 更新当前状态
            query_list, meta_list, result_list, card_dict_list = self.update_exploration_state()

            # 完成结果的更新
            self.query_global.extend(query_list), self.meta_global.extend(meta_list)
            self.result_global.extend(result_list), self.card_dict_global.extend(card_dict_list)

            # 调整长任务
            self.adjust_long_task()

            # 调整短任务
            self.launch_short_task()
            
            epoch_end_time = time.time()
            # 调整程序睡眠的时长
            curr_time = time.time()
            print("main_process: epoch_delta_time = {:.3f}. total_delta_time = {:.3f}. total_target_time = {:.3f}".\
                  format(epoch_end_time - epoch_start_time, curr_time - init_time, total_time))

            # time.sleep(interval)        # 休眠等待执行
            # 2024-03-11: 修改休眠的逻辑，优化探索效率
            time.sleep(common_config.calculate_gap_time(epoch_end_time - epoch_start_time))

            # 运行时间到达阈值，退出循环
            if curr_time - init_time > total_time:
                print(f"main_process: while loop finish. global_cnt = {global_cnt}.")
                break
            
            # 由exit_func判断条件，实现提前退出的情况
            if self.exit_func() == True:
                print(f"main_process: exit_func return True. global_cnt = {global_cnt}.")
                break

        return self.query_global, self.meta_global, self.result_global, self.card_dict_global

# %%

class AdvancedAgent(TaskAgent):
    """
    高级的决策器，逻辑和之前的task agent存在区别

    Members:
        field1:
        field2:
    """

    def __init__(self, workload, inspector: state_inspection.WorkloadInspector = None):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        super().__init__(workload, inspector)
        # 构造候选的任务集
        self.construct_candidate_task = default_func

    def load_external_func(self, in_func, mode):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        if mode in ["short_task", "long_task", "eval_state", "candidate_task"]:
            self.function_load_set.add(mode)

        if mode == "short_task":
            # 
            self.launch_short_task = in_func
        elif mode == "long_task":
            #
            self.adjust_long_task = in_func
        elif mode == "eval_state":
            #
            self.update_exploration_state = in_func
        elif mode == "candidate_task":
            # 构造候选任务
            self.construct_candidate_task = in_func
        elif mode == "exit":
            # 退出函数
            self.exit_func = in_func
        else:
            raise ValueError("load_external_func: Unsupported mode ({}). \
                available_mode = ['short_task', 'long_task', 'eval_state']".format(mode))

    def main_process(self, total_time):
        """
        探索循环过程

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        interval = 5        # 每隔5s处理一次信息
        init_time = time.time()
        global_cnt = 0

        assert type(total_time) in (int, float), f"main_process: time TypeError. total_time = {total_time}"
        assert len(self.function_load_set) >= 3, \
            f"main_process: has not load enough function. function_load_set = {self.function_load_set}"
        while True:
            global_cnt += 1
            epoch_start_time = time.time()    # 开始时间

            # 
            conn_num = self.inspector.get_database_conn_state()
            total_num, total_cpu = self.inspector.get_server_cpu_state()
            free, used, buffers = self.inspector.get_server_memory_usage()
            
            print("main_process: global_cnt = {}.".format(global_cnt))
            print("main_process: elasped_time = {:.2f}. conn_num = {}. total_proc_num = {}. total_proc_cpu = {:.2f}. "\
                  "memory_free = {:.2f}. memory_used = {:.2f}. memory_buffers = {:.2f}".\
                  format(epoch_start_time - init_time, conn_num, total_num, total_cpu, free, used, buffers))

            # 更新当前状态，此状态下不考虑创建新的任务，只负责收集结果
            query_list, meta_list, result_list, card_dict_list = self.update_exploration_state()

            # 完成结果的更新
            self.query_global.extend(query_list), self.meta_global.extend(meta_list)
            self.result_global.extend(result_list), self.card_dict_global.extend(card_dict_list)

            # 调整短任务，启动新task到数据库中
            self.launch_short_task()

            # 调整长任务，确定新增任务的数目
            self.adjust_long_task()

            # 根据update_exploration_state的结果构造candidate task
            self.construct_candidate_task()

            epoch_end_time = time.time()
            # 调整程序睡眠的时长
            curr_time = time.time()
            print("main_process: epoch_delta_time = {:.3f}. total_delta_time = {:.3f}. total_target_time = {:.3f}".\
                  format(epoch_end_time - epoch_start_time, curr_time - init_time, total_time))

            # time.sleep(interval)        # 休眠等待执行
            # 2024-03-11: 修改休眠的逻辑，优化探索效率
            time.sleep(common_config.calculate_gap_time(epoch_end_time - epoch_start_time))

            # 运行时间到达阈值，退出循环
            if curr_time - init_time > total_time:
                print(f"main_process: while loop finish. global_cnt = {global_cnt}.")
                break
            
            # 由exit_func判断条件，实现提前退出的情况
            if self.exit_func() == True:
                print(f"main_process: exit_func return True. global_cnt = {global_cnt}.")
                break

        return self.query_global, self.meta_global, self.result_global, self.card_dict_global

# %%
