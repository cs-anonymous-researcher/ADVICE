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
from result_analysis import case_analysis
from data_interaction import postgres_connector
from utility import utils, workload_spec
import psycopg2 as pg
from asynchronous import construct_input
# %%

project_dir = "/home/lianyuan"

# 本地的PostgreSQL路径
local_pg_data_path = "/home/lianyuan/Database/Installation/pgsql/data"
local_pg_ctl_path = "/home/lianyuan/Database/Installation/pgsql/bin/pg_ctl"

# %%

class ActualExecutionEvaluation(object):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, workload, new_config: dict = {}, ssh_info: dict = {}):
        """
        {Description}

        Args:
            workload:
            new_config:
            ssh_info:
        """
        self.workload = workload
        self.db_conn = postgres_connector.connector_instance(workload, extra_info = new_config)
        self.ssh_info = ssh_info

        if len(ssh_info) > 0:
            print(f"ActualExecutionEvaluation.__init__: ssh_info = {ssh_info}")
            host_password = ssh_info["host_password"]
            pg_data_path = ssh_info["pg_data_path"]
            pg_ctl_path = ssh_info["pg_ctl_path"]
            host_name = ssh_info["host_name"]
        else:
            host_password = "XXXXXX"
            pg_data_path = local_pg_data_path
            pg_ctl_path = local_pg_ctl_path

        clean_command = f"echo {host_password}|sudo -S sh -c \"echo 3 > /proc/sys/vm/drop_caches\""
        # close_db_command = f"pg_ctl -D {pg_data_path} stop -m immediate"
        # start_db_command = f"pg_ctl -D {pg_data_path} -l logfile -o \"-F -p 6432\" start"
        close_db_command = f"{pg_ctl_path} -D {pg_data_path} stop -m immediate"
        start_db_command = f"{pg_ctl_path} -D {pg_data_path} -l logfile -o \"-F -p 6432\" start"

        if len(ssh_info) > 0:
            self.clean_command = f"ssh {host_name} '{clean_command}'"
            self.close_db_command = f"ssh {host_name} '{close_db_command}'"
            self.start_db_command = f"ssh {host_name} '{start_db_command}'"
        else:
            self.clean_command = clean_command
            self.close_db_command = close_db_command
            self.start_db_command = start_db_command

    def show_command_info(self,):
        """
        展示相关的命令信息
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        print(f"clean_command = {self.clean_command}")
        print(f"close_db_command = {self.close_db_command}")
        print(f"start_db_command = {self.start_db_command}")


    def warm_up_query(self, query_text):
        """
        针对查询进行预热
    
        Args:
            query_text:
            arg2:
        Returns:
            warm_up_time:
            return2:
        """
        # conf_dict = construct_input.conn_dict(self.workload)
        # local_conn = pg.connect(**conf_dict)

        # 考虑修改配置
        time_start = time.time()
        # with local_conn.cursor() as cursor:
        #     cursor.execute(query_text)
        # print(f"{self.db_conn.config}")
        self.db_conn.execute_single_sql(query_text)
        time_end = time.time()
        return time_end - time_start

    def execution_query(self, query_text, subquery_dict, \
            single_table_dict, start_mode = "hot", has_warm_up = False):
        """
        {Description}

        Args:
            query_text:
            subquery_dict:
        Returns:
            plan_dict:
            delta_time:
        """
        head = "EXPLAIN (FORMAT JSON, ANALYZE)"
        hint_sql_text = self.db_conn.inject_cardinalities_sql(\
            query_text, subquery_dict, single_table_dict, head=head)
        
        assert start_mode in ("hot", "cold"), f"execution_query: start_mode = {start_mode}."

        if start_mode == "hot":
            if has_warm_up == False:
                self.warm_up_query(hint_sql_text)
        elif start_mode == "cold":
            # 清理所有的cache
            self.clean_database_cache()

        # 在单进程模式下执行
        time_start = time.time()
        # plan_dict = self.db_conn.execute_single_sql(hint_sql_text, use_cache=False)
        plan_dict = self.db_conn.execute_sql_under_single_process(hint_sql_text)
        time_end = time.time()
        delta_time = time_end - time_start

        # print(f"execution_query: delta_time = {time_end - time_start:.2f}")
        return plan_dict, delta_time

    def close_all_connections(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        db_name = workload_spec.workload2database[self.workload]

        # 中止所有活跃的数据库进程
        terminate_query = """SELECT pg_terminate_backend(pg_stat_activity.pid) FROM pg_stat_activity WHERE \
            pg_stat_activity.datname = '{}' AND pid <> pg_backend_pid();""".format(db_name)
        
        with self.db_conn.conn.cursor() as cursor:
            cursor.execute(terminate_query)

    
    def clean_database_cache(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        #
        print("clean_database_cache")

        saved_dir = os.getcwd() 
        os.chdir(project_dir)
        os.system(self.close_db_command)
        time.sleep(3)
        self.clean_os_cache()
        os.system(self.start_db_command)
        os.chdir(saved_dir)

        return None

    def clean_os_cache(self,):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        os.system(self.clean_command)


    def compare_instance(self, query_text, card_dict, start_mode = "hot"):
        """
        {Description}
    
        Args:
            query_text:
            card_dict:
        Returns:
            result_dict: keys = ("true_execution_time", "est_execution_time", "error_ratio")
            return2:
        """
        subquery_true, single_table_true, subquery_estimation, \
            single_table_estimation = utils.extract_card_info(card_dict)

        #
        # print("compare_instance: execute true plan")
        plan_dict_true, delta_time_true = self.execution_query(query_text, \
            subquery_true, single_table_true, start_mode=start_mode, has_warm_up=False)
        
        # 
        # print("compare_instance: execute estimated plan")
        plan_dict_est, delta_time_est = self.execution_query(query_text, \
            subquery_estimation, single_table_estimation, start_mode=start_mode, has_warm_up=True)
        

        result_dict = {
            "true_execution_time": delta_time_true,
            "est_execution_time": delta_time_est,
            "error_ratio": delta_time_est / delta_time_true
        }
        return result_dict
    

    def result_construction(self, query_list, card_dict_list, p_error_list, start_mode): 
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        out_list = []
        assert len(query_list) == len(card_dict_list) == len(p_error_list)

        for idx, (query, card_dict, p_error) in enumerate(zip(query_list, card_dict_list, p_error_list)):
            result_dict = self.compare_instance(query, card_dict, start_mode)

            print(f"result_construction: idx = {idx}. true_plan_time = {result_dict['true_execution_time']:.3f}. "\
                f"est_plan_time = {result_dict['est_execution_time']:.3f}. time_error = "\
                f"{result_dict['error_ratio']:.3f}. p_error = {p_error:.3f}. query = {query}")
            out_list.append(result_dict)

        return out_list

# %%
