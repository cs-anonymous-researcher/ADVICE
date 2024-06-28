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
from multiprocessing import Process
import socket
from copy import deepcopy
from ce_adapter import deepdb_caller
from base import BaseCaller

# %%

project_dir = ""

# 启动server的命令和调用估计器的不一样
launch_multi_deepdb_server_job_light_cmd = "/home/lianyuan/anaconda3/envs/deepdb/bin/python3 maqp_server.py --evaluate_cardinalities \
    --max_variants 1 \
    --dataset imdb-light \
    --target_path ./baselines/cardinality_estimation/results/deepDB/job_custom_result.csv \
    --ensemble_location ../imdb-benchmark/spn_ensembles/ensemble_relationships_imdb-light_1000000.pkl \
    --query_file_location ./benchmarks/job-light/sql/job_custom_queries.sql \
    --ground_truth_file_location ./benchmarks/job-light/sql/job_custom_true_cardinalities.csv"

launch_multi_deepdb_server_stats_cmd = "/home/lianyuan/anaconda3/envs/deepdb/bin/python3 maqp_server.py --evaluate_cardinalities \
    --max_variants 1 \
    --dataset stats \
    --target_path ./baselines/cardinality_estimation/results/deepDB/stats_custom_result.csv \
    --ensemble_location ../stats-benchmark/spn_ensembles/ensemble_relationships_stats_100000.pkl \
    --query_file_location ./benchmarks/stats/sql/stats_custom_queries.sql \
    --ground_truth_file_location ./benchmarks/stats/sql/stats_custom_true_cardinalities.csv"

# 复制过来再做修改
local_workload_option = deepcopy(deepdb_caller.workload_option)

local_workload_option["job"]["exec_command"] = launch_multi_deepdb_server_job_light_cmd
local_workload_option["stats"]["exec_command"] = launch_multi_deepdb_server_stats_cmd
local_workload_option["job"]['workload_name'] = "job-light"
local_workload_option['stats']['workload_name'] = "stats"
local_workload_option["job"]["port"] = 12345
local_workload_option["stats"]["port"] = 12345

# %%

def index_by_list(val_list, index_list):
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    return [val_list[i] for i in index_list]

def merge_along_index(val_list1, val_list2, index_list1, index_list2):
    """
    根据index合并两个列表
    
    Args:
        val_list1: 
        val_list2: 
        index_list1: 
        index_list2:
    Returns:
        merged_list:
    """
    merged_list = [None for _ in index_list1 + index_list2]

    for idx, val in zip(index_list1, val_list1):
        merged_list[idx] = val

    for idx, val in zip(index_list2, val_list2):
        merged_list[idx] = val

    return merged_list

def get_DeepDB_server_instance(workload = "job"):
    """
    获得DeepDBServer的实例

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    print("get_DeepDB_server_instance: workload = {}.".format(workload))
    wkld_params = local_workload_option[workload]
    instance = DeepDBServer(**wkld_params)
    return instance


class DeepDBServer(BaseCaller):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, port, workload_name, query_path = "./benchmarks/job-light/sql/job_custom_queries.sql", \
        label_path = "./benchmarks/job-light/sql/job_custom_true_cardinalities.csv", \
        result_path = "./baselines/cardinality_estimation/results/deepDB/job_custom_result.csv", 
        exec_command = launch_multi_deepdb_server_job_light_cmd, mapping_func = lambda a: a):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        self.port = port        # 基数估计服务的端口
        self.workload_name = workload_name
        self.query_path = query_path
        self.label_path = label_path
        self.result_path = result_path
        self.exec_command = exec_command
        self.project_path = "/home/lianyuan/DBGenerator/deepdb"     # 工程路径

        self.current_time = time.time()                        # 获得当前的时间
        self.query_cache = {}
        self.mapping_func = mapping_func
    

    def initialization(self,):
        """
        初始化，启动服务
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        self.launch_service()

    def get_estimation_in_batch(self, query_list, label_list = None):
        """
        获得一批查询的估计结果

        Args:
            query_list:
            label_list:
        Returns:
            merged_result:
            return2:
        """
        print("DeepDBServer.get_estimation_in_batch: query_list = {}.".format(query_list))
        # 通过query_cache进行过滤，避免重复计算
        query_cache = self.query_cache

        cache_index, left_index = [], []
        cache_result = []

        for idx, query in enumerate(query_list):
            if query in query_cache.keys():
                cache_index.append(idx)
                cache_result.append(query_cache[query])

        all_index = [i for i in range(len(query_list))]
        left_index = list(sorted(set(all_index).difference(cache_index)))

        #
        if label_list is None or len(query_list) != len(label_list):
            label_list = [1000 for _ in query_list]

        # filter operation
        query_list = index_by_list(query_list, left_index)  
        label_list = index_by_list(label_list, left_index)

        if len(label_list) == 0:
            # 没有需要eval的query了，直接返回结果
            print("DeepDBServer.get_estimation_in_batch: all queries are found in cache")
            return cache_result

        # print("old query_list = {}.".format(query_list))
        query_list = self.mapping_func(query_list)          # 完成名字的映射

        # 处理as的问题
        query_list = [query.replace(" as ", " ") for query in query_list]
        print("new query_list = {}.".format(query_list))

        print("DeepDBServer.get_estimation_in_batch: data_serialization.")
        self.data_serialization(query_list, label_list)     # 数据序列化
        # self.launch_project(print_log=True)
        print("DeepDBServer.get_estimation_in_batch: execute_card_estimation.")
        self.execute_card_estimation()
        print("DeepDBServer.get_estimation_in_batch: fetch_result.")
        result_list = self.fetch_result(self.result_path)

        merged_result = merge_along_index(result_list, cache_result, left_index, cache_index)
        return merged_result
    
    def preload_queries(self, query_list, label_list):
        """
        针对stats数据集的预取
        
        Args:
            query_list:
            label_list:
        Returns:
            res1:
            res2:
        """
        result_list = self.get_estimation_in_batch(query_list, label_list)

        for k, v in zip(query_list, result_list):
            self.query_cache[k] = v

    def get_estimation_by_query(self, subquery_dict, single_table_dict):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        subquery_res = {}
        single_table_res = {}

        query_list = list(subquery_dict.keys())
        # print("query_list = {}".format(query_list))
        card_list = self.get_estimation_in_batch(query_list = query_list)

        for k, v in zip(query_list, card_list):
            subquery_res[k] = v

        query_list = list(single_table_dict.keys())
        # print("query_list = {}".format(query_list))
        card_list = self.get_estimation_in_batch(query_list = query_list)

        for k, v in zip(query_list, card_list):
            single_table_res[k] = v

        return subquery_res, single_table_res
    

    def service_detection(self,):
        """
        检测当前的服务有效性

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        pass

    def launch_service(self,):
        """
        启动基数估计服务
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """

        # 首先检测port是否被占用了，如果被占用，默认DeepDB server已经启动了
        def is_port_in_use(port):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex(('localhost', port)) == 0

        if is_port_in_use(self.port) == True:
            print("端口已经被占用了")
            return
        
        saved_dir = os.getcwd()
        os.chdir(self.project_path)  
        print("Enter DeepDB directory")
        p = Process(target = lambda: os.system(self.exec_command), args=())  # 使用多进程启动该服务
        p.start()           # 启动server
        time.sleep(3)       # 主要得保证server顺利启动了
        print("Server has started")
        # 切回原来的目录
        os.chdir(saved_dir)
        
    def stop_service(self,):
        """
        停止基数估计服务
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        DeepDB_socket = socket.socket()
        DeepDB_socket.connect(("localhost", self.port))
        print("Send stop command. Exit server")
        DeepDB_socket.send("stop".encode())

    def data_serialization(self, query_list, label_list):
        """
        {Description}

        Args:
            query_list:
            label_list:
        Returns:
            return1:
            return2:
        """
        # 调整路径到deepdb项目中

        # print("query_list = {}".format(query_list))
        saved_path = os.getcwd()
        os.chdir(self.project_path)
        print("DeepDBServer.data_serialization: saved_path = {}. project_path = {}. label_path = {}. query_path = {}.".\
              format(saved_path, self.project_path, self.label_path, self.query_path))
        with open(self.query_path, "w") as f_out:
            f_out.write("\n".join(query_list))
        
        result_list = zip(list(range(len(query_list))), ["\"{}\"".format(i) for i in query_list], label_list)
        print("result_list = {}".format(result_list))

        label_df = pd.DataFrame(result_list, columns=["query_no", "query", "cardinality_true"])
        label_df.to_csv(self.label_path, index = False)

        os.chdir(saved_path)


    def fetch_result(self, result_path):
        """
        获取结果
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        result_modified_time = os.path.getmtime(\
            p_join(self.project_path, result_path))

        if result_modified_time < self.current_time:
            raise ValueError("结果文件并没有被修改: file modified time < instance creation time!")

        saved_path = os.getcwd()    # 切换到相应目录
        os.chdir(self.project_path)
        res_df = pd.read_csv(result_path)
        os.chdir(saved_path)
        return list(res_df['cardinality_predict'])


    def execute_card_estimation(self,):
        """
        执行基数估计
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        # 通过socket调用DeepDB程序
        print("DeepDBServer.execute_card_estimation: begin")
        DeepDB_socket = socket.socket()
        DeepDB_socket.connect(("localhost", self.port))
        DeepDB_socket.send("execute".encode())
        recv_msg = DeepDB_socket.recv(1024)     # 确认DeepDB程序已调用成功
        # print("")
        print("DeepDBServer.execute_card_estimation: end. recv_msg = {}.".format(recv_msg))
        return recv_msg    