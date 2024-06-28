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

from base import BaseCaller
from ce_adapter.case_process import default_mapping, stats_name_mapping

# %%

launch_multi_deepdb_job_light_cmd = """/home/lianyuan/anaconda3/envs/deepdb/bin/python3 maqp.py --evaluate_cardinalities \
    --max_variants 1 \
    --dataset imdb-light \
    --target_path ./baselines/cardinality_estimation/results/deepDB/job_custom_result.csv \
    --ensemble_location ../imdb-benchmark/spn_ensembles/ensemble_relationships_imdb-light_1000000.pkl \
    --query_file_location ./benchmarks/job-light/sql/job_custom_queries.sql \
    --ground_truth_file_location ./benchmarks/job-light/sql/job_custom_true_cardinalities.csv"""


launch_multi_deepdb_stats_cmd = """/home/lianyuan/anaconda3/envs/deepdb/bin/python3 maqp.py --evaluate_cardinalities \
    --max_variants 1 \
    --dataset stats \
    --target_path ./baselines/cardinality_estimation/results/deepDB/stats_custom_result.csv \
    --ensemble_location ../stats-benchmark/spn_ensembles/ensemble_relationships_stats_100000.pkl \
    --query_file_location ./benchmarks/stats/sql/stats_custom_queries.sql \
    --ground_truth_file_location ./benchmarks/stats/sql/stats_custom_true_cardinalities.csv"""


workload_option = {
    "job": {
        "query_path": "./benchmarks/job-light/sql/job_custom_queries.sql",
        "label_path": "./benchmarks/job-light/sql/job_custom_true_cardinalities.csv",
        "result_path": "./baselines/cardinality_estimation/results/deepDB/job_custom_result.csv",
        "exec_command": launch_multi_deepdb_job_light_cmd,
        "mapping_func": default_mapping
    },
    "stats": {
        "query_path": "./benchmarks/stats/sql/stats_custom_queries.sql",
        "label_path": "./benchmarks/stats/sql/stats_custom_true_cardinalities.csv",
        "result_path": "./baselines/cardinality_estimation/results/deepDB/stats_custom_result.csv",
        "exec_command": launch_multi_deepdb_stats_cmd,
        "mapping_func": stats_name_mapping
    },
    "music": {
        "query_path": "",
        "label_path": "",
        "result_path": "",
        "exec_command": "",
        "mapping_func": ""
    },
    "release": {

    }
}


def preload_queries(preload_option):
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    if preload_option['workload'] == "stats":
        # location = preload_option['files']
        single_table_path = "../workload/stats/stats_Expt_single_table_sub_query.sql"
        subquery_path = "../workload/stats/stats_Expt_sub_queries.sql"

        query_list1, label_list1 = file_process_stats(subquery_path)
        query_list2, label_list2 = file_process_stats(single_table_path)

        return query_list1 + query_list2, label_list1 + label_list2

def file_process_stats(f_path):
    """
    {Description}
    
    Args:
        f_path:
    Returns:
        query_list:
    """
    query_list = []
    label_list = []
    with open(f_path) as f_in:
        line_list = f_in.read().splitlines()

    for l in line_list:
        elems = l.split("||")[0]
        query_list.append(elems[0])
        label_list.append(elems[1])

    return query_list, label_list


def get_DeepDB_caller_instance(workload = "job"):
    """
    {Description}
    
    Args:
        workload:
    Returns:
        instance:
    """
    wkld_params = workload_option[workload]
    instance = DeepDBCaller(**wkld_params)
    return instance
    

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


class DeepDBCaller(BaseCaller):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, query_path = "./benchmarks/job-light/sql/job_custom_queries.sql", \
        label_path = "./benchmarks/job-light/sql/job_custom_true_cardinalities.csv", \
        result_path = "./baselines/cardinality_estimation/results/deepDB/job_custom_result.csv",
        exec_command = launch_multi_deepdb_job_light_cmd, mapping_func = lambda a: a):
        """
        {Description}

        Args:
            query_path:
            label_path:
            result_path:
        """
        self.query_path = query_path
        self.label_path = label_path
        self.result_path = result_path
        self.exec_command = exec_command
        self.project_path = "/home/lianyuan/DBGenerator/deepdb"     # 工程路径
        self.mapping_func = mapping_func
        self.current_time = time.time()                        # 获得当前的时间
        self.query_cache = {}

    def get_estimation_in_batch(self, query_list, label_list = None):
        """
        {Description}

        Args:
            query_list:
            label_list:
        Returns:
            result_list:
        """
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

        # filter operation
        query_list = index_by_list(query_list, left_index)  
        label_list = index_by_list(label_list, left_index)

        if len(label_list) == 0:
            # 没有需要eval的query了，直接返回结果
            return cache_result
        #
        if label_list is None or len(query_list) != len(label_list):
            label_list = [1000 for _ in query_list]
        # print("old query_list = {}.".format(query_list))
        query_list = self.mapping_func(query_list)          # 完成名字的映射

        # 处理as的问题
        query_list = [query.replace(" as ", " ") for query in query_list]
        # print("new query_list = {}.".format(query_list))

        self.data_serialization(query_list, label_list)     # 数据序列化
        self.launch_project(print_log=True)
        result_list = self.fetch_result(self.result_path)

        merged_result = merge_along_index(result_list, cache_result, left_index, cache_index)
        return result_list

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
        以query为单位获得基数的信息
        
        Args:
            subquery_dict:
            single_table_dict:
        Returns:
            subquery_res:
            single_table_res:
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
        with open(self.query_path, "w") as f_out:
            f_out.write("\n".join(query_list))
        
        result_list = zip(list(range(len(query_list))), ["\"{}\"".format(i) for i in query_list], label_list)
        print("result_list = {}".format(result_list))

        label_df = pd.DataFrame(result_list, columns=["query_no", "query", "cardinality_true"])
        label_df.to_csv(self.label_path, index = False)

        os.chdir(saved_path)

    def launch_project(self, print_log = False, log_dir = "../log"):
        """
        启动项目
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        current_path = os.getcwd()
        if print_log == True:
            log_name = "{}.log".format(time.strftime("%Y%m%d%H%M%S", time.localtime()))
            if log_dir.startswith("/"):
                log_path = p_join(log_dir, log_name)
            else:
                log_path = p_join(current_path, log_dir, log_name)
            out_file = log_path         # 
        else:
            # out_file = "/dev/null 2>&1"
            out_file = "/dev/null"

        print("out_file = {}".format(out_file))
        os.chdir(self.project_path)      # 切换路径
        os.system(self.exec_command + " > {} 2>&1".format(out_file))  # 输出结果的优化
        os.chdir(current_path)           # 切回路径

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
