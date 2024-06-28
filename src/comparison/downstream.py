#!/usr/bin/env python
# -*- coding: utf-8 -*-


# %%
import numpy as np
import pandas as pd
from os.path import join as p_join
from matplotlib import pyplot as plt
import json, os, time, re, math
import pickle
from collections import defaultdict
from itertools import combinations
from copy import deepcopy

from error_analyze import plan_card_evaluation
from utility import graph_data, utils, workload_parser
from comparison import result_base
from typing import Iterable

# %%
from query import query_construction
from estimation import case_based_estimation
from result_analysis import case_analysis

# %%

class QueryPairEvaluator(object):
    """
    两个查询的评测器

    Members:
        field1:
        field2:
    """

    def __init__(self, workload):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        self.workload = workload

    def card_dict_mask(self, in_card_dict):
        """
        先考虑把所有subquery的信息给删了？
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        out_card_dict = deepcopy(in_card_dict)
        out_card_dict['true']['subquery'] = {}
        out_card_dict['estimation']['subquery'] = {}
        
        return out_card_dict

    def construct_cmp_info(self, meta1, card_dict1, meta2, card_dict2):
        """
        {Description}

        Args:
            meta1:
            card_dict1:
            meta2:
            card_dict2:
        Returns:
            meta_pair_list: 元信息对的列表
            true_distance_list:
            est_distance_list: 估计精度距离列表
        """
        mask_dict2 = self.card_dict_mask(card_dict2)
        out_meta, out_card_dict = case_based_estimation.case_pair_estimation(\
            self.workload, meta1, card_dict1, meta2, mask_dict2, mode="both")

        # 
        meta_pair_list, true_distance_list, est_distance_list = [], [], []

        subquery_true_1, _, subquery_est_1, _ = utils.extract_card_info(out_card_dict)
        subquery_true_2, _, subquery_est_2, _ = utils.extract_card_info(card_dict1)

        distance_true: dict = self.build_distance_dict(subquery_true_2, subquery_true_1)
        distance_est: dict = self.build_distance_dict(subquery_est_2, subquery_est_1)

        # 
        instance1 = case_analysis.construct_case_instance(meta1, card_dict1, self.workload)
        instance2 = case_analysis.construct_case_instance(meta2, card_dict2, self.workload)

        subquery_keys = distance_true.keys()
        for k in subquery_keys:
            submeta_1 = instance1.construct_submeta(k)
            submeta_2 = instance2.construct_submeta(k)

            meta_pair_list.append((submeta_1, submeta_2))
            true_distance_list.append(distance_true[k])
            est_distance_list.append(distance_est[k])
        # 
        return meta_pair_list, true_distance_list, est_distance_list

    def build_distance_dict(self, card_ref, card_est):
        """
        {Description}

        Args:
            card_ref:
            card_est:
        Returns:
            distance_dict:
        """
        card_merge = utils.dict_concatenate(card_ref, card_est)
        try:
            card_ratio = utils.dict_apply(card_merge, \
                lambda a: (a[0] + 1.0) / (a[1] + 1.0), mode="value")
        except TypeError as e:
            print(f"build_distance_dict: meet TypeError. card_merge = {card_merge}.")
            raise e

        card_log = utils.dict_apply(card_ratio, lambda a: np.log(a), mode="value")
        return card_log
        
# %%

class OutputConstructor(result_base.ResultBase):
    """
    {Description}

    Members:
        field1:
        field2:
    """
    def __init__(self, workload, intermediate_dir = "/home/lianyuan/Research/CE_Evaluator/intermediate", 
            result_dir = "/home/lianyuan/Research/CE_Evaluator/result", 
            config_dir = "/home/lianyuan/Research/CE_Evaluator/evaluation/config"):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        super().__init__(workload, intermediate_dir, result_dir)
        self.workload = workload
        self.intermediate_dir = intermediate_dir
        self.config_dir = config_dir
        self.result_dir = result_dir
        self.result_meta_dict = self.load_meta()
        self.result_processor = graph_data.ResultProcessor(workload, mode="tuple")

    def load_meta(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        # 加载结果的元信息
        meta_path = p_join(self.intermediate_dir, self.workload, \
                           "experiment_obj", "result_meta.json")
        meta_dict = utils.load_json(data_path=meta_path)
        return meta_dict
    
    def set_config(self, conf_name):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        conf_path = p_join(self.config_dir, self.workload, conf_name)
        self.result_dict = utils.load_json(conf_path)['result_dict']

    def construct_dataset_by_method(self, ce_method, search_method = "stateful_parallel", output_format = "data"):
        """
        {Description}
    
        Args:
            ce_method:
            search_method:
        Returns:
            return1:
            return2:
        """
        ce_method_mapping = {
            "internal": "PostgreSQL",
            "SQLServer": "SQLServer",
            "DeepDB_rdc": "DeepDB-RDC",
            "DeepDB_jct": "DeepDB-JCT",
            "MSCN": "MSCN",
            "FCN": "FCN", 
            "FCNPool": "FCNPool"
        }

        result_id_list = [str(item) for item in self.result_dict[ce_method_mapping[ce_method]]]
        verify_complete, verify_estimation, verify_top = True, False, True

        assert output_format in ("graph", "data")
        graph_list = []
        meta_total, card_dict_total = [], []

        for result_id in result_id_list:
            instance_meta = self.result_meta_dict[result_id]
            curr_search = instance_meta['search_method']
            if curr_search != search_method:
                continue

            obj_path = instance_meta['obj_path']
            result_obj = self.load_object(obj_path)

            if len(result_obj) == 5:
                result_obj = result_obj[:4]
                
            result_obj = self.filter_wrap_func(result_obj, \
                verify_complete, verify_estimation, verify_top, curr_search)
            print(f"construct_dataset: len(result_obj) = {len(result_obj)}")

            if output_format == "graph":
                instance_list = list(zip(*result_obj))
                for data_input in instance_list:
                    graph_local = self.result_processor.build_correlation_graph(data_input)
                    graph_list.append(graph_local)
            elif output_format == "data":
                query_local, meta_local, result_local, card_dict_local = result_obj
                meta_total.extend(meta_local)
                card_dict_total.extend(card_dict_local)

        if output_format == "graph":
            return graph_list
        elif output_format == "data":
            return meta_total, card_dict_total

    def construct_dataset_by_id_list(self, id_list):
        """
        {Description}

        Args:
            id_list:
            ce_method:
        Returns:
            graph_list:
            return2:
        """
        verify_complete, verify_estimation, verify_top = True, False, True
        graph_list = []

        for result_id in id_list:
            instance_meta = self.result_meta_dict[result_id]
            obj_path = instance_meta['obj_path']
            result_obj = self.load_object(obj_path)
            method_name = instance_meta['search_method']
            result_obj = self.filter_wrap_func(result_obj, \
                verify_complete, verify_estimation, verify_top, method_name)
            print(f"construct_dataset_by_id_list: len(result_obj) = {len(result_obj)}")

            instance_list = list(zip(*result_obj))

            for data_input in instance_list:
                graph_local = self.result_processor.build_correlation_graph(data_input)
                graph_list.append(graph_local)
                
        return graph_list
    
    def construct_query_list(self, id_list, number_limit = None):
        """
        {Description}
    
        Args:
            id_list:
            arg2:
        Returns:
            query_text_list:
            return2:
        """
        query_text_list = []

        for result_id in id_list:
            instance_meta = self.result_meta_dict[result_id]
            obj_path = instance_meta['obj_path']
            result_obj = self.load_object(obj_path)

            if len(result_obj) == 5:
                # 排除time的信息
                result_obj = result_obj[:4]

            instance_list = list(zip(*result_obj))

            # print(f"construct_dataset: len(result_obj[0]) = {len(result_obj[0])}.")
            # print(f"construct_dataset: len(result_obj) = {len(result_obj)}. len(instance_list) = {len(instance_list)}. obj_path = {obj_path}.")

            for data_input in instance_list:
                query_text, query_meta, result, card_dict = data_input
                subquery_true, single_table_true, subquery_est, \
                    single_table_est = utils.extract_card_info(card_dict)
                query_batch = self.construct_all_queries(query_text, \
                    subquery_est.keys(), single_table_est.keys())
                
                query_text_list.extend(query_batch)

        # 根据number_limit来截断数据
        if number_limit is not None and number_limit < len(query_text_list):
            query_text_list = np.random.choice(query_text_list, number_limit)

        return query_text_list
    

    def construct_all_queries(self, top_query: str, subquery_keys: Iterable, single_table_keys: Iterable):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            query_batch:
            return2:
        """
        query_parser = workload_parser.SQLParser(top_query, self.workload)
        query_batch = []

        for k in subquery_keys:
            query_batch.append(query_parser.construct_sub_queries(k))

        for k in single_table_keys:
            query_batch.append(query_parser.construct_sub_queries([k,]))

        return query_batch

    def construct_instance_pair(self, id_list, number_limit = None):
        """
        构造实例对，寻找schema_list相同的进行匹配
    
        Args:
            arg1:
            arg2:
        Returns:
            meta_pair_list:
            distance_list:
        """
        instance_dict = defaultdict(list)

        for result_id in id_list:
            instance_meta = self.result_meta_dict[result_id]
            obj_path = instance_meta['obj_path']
            result_obj = self.load_object(obj_path)
            print(f"construct_dataset: len(result_obj) = {len(result_obj)}")
            instance_list = list(zip(*result_obj))

            for data_input in instance_list:
                if len(data_input) == 4:
                    query_text, query_meta, result, card_dict = data_input
                elif len(data_input) == 5:
                    query_text, query_meta, result, card_dict, start_time = data_input

                instance_repr = tuple(sorted(query_meta[0]))
                instance_dict[instance_repr].append(\
                    (query_text, query_meta, result, card_dict))
                
        def pairwise_split(local_list, num_limit = 10000):
            pair_list = []
            for idx1, idx2 in combinations(range(len(local_list)), 2):
                if idx1 == idx2:
                    continue
                else:
                    pair_list.append((local_list[idx1], local_list[idx2]))
            print(f"construct_instance_pair.pairwise_split: "
                  f"len(local_list) = {len(local_list)}. len(pair_list) = {len(pair_list)}")
            
            return pair_list
        
        meta_pair_list, distance_list = [], []
        pair_eval = QueryPairEvaluator(self.workload)

        for k, v in instance_dict.items():
            pair_list = pairwise_split(v)

            for inst1, inst2 in pair_list:
                _, query_meta1, _, card_dict1 = inst1
                _, query_meta2, _, card_dict2 = inst2

                meta_pair_local, true_distance_local, est_distance_local = \
                    pair_eval.construct_cmp_info(query_meta1, card_dict1, query_meta2, card_dict2)

                meta_pair_list.extend(meta_pair_local)
                distance_list.extend(true_distance_local)

        assert len(meta_pair_list) == len(distance_list)

        if number_limit is not None and number_limit < len(meta_pair_list):
            selected_idx = np.random.choice(range(len(meta_pair_list)), number_limit)
            meta_pair_res = utils.list_index(meta_pair_list, selected_idx)
            distance_res = utils.list_index(distance_list, selected_idx)
        else:
            meta_pair_res, distance_res = meta_pair_list, distance_list

        return meta_pair_res, distance_res


    def est_card_substitute(self, ):
        """
        基数估计替代

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        pass

# %%

class WorkloadConstructor(result_base.ResultBase):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, workload, intermediate_dir = "/home/lianyuan/Research/CE_Evaluator/intermediate", 
            result_dir = "/home/lianyuan/Research/CE_Evaluator/result", 
            config_dir = "/home/lianyuan/Research/CE_Evaluator/evaluation/config"):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        super().__init__(workload, intermediate_dir, result_dir, config_dir)
        self.config_dir = config_dir
        meta_path = p_join(intermediate_dir, workload, "experiment_obj", "result_meta.json")
        self.meta_dict = utils.load_json(data_path=meta_path)


    def construct_specific_workload(self, ce_method, query_num):
        """
        {Description}

        Args:
            ce_method:
            query_num: 查询数目
        Returns:
            query_selected:
            card_dict_selected:
        """
        query_list, card_dict_list, error_list = self.aggregate_result(ce_method)

        index_sorted = list(np.argsort(error_list)[::-1])
        assert len(index_sorted) >= query_num
        index_selected = index_sorted[:query_num]

        query_selected = utils.list_index(query_list, index_selected)
        card_dict_selected = utils.list_index(card_dict_list, index_selected)
        
        return query_selected, card_dict_selected

    def save_result(self, query_list, card_dict_list, ce_method, out_dir = "/home/lianyuan/Research/CE_Evaluator/data/workload/interactive"):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        pickle_name = f"{self.workload}_{ce_method}_interactive.pkl"
        pickle_path = p_join(out_dir, self.workload, pickle_name)
        utils.dump_pickle((query_list, card_dict_list), pickle_path)

    def aggregate_result(self, ce_method):
        """
        {Description}
    
        Args:
            ce_method:
            arg2:
        Returns:
            return1:
            return2:
        """
        ce_method_alias = {
            "internal": "PostgreSQL",
            "SQLServer": "SQLServer",
            "DeepDB_jct": "DeepDB-JCT",
            "DeepDB_rdc": "DeepDB-RDC"
        }
        result_list = self.result_config[ce_method_alias[ce_method]]
        query_list, card_dict_list, error_list = [], [], []
        for result_id in result_list:
            try:
                instance_meta = self.result_meta_dict[str(result_id)]
            except KeyError as e:
                print(f"parse_config.load_result_by_id: meet KeyError. result_meta_dict = {self.result_meta_dict.keys()}.")
                raise e
            
            obj_path = instance_meta['obj_path']
            result_obj = self.load_object(obj_path)

            if len(result_obj) == 5:
                result_obj = result_obj[:4]

            result_obj = self.filter_wrap_func(result_obj, card_top = True, 
                card_complete = True, card_estimation = False, ce_type = ce_method)
            query_local, meta_local, result_local, card_dict_local = result_obj
            error_local = [item[0] for item in result_local]

            query_list.extend(query_local)
            error_list.extend(error_local)
            card_dict_list.extend(card_dict_local)

        # sorted_index = list(np.argsort())
        return query_list, card_dict_list, error_list
        

    def load_result(self, conf_name):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        conf_path = p_join(self.config_dir, self.workload, conf_name)
        conf_dict = utils.load_json(conf_path)

        self.result_config = conf_dict["result_dict"]
        return self.result_config

# %%
