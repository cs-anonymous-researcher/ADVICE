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
from collections import defaultdict
from itertools import permutations

from utilities import meta_info, utils, query_construction
# %%

class ResultProcessor(object):
    """
    {Description}

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
        self.alias_mapping = query_construction.abbr_option[workload]

    # def build_correlation_graph(self, query_instance: node_query.QueryInstance = None):
    def build_correlation_graph(self, query_meta, card_dict):
        """
        {Description}
    
        Args:
            query_meta: 
            card_dict:
        Returns:
            attr_dict: 
            graph_dict:
        """
        alias_mapping = query_construction.abbr_option[self.workload]

        attr_dict = {}
        graph_dict = {
            "node_dict": defaultdict(list),
            "edge_set": set()
        }
        # local_parser = workload_parser.SQLParser(query_instance.query_text, self.workload)

        subquery_true, single_table_true = card_dict["true"]["subquery"], card_dict["true"]["single_table"]
        subquery_est, single_table_est = card_dict["estimation"]["subquery"], card_dict["estimation"]["single_table"]

        # 对齐subquery_true和single_table_true
        subquery_diff, single_table_diff = set(subquery_est.keys()).difference(subquery_true.keys()), \
            set(single_table_est.keys()).difference(single_table_true.keys())

        for k in subquery_diff:
            subquery_true[k] = None
        
        for k in single_table_diff:
            single_table_true[k] = None

        card_pair_list = list(subquery_true.items()) + list(single_table_true.items()) + \
                        list(subquery_est.items()) + list(single_table_est.items())
        
        field_list = ["true_card" for _ in range(len(subquery_true) + len(single_table_true))] + \
                     ["est_card" for _ in range(len(subquery_est) + len(single_table_est))]
        
        global_cnt = -1

        # 构造attr_dict
        repr2id = dict()
        for (q_repr, card), field in zip(card_pair_list, field_list):
            # print(f"build_correlation_graph: q_repr = {q_repr}. field = {field}. card = {card}")
            if q_repr not in repr2id:
                global_cnt += 1
                curr_node_id = global_cnt
                repr2id[q_repr] = curr_node_id
            else:
                curr_node_id = repr2id[q_repr]

            local_meta = meta_info.generate_subquery_meta(query_meta, \
                alias_list=q_repr, alias_mapping=alias_mapping)
            
            try:
                alias_tuple = tuple(sorted([self.alias_mapping[t] for t in local_meta[0]]))
            except TypeError as e:
                print(f"build_correlation_graph: meet TypeError. local_meta = {local_meta}")
                raise e
            # print(f"build_correlation_graph: alias_tuple = {alias_tuple}. local_meta = {local_meta}.")
            
            if curr_node_id not in attr_dict:
                attr_dict[curr_node_id] = {
                    # "query": query_construction.construct_origin_query(\
                    #     query_meta=local_meta, workload=self.workload),
                    "query": "",
                    "meta": local_meta,
                    "alias_tuple": alias_tuple,
                }
            attr_dict[curr_node_id][field] = card

        # print("build_correlation_graph: len(attr_dict) = {}. attr_dict.keys = {}".\
        #       format(len(attr_dict), attr_dict.keys()))
        
        # print("build_correlation_graph: print alias_tuple info")
        # for k, v in attr_dict.items():
        #     print("node_id = {}. alias_tuple = {}.".format(k, v['alias_tuple']))

        # 构造graph_dict
        # for id1, id2 in enumerate(list(attr_dict.keys())):
        for id1, id2 in permutations(list(attr_dict.keys()), 2):
            if id1 == id2:
                continue

            alias_set1, alias_set2 = \
                set(attr_dict[id1]['alias_tuple']), set(attr_dict[id2]['alias_tuple'])
            
            # 判断是否是一个子集包含的关系，有的话就添加边
            if set(alias_set1).issubset(set(alias_set2)) and len(set(alias_set1)) + 1 == len(set(alias_set2)):
                # print("build_correlation_graph: id1 = {}. id2 = {}. alias_set1 = {}. alias_set2 = {}.".\
                #       format(id1, id2, alias_set1, alias_set2))
                # 更新node_dict
                graph_dict['node_dict'][id1].append(id2)
                # 更新edge_set
                graph_dict['edge_set'].add((id1, id2))

        # for k, v in attr_dict.items():
        #     print(f"build_correlation_graph: id = {k}. alias_tuple = {v['alias_tuple']}.")
        return attr_dict, graph_dict


    def build_graph_batch(self, instance_list):
        """
        构造输出的字典
    
        Args:
            arg1:
            arg2:
        Returns:
            out_dict:
            return2:
        """
        # time_start = time.time()
        result_list = []
        for idx, instance in enumerate(instance_list):
            query_meta, card_dict = instance
            attr_dict, graph_dict = self.build_correlation_graph(query_meta, card_dict)
            result_list.append((attr_dict, graph_dict))
        # time_end = time.time()

        # length = len(instance_list)
        # delta_time = (time_end - time_start) * 1000
        # print(f"build_graph_batch: delta_time = {delta_time:.2f}. "\
        #     f"length = {length}. average_time = {delta_time / length:.2f}")
        return result_list
    
# %%
def process_instance(workload, query_meta, card_dict):
    """
    {Description}

    Args:
        workload:
        query_meta:
        card_dict:
    Returns:
        attr_dict:
        graph_dict:
    """
    local_processor = ResultProcessor(workload)
    attr_dict, graph_dict = \
        local_processor.build_correlation_graph(query_meta, card_dict)

    return attr_dict, graph_dict


# %%
def process_instance_list(workload, instance_list):
    """
    {Description}
    
    Args:
        workload:
        instance_list:
    Returns:
        result_list:
        res2:
    """
    # time_start = time.time()
    local_processor = ResultProcessor(workload)
    instance_list = list(instance_list)
    result_list = local_processor.build_graph_batch(instance_list)
    # time_end = time.time()

    # length = len(instance_list)
    # delta_time = (time_end - time_start) * 1000
    # print(f"process_instance_list: delta_time = {delta_time:.2f}. "\
    #       f"length = {length}. average_time = {delta_time / length:.2f}")
    return result_list

# %%
