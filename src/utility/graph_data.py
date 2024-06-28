#!/usr/bin/env python
# -*- coding: utf-8 -*-


# %%
import numpy as np
import pandas as pd
from os.path import join as p_join
from matplotlib import pyplot as plt
import pickle, psutil

from plan import node_query
from query import query_construction
from utility import utils, workload_parser
from collections import defaultdict
from itertools import permutations


# %%

class InstanceAdapter(object):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, mode, workload):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        self.workload = workload
        self.mode = mode

    def process_data(self, data_input):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        if self.mode == "query_instance":
            return self.process_query_instance(data_input)
        elif self.mode == "tuple":
            return self.process_tuple(data_input)
        else:
            raise ValueError(f"process_data: mode = {self.mode}.")

    def process_query_instance(self, data_input: node_query.QueryInstance):
        """
        {Description}

        Args:
            data_input:
            arg2:
        Returns:
            query_parser:
            subquery_true: 
            single_table_true:
            subquery_est: 
            single_table_est:
        """
        query_instance = data_input
        local_parser = workload_parser.SQLParser(query_instance.query_text, self.workload)
        subquery_true, single_table_true = \
            query_instance.true_card_dict, query_instance.true_single_table
        subquery_est, single_table_est = \
            query_instance.estimation_card_dict, query_instance.estimation_single_table

        return local_parser, subquery_true, single_table_true, subquery_est, single_table_est
    
    def process_tuple(self, data_input: tuple):
        """
        {Description}

        Args:
            data_input:
            arg2:
        Returns:
            query_parser:
            subquery_true: 
            single_table_true:
            subquery_est: 
            single_table_est:        
        """
        if len(data_input) == 4:
            query_text, query_meta, result, card_dict = data_input
        elif len(data_input) == 5:
            query_text, query_meta, result, card_dict, start_time = data_input

        local_parser = workload_parser.SQLParser(query_text, self.workload)
        subquery_true, single_table_true, subquery_est, \
              single_table_est = utils.extract_card_info(card_dict)
        
        return local_parser, subquery_true, single_table_true, \
            subquery_est, single_table_est
    
# %%

class ResultProcessor(object):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, workload, mode = "tuple"):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        self.workload = workload
        self.adapter = InstanceAdapter(mode, workload)
        self.alias_mapping = query_construction.abbr_option[workload]

    def result_fetch(self,):
        """
        获取结果
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        res_list = []

        for k, v in self.proc_dict.items():
            try:
                f_path = p_join(self.result_dir, v['out_name'])
                with open(f_path, "rb") as f_in:
                    query_instance = pickle.load(f_in)
                    res_list.append(query_instance)
            except FileNotFoundError as e:
                continue

        return res_list
    
    def build_correlation_graph(self, data_input):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            attr_dict: 
            graph_dict:
        """
        attr_dict = {}
        graph_dict = {
            "node_dict": defaultdict(list),
            "edge_set": set()
        }

        local_parser, subquery_true, single_table_true, \
            subquery_est, single_table_est = self.adapter.process_data(data_input)

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

            if isinstance(q_repr, str) == True:
                local_meta = local_parser.generate_subquery_meta(alias_list=[q_repr,])
            else:
                local_meta = local_parser.generate_subquery_meta(alias_list=q_repr)

            alias_tuple = tuple(sorted([self.alias_mapping[t] for t in local_meta[0]]))
            # print(f"build_correlation_graph: alias_tuple = {alias_tuple}. local_meta = {local_meta}.")
            # print(f"build_correlation_graph: q_repr = {q_repr}. local_meta = {local_meta}.")
            if curr_node_id not in attr_dict:
                attr_dict[curr_node_id] = {
                    "query": query_construction.construct_origin_query(\
                        query_meta=local_meta, workload=self.workload),
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


    def construct_output_dict(self,):
        """
        构造输出的字典
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        out_dict = {}

        for idx, instance in enumerate(self.query_instance_list):
            attr_dict, graph_dict = self.build_correlation_graph(instance)
            out_path = p_join(self.result_dir, "data", f"{idx}.pkl")

            # 转换成普通的dict类型再导出
            utils.dump_pickle(res_obj=(dict(attr_dict), dict(graph_dict)), data_path=out_path)
            out_dict[idx] = (attr_dict, graph_dict, out_path)
            
        return out_dict
    
# %%
