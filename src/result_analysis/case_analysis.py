#!/usr/bin/env python
# -*- coding: utf-8 -*-


# %%

import numpy as np
import pandas as pd
from os.path import join as p_join
from matplotlib import pyplot as plt
import json, os, time, re, math
import pickle
import graphviz
from copy import deepcopy
from itertools import combinations

# %%
from workload import physical_plan_info
from query import query_exploration
from plan import join_analysis
from data_interaction import postgres_connector
from utility import utils

from query import query_construction
# %%

class MultiCaseAnalyzer(object):
    """
    example_format = ("list", "tree", ("query", "meta", "card_dict", "target_table"))
    

    Members:
        field1:
        field2:
    """

    def __init__(self, workload, result_format):
        """
        {Description}

        Args:
            workload:
            result_format:
        """
        self.workload = workload
        self.result_format = result_format
        self.multi_case_result = None

    def load_result(self, data_input, mode = "path"):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        def parse_object(curr_obj, format_list):
            # 解析具体的对象，根据当前的format选择
            curr_format = format_list[0]
            curr_result = None

            if isinstance(curr_format, str) == True:
                if curr_format == "list":
                    curr_result = []
                    for item in curr_obj:
                        curr_result.append(parse_object(item, format_list[1:]))
                elif curr_format == "tree":
                    curr_result = TreeAnalyzer(self.workload, curr_obj)
                else:
                    raise ValueError(f"parse_object: Unsupported format value: {curr_format}")
            elif isinstance(curr_format, tuple) == True:
                # 
                curr_result = CaseAnalyzer(*curr_result, self.workload)
            else:
                raise TypeError(f"parse_object: Unsupported format type: {type(curr_format)}")
            
            return curr_result

        assert mode in ("path", "object")

        if mode == "path":
            data_path = data_input
            res_pickle = utils.load_pickle(data_path)
            # self.multi_case_result = parse_object(res_pickle, self.result_format)
        elif mode == "object":
            res_pickle = data_input
        
        self.multi_case_result = parse_object(res_pickle, self.result_format)
        return self.multi_case_result

    def show_result(self,):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        def print_indent(level):
            factor = 5
            print(" "*level*factor, end="")

        def show_object(curr_result, format_list, level, index):
            # 展示结果
            curr_format = format_list[0]
            if isinstance(curr_format, str) == True:
                if curr_format == "list":
                    #
                    print_indent(level)
                    print(f"{index} @ list_object")

                    for idx, item in enumerate(curr_result):
                        show_object(item, format_list[1:], level + 1, idx)
                elif curr_format == "tree":
                    #
                    curr_result: TreeAnalyzer = curr_result
                    key_list = sorted(curr_result.case_dict.keys())

                    print_indent(level)
                    print(f"{index} @ tree_object")

                    for idx, key in enumerate(key_list):
                        local_analyzer: CaseAnalyzer = curr_result.case_dict[key]
                        info_str = f"node_repr: {local_analyzer.case_repr()}. p_error: {local_analyzer.p_error:.2f}."
                        print_indent(level + 1)
                        print(f"{idx} @ {info_str}")
                else:
                    raise ValueError(f"parse_object: Unsupported format value: {curr_format}")
            elif isinstance(curr_format, tuple) == True:
                curr_result: CaseAnalyzer = curr_result
                info_str = f"node_repr: {local_analyzer.case_repr()}. p_error: {local_analyzer.p_error:.2f}."
                print_indent(level)
                print(f"{idx} @ {info_str}")
            else:
                raise TypeError(f"parse_object: Unsupported format type: {type(curr_format)}")
            
        show_object(self.multi_case_result, self.result_format, 0, 0)


    def get_obj_by_index(self, index_list):
        """
        根据索引，获得单个case实例
        
        Args:
            index_list: 多层索引组成的列表
            arg2:
        Returns:
            res1:
            res2:
        """
        res_obj = None

        def find_obj(curr_result, format_list, index_list):
            try:
                curr_format, curr_index = format_list[0], index_list[0]
            except IndexError:
                # index结束了，直接返回当前结果
                return curr_result
            
            if isinstance(curr_format, str) == True:
                if curr_format == "list":
                    #
                    item = curr_result[curr_index]
                    return find_obj(item, format_list[1:], index_list[1:])
                elif curr_format == "tree":
                    #
                    curr_result: TreeAnalyzer = curr_result
                    key_list = sorted(curr_result.case_dict.keys())
                    key = key_list[curr_index]
                    local_analyzer: CaseAnalyzer = curr_result.case_dict[key]
                    curr_result = local_analyzer
                else:
                    raise ValueError(f"parse_object: Unsupported format value: {curr_format}")
            elif isinstance(curr_format, tuple) == True:
                curr_result: CaseAnalyzer = curr_result
            else:
                raise TypeError(f"parse_object: Unsupported format type: {type(curr_format)}")
            
            return curr_result
            
        res_obj = find_obj(self.multi_case_result, self.result_format, index_list)
        return res_obj


# %%

class CaseAnalyzer(object):
    """
    单个例子的分析器

    Members:
        field1:
        field2:
    """

    def __init__(self, query: str, meta: tuple, result: tuple, card_dict: dict, workload: str):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        self.workload = workload
        try:
            card_dict = utils.card_dict_normalization(card_dict)
        except TypeError as e:
            print(f"CaseAnalyzer.__init__: meet TypeError. workload = {workload}. meta = {meta}. \ncard_dict = {card_dict}.")
            raise e
        
        if (query.startswith("SELECT") or query.startswith("select")) == False:
            query = query_construction.construct_origin_query(meta, workload)

        self.query, self.meta = query, meta
        self.result, self.card_dict = result, card_dict

        # if with_instance == True:
        self.true_physical, self.estimation_physical, self.plan_true, \
            self.plan_estimation = self.generate_instance()

    @utils.timing_decorator
    def generate_instance(self,):
        """
        生成相关的查询计划实例

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        self.db_conn: postgres_connector.Connector = \
            postgres_connector.connector_instance(workload=self.workload)
        query_ctrl = query_exploration.QueryController(db_conn=self.db_conn, workload=self.workload)
        
        query_ctrl.set_query_instance(self.query, self.meta)

        subquery_true, single_table_true, subquery_estimation, \
            single_table_estimation = utils.extract_card_info(self.card_dict)

        true_plan: dict = query_ctrl.get_plan_by_external_card(subquery_true, single_table_true)
        estimation_plan: dict = query_ctrl.get_plan_by_external_card(\
            subquery_estimation, single_table_estimation)

        db_conn = query_ctrl.db_conn

        true_physical = physical_plan_info.PhysicalPlan(query_text = self.query, \
            plan_dict = true_plan, db_conn = db_conn)
        estimation_physical = physical_plan_info.PhysicalPlan(query_text = self.query, \
            plan_dict = estimation_plan, db_conn = db_conn)

        plan_true = true_physical.get_specific_plan(subquery_true, single_table_true)
        plan_estimation = estimation_physical.get_specific_plan(subquery_true, single_table_true)

        self.true_cost = true_physical.get_plan_cost(subquery_true, single_table_true)
        self.estimation_cost = estimation_physical.get_plan_cost(subquery_true, single_table_true)

        self.p_error = self.estimation_cost / self.true_cost

        # 2024-03-16: 用于debug
        # print(f"generate_instance: query_meta = {self.meta}.")
        # print(f"generate_instance: card_dict = {utils.display_card_dict(self.card_dict)}.")
        # print(f"generate_instance: est_cost = {self.estimation_cost:.2f}. true_cost = {self.true_cost:.2f}. p_error = {self.p_error:.2f}.")

        # 20240222: 
        if len(self.result) != 3:
            self.result = self.p_error, self.estimation_cost, self.true_cost

        return true_physical, estimation_physical, plan_true, plan_estimation


    def infer_error_mode(self,):
        """
        推测当前查询的错误模式
    
        Args:
            arg1:
            arg2:
        Returns:
            mode: ("over-estimation", "under-estimation")
            return2:
        """
        mode = None
        subquery_true, _, subquery_est, _ = utils.extract_card_info(self.card_dict)
        top_query_key = utils.dict_max(subquery_true, val_func=lambda a: len(a[0]))[0]
        # print(f"infer_error_mode: top_query_key = {top_query_key}.")

        card_error = (subquery_true[top_query_key] + 1.0) / (subquery_est[top_query_key] + 1.0)

        if card_error >= 1.25:
            mode = "under-estimation"
        elif card_error <= 0.8:
            mode = "over-estimation"
        else:
            mode = "unknown"

        return mode


    def node_info(self, node_item):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        item = node_item
        return f"{item['repr']}\nTrueCard: {item['card_true']}\nEstCard: {item['card_est']}"\
               f"\nCost: {item['cost']}\nType: {item['physical']}"

    def case_repr(self,):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        alias_mapping = query_construction.abbr_option[self.workload]
        return ",".join([alias_mapping[s] for s in self.meta[0]])

    def build_node_dict(self, cost_tuple, join_dict, scan_dict, \
            subquery_true, single_table_true, subquery_est, single_table_est):
        """
        构建节点属性的字典
        {
            "cost":
            "physical":
            "cardinality":
        }
    
        Args:
            cost_tuple: 
            join_dict: 
            scan_dict: 
            subquery_dict: 
            single_table_dict:
        Returns:
            node_attr_dict:
            return2:
        """

        _, join_cost, scan_cost = cost_tuple
        join_cost = utils.dict_apply(join_cost, \
            lambda a: tuple(a.split()), mode = "key")
        join_dict = utils.dict_apply(join_dict, \
            lambda a: tuple(a.split()), mode = "key")

        # print(f"cost_scan = {scan_cost.keys()}")
        # print(f"cost_join = {join_cost.keys()}")

        # print(f"join_dict = {join_dict.keys()}")
        # print(f"scan_dict = {scan_dict.keys()}")
        # print(f"subquery_dict = {subquery_dict.keys()}")
        # print(f"single_table_dict = {single_table_dict.keys()}")

        # 
        node_attr_dict = {}

        scan_key_list = list(scan_cost.keys())
        join_key_list = list(join_cost.keys())

        for k in scan_key_list:
            node_attr_dict[k] = {
                "repr": k,
                "card_true": single_table_true[k],
                "card_est": single_table_est[k],
                "physical": scan_dict[k],
                "cost": scan_cost[k]
            }

        for k in join_key_list:
            node_attr_dict[k] = {
                "repr": " ".join(k), 
                "card_true": subquery_true[k],
                "card_est": subquery_est[k],
                "physical": join_dict[k],
                "cost": join_cost[k]
            }

        return node_attr_dict

    def plot_plan_comparison(self,):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        jo_analyzer_true = join_analysis.JoinOrderAnalyzer(\
            join_order_str=self.true_physical.leading)
        jo_analyzer_estimation = join_analysis.JoinOrderAnalyzer(\
            join_order_str=self.estimation_physical.leading)

        subquery_true, single_table_true, subquery_estimation, \
            single_table_estimation = utils.extract_card_info(self.card_dict)
        
        plan_cost_true = query_exploration.get_plan_component_cost(self.plan_true)
        plan_cost_estimation = query_exploration.get_plan_component_cost(self.plan_estimation)

        join_true, scan_true = self.true_physical.join_ops, self.true_physical.scan_ops
        true_node_dict = self.build_node_dict(plan_cost_true, join_true, \
            scan_true, subquery_true, single_table_true, subquery_estimation, single_table_estimation)

        join_est, scan_est = self.estimation_physical.join_ops, self.estimation_physical.scan_ops
        estimation_node_dict = self.build_node_dict(plan_cost_estimation, join_est, \
            scan_est, subquery_true, single_table_true, subquery_estimation, single_table_estimation)

        true_edges = jo_analyzer_true.get_all_edges()
        estimation_edges = jo_analyzer_estimation.get_all_edges()

        # 创建图
        true_graph = graphviz.Graph("true plan graph")
        est_graph = graphviz.Graph("est plan graph")

        # 创建节点
        for k, v in true_node_dict.items():
            true_graph.node(name=str(k), label=self.node_info(v))

        for k, v in estimation_node_dict.items():
            est_graph.node(name=str(k), label=self.node_info(v))

        # 创建边
        for src_node, dst_node in true_edges:
            true_graph.edge(str(src_node), str(dst_node))

        for src_node, dst_node in estimation_edges:
            est_graph.edge(str(src_node), str(dst_node))

        # return true_node_dict, estimation_node_dict, true_inter_tree, estimation_inter_tree
        return true_graph, est_graph
    
    def get_plan_join_order(self, mode):
        """
        获得查询计划的连接顺序
    
        Args:
            mode:
            arg2:
        Returns:
            flag: 这个查询计划是否为zig-zag类型的
            join_order_list: 连接顺序列表
        """
        flag, join_order_list = False, []

        assert mode in ("true", "estimation"), f"get_plan_join_order: mode = {mode}"

        if mode == "true":
            jo_analyzer = join_analysis.JoinOrderAnalyzer(\
                join_order_str=self.true_physical.leading)
        elif mode == "estimation":
            jo_analyzer = join_analysis.JoinOrderAnalyzer(\
                join_order_str=self.estimation_physical.leading)

        flag = not jo_analyzer.is_bushy()
        # join_order_list = jo_analyzer.get_leading_tables()
        join_order_list = jo_analyzer.get_leading_order()
        
        return flag, join_order_list
    
    def construct_submeta(self, alias_list):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        alias_mapping = query_construction.abbr_option[self.workload]
        schema_list, filter_list = [], []

        for s in self.meta[0]:
            if alias_mapping[s] in alias_list:
                schema_list.append(s)

        for f in self.meta[1]:
            if f[0] in alias_list:
                filter_list.append(f)

        return schema_list, filter_list


    def get_leading_meta(self, leading_num, mode = "true"):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            query_meta:
            valid:
        """
        alias_list, flag = self.get_leading_alias(leading_num, mode)

        if flag == True:
            return self.construct_submeta(alias_list), flag
        else:
            return ([], []), flag

    def get_leading_alias(self, leading_num, mode = "true"):
        """
        {Description}
    
        Args:
            leading_num:
            mode:
        Returns:
            alias_list:
            flag:
        """
        assert mode in ("true", "estimation")
        alias_list = []

        valid_cnt = 0
        if mode == "true":
            join_key_list = self.true_physical.join_ops.keys()
        elif mode == "estimation":
            join_key_list = self.estimation_physical.join_ops.keys()
        
        for k in join_key_list:
            if len(k.split(" ")) == leading_num:
                alias_list = k.split(" ")
                valid_cnt += 1

        print(f"get_leading_alias: join_key_list = {join_key_list}. valid_cnt = {valid_cnt}. alias_list = {alias_list}")
        flag = valid_cnt == 1
        return alias_list, flag
    
    def card_dict_filter(self, alias_list):
        """
        获得基数字典的子集
        
        Args:
            alias_list:
            arg2:
        Returns:
            sub_card_dict:
            res2:
        """
        subquery_true, single_table_true, subquery_est, \
            single_table_est = utils.extract_card_info(self.card_dict)

        subquery_true_sub = utils.dict_subset(subquery_true, \
            lambda a: set(a).issubset(alias_list), mode="key")
        single_table_true_sub = utils.dict_subset(single_table_true, \
            lambda a: a in alias_list, mode="key")

        subquery_est_sub = utils.dict_subset(subquery_est, \
            lambda a: set(a).issubset(alias_list), mode="key")
        single_table_est_sub = utils.dict_subset(single_table_est, \
            lambda a: a in alias_list, mode="key")

        sub_card_dict = utils.pack_card_info(subquery_true_sub, \
            single_table_true_sub, subquery_est_sub, single_table_est_sub)
        
        return sub_card_dict
    

    def get_column_info(self, ):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        pass


# %% 

class FuzzyCaseAnalyzer(object):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, query: str, meta: tuple, result: tuple, card_dict: dict, workload: str):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        self.workload = workload
        # card_dict = utils.card_dict_normalization(card_dict)
        
        if (query.startswith("SELECT") or query.startswith("select")) == False:
            query = query_construction.construct_origin_query(meta, workload)

        self.query, self.meta = query, meta
        self.result, self.card_dict = result, card_dict

        self.true_card_keys, self.est_card_keys = self.eval_card_dict_state()


    def sample_on_card_dict(self, out_case_num):
        """
        {Description}

        Args:
            out_case_num:
            arg2:
        Returns:
            case_analyzer_list:
            return2:
        """
        if out_case_num * 3 > self.total_comb_num:
            out_case_num = self.total_comb_num // 3

        case_analyzer_list = []

        for _ in range(out_case_num):
            out_card_dict, signature = self.one_instance_sample()
            local_analyzer = CaseAnalyzer(self.query, \
                self.meta, (), out_card_dict, self.workload)
            case_analyzer_list.append(local_analyzer)

        return case_analyzer_list

    def card_dict_sub(self, card_ref: dict, card_new: dict):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        card_out = deepcopy(card_ref)

        for k, v in card_new.items():
            card_out[k] = v

        return card_out

    def one_instance_sample(self,):
        """
        通过采样生成一个实例，
    
        Args:
            arg1:
            arg2:
        Returns:
            out_card_dict:
            tuple_signature:
        """
        list_signature = []

        subquery_true, single_table_true, subquery_est, single_table_est = \
            utils.extract_card_info(self.card_dict)
        #
        subquery_true_sub = {}
        for key in self.true_card_keys:
            out_val = np.random.choice(subquery_true[key])
            subquery_true_sub[key] = out_val
            list_signature.append(out_val)
        #
        subquery_est_sub = {}
        for key in self.est_card_keys:
            out_val = np.random.choice(subquery_est[key])
            subquery_est_sub[key] = out_val        
            list_signature.append(out_val)

        subquery_true_out = self.card_dict_sub(subquery_true, subquery_true_sub)
        subquery_est_out = self.card_dict_sub(subquery_est, subquery_est_sub)

        out_card_dict = utils.pack_card_info(subquery_true_out, 
            deepcopy(single_table_true), subquery_est_out, deepcopy(single_table_est))

        return out_card_dict, tuple(list_signature)

    def eval_card_dict_state(self,):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        true_card_keys, est_card_keys = [], []

        subquery_true, single_table_true, subquery_est, single_table_est = \
            utils.extract_card_info(self.card_dict)
        
        total_comb_num = 1
        for k, v in subquery_true.items():
            if isinstance(v, (tuple, list)):
                true_card_keys.append(k)
                total_comb_num *= len(v)

        for k, v in subquery_est.items():
            if isinstance(v, (tuple, list)):
                est_card_keys.append(k)
                total_comb_num *= len(v)

        self.total_comb_num = total_comb_num
        return true_card_keys, est_card_keys


# %%

class TreeAnalyzer(object):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, workload, instance_list):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        self.workload = workload
        self.alias_mapping = query_construction.abbr_option[workload]
        self.instance_list = instance_list
        self.case_dict = {}
        self.edge_list = []

        for item in instance_list:
            if len(item) == 4:
                query_text, query_meta, card_dict, target_table = item
            elif len(item) == 3:
                query_text, query_meta, card_dict = item
                target_table = ""
            elif len(item) == 5:
                query_text, query_meta, result, card_dict, target_table = item
            else:
                # print(f"TreeAnalyzer.__init__: item = {item}.")
                print(f"TreeAnalyzer.__init__: print item elements.")
                for idx, i in enumerate(item):
                    print(f"TreeAnalyzer.__init__: item[{idx}] = {i}")

                raise ValueError(f"TreeAnalyzer.__init__: len(item) = {len(item)}")
            
            # elif len(item) == 4:
            #     query_text, query_meta, result, card_dict = item
            local_analyzer = CaseAnalyzer(query_text, query_meta, (), card_dict, workload)

            try:
                target_alias = self.alias_mapping[target_table]
            except KeyError:
                target_alias = ""

            node_repr = self.get_node_repr(query_meta, target_alias)
            # node_repr = tuple([self.alias_mapping[s] for s in query_meta[0]]), target_alias
            self.case_dict[node_repr] = local_analyzer

        self.construct_tree()

    def get_node_repr(self, query_meta, target_alias):
        """
        {Description}
    
        Args:
            query_meta: 
            target_alias:
        Returns:
            return1:
            return2:
        """
        res_repr = tuple([self.alias_mapping[s] \
            for s in query_meta[0]]), target_alias
        return res_repr
        
    def construct_tree(self,):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        key_list = self.case_dict.keys()

        for key1, key2 in combinations(key_list, 2):
            set1, target1 = set(key1[0]), key1[1]
            set2, target2 = set(key2[0]), key2[1]

            if set1.union(set2) == set2 and len(set1) + 1 == len(set2) \
                and set2.difference(set1).pop() == target2:
                self.edge_list.append((key1, key2))

        return self.edge_list

    def add_new_node(self, node_repr):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        key_list = self.case_dict.keys()
        set2, target2 = set(node_repr[0]), node_repr[1]

        for key1 in key_list:
            set1, target1 = set(key1[0]), key1[1]

            # 正反两个方向都要判断
            if set1.union(set2) == set2 and len(set1) + 1 == len(set2) \
                and set2.difference(set1).pop() == target2:
                self.edge_list.append((key1, node_repr))

            if set2.union(set1) == set1 and len(set2) + 1 == len(set1) \
                and set1.difference(set2).pop() == target1:
                self.edge_list.append((node_repr, key1))

        return self.edge_list
    

    def plot_subtree(self, repr_subset):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        pass
    
    def make_label(self, node_key):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        alias_tuple, target_alias = node_key
        local_analyzer: CaseAnalyzer = self.case_dict[node_key]

        out_str = f"{','.join(alias_tuple)}\np_error={local_analyzer.p_error:.2f}"

        return out_str

    def plot_tree(self,):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        graph = graphviz.Graph()
        
        for k, v in self.case_dict.items():
            # graph.node(str(k), label=str(k))
            graph.node(str(k), label=self.make_label(k))

        for src_node, dst_node in self.edge_list:
            graph.edge(str(src_node), str(dst_node))

        return graph


    def get_root_node(self,):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            root_analyzer:
            res2:
        """
        for k, v in self.case_dict.items():
            alias_tuple, target_alias = k
            if target_alias == "":
                return v
        return None

    def complement_root(self,):
        """
        考虑root缺失的情况，补全树的根节点
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        if self.get_root_node() is not None:
            print("complement_root: root_node already exists")

        analyzer_list = self.case_dict.values()
        element_list = [(analyzer.meta, analyzer.card_dict) for analyzer in analyzer_list]
        
        if len(element_list) < 2:
            print("complement_root: len(element_list) < 2")
            return None
        else:
            query_meta1, card_dict1 = element_list[0]
            query_meta2, card_dict2 = element_list[1]

            root_meta, root_card_dict = self.node_intersection(query_meta1, card_dict1, query_meta2, card_dict2)
            
            for new_meta, new_card_dict in element_list[2:]:
                root_meta, root_card_dict = self.node_intersection(\
                    root_meta, root_card_dict, new_meta, new_card_dict)

            root_query = query_construction.construct_origin_query(root_meta, self.workload)
            analyzer = CaseAnalyzer(root_query, root_meta, (), root_card_dict, self.workload)
            node_repr = self.get_node_repr(root_meta, "")

            self.case_dict[node_repr] = analyzer
            self.add_new_node(node_repr)

    def node_intersection(self, query_meta1, card_dict1, query_meta2, card_dict2):
        """
        {Description}
        
        Args:
            query_meta1:
            card_dict1: 
            query_meta2: 
            card_dict2:
        Returns:
            res1:
            res2:
        """
        out_meta = utils.meta_intersection(query_meta1, query_meta2)

        subquery_true1, single_table_true1, subquery_est1, \
            single_table_est1 = utils.extract_card_info(card_dict1)
        subquery_true2, single_table_true2, subquery_est2, \
            single_table_est2 = utils.extract_card_info(card_dict2)

        subquery_true_out = utils.dict_intersection(subquery_true1, subquery_true2)
        single_table_true_out = utils.dict_intersection(single_table_true1, single_table_true2)
        subquery_est_out = utils.dict_intersection(subquery_est1, subquery_est2)
        single_table_est_out = utils.dict_intersection(single_table_est1, single_table_est2)

        out_dict = utils.pack_card_info(subquery_true_out, \
            single_table_true_out, subquery_est_out, single_table_est_out)

        return out_meta, out_dict
# %%

def construct_case_instance(query_meta, card_dict, workload) -> CaseAnalyzer:
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    res_analyzer = CaseAnalyzer("", query_meta, (), card_dict, workload)
    return res_analyzer

# %%
