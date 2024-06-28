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
from multiprocessing import Pool
from collections import defaultdict

import hashlib, shutil 
from utility import generator, utils
from data_interaction import data_management, mv_management
from grid_manipulation import grid_construction, grid_preprocess, grid_advance, grid_analysis
from query import query_construction, query_exploration, ce_injection
import base
from itertools import product
from plan import node_query, plan_analysis

# %%

class SearchInitilization(object):
    """
    搜索过程的初始化，确定一开始值得探索的节点

    Members:
        field1:
        field2:
    """

    def __init__(self, schema_total, ce_handler, workload, data_manager_ref:data_management.DataManager, \
        query_ctrl_ref: query_exploration.QueryController, split_budget = 100):
        """
        {Description}

        Args:
            schema_total: 所有合法的schema列表
            ce_handler: 基数估计器的句柄
            workload: 负载
            data_manager_ref: 数据管理器的引用
            query_ctrl_ref: 查询控制器的引用
        """
        self.schema_total = schema_total
        if len(schema_total) == 0:
            raise ValueError("schema_total is empty!")
        
        self.ce_handler = ce_handler
        self.workload = workload
        self.data_manager = data_manager_ref
        self.query_ctrl = query_ctrl_ref
        self.alias_mapping = query_construction.abbr_option[workload]
        self.alias_inverse = {}
        self.db_conn = query_ctrl_ref.db_conn   # 设置数据库连接，用以获取真实基数

        self.bins_builder = grid_preprocess.BinsBuilder(workload=self.workload, \
            data_manager_ref=data_manager_ref, default_split_budget = split_budget)      #
        for k, v in self.alias_mapping.items():
            self.alias_inverse[v] = k

        self.get_all_schema_combinations()


    def join_order_priori(self, table_subset, num = 50):
        """
        获得关于join_order的先验信息，关于table_subset
    
        Args:
            table_subset: 整个workload下的数据表组成
            arg2:
        Returns:
            query_list:
            meta_list:
            join_order_list:
        """
        global_generator = generator.QueryGenerator(schema_list = table_subset, 
                    dm_ref=self.data_manager, bins_builder=self.bins_builder, workload=self.workload)
        
        query_list, meta_list = \
            global_generator.generate_batch_queries(num = num, with_meta=True)
        
        # 装填外部信息
        workload = self.workload
        data_manager = data_management.DataManager(wkld_name=workload)
        mv_manager = mv_management.MaterializedViewManager(workload=workload)
        external_dict = {
            "data_manager": data_manager,
            "mv_manager": mv_manager,
            "ce_handler": ce_injection.PGInternalHandler(workload=workload),
            "query_ctrl": query_exploration.QueryController(workload=workload),
            "multi_builder": grid_preprocess.MultiTableBuilder(workload = workload, \
                data_manager_ref = data_manager, mv_manager_ref = mv_manager, dynamic_config={})
        }

        # ts = time.time()
        join_order_list = [node_query.get_query_join_order(workload=self.workload, query_meta=meta_info,
                                external_dict=external_dict) for meta_info in meta_list]
        # te = time.time()
        # print("delta_time = {}.".format(te - ts))

        self.join_order_list = join_order_list
        return query_list, meta_list, join_order_list


    def join_order_type_analysis(self, join_order_list):
        """
        连接顺序的分析(是bushy还是zigzag的)
        
        Args:
            join_order_list:
            arg2:
        Returns:
            bushy_num:
            bushy_list:
            zigzag_num:
            zigzag_list:
        """
        bushy_list = []
        zigzag_list = []
        bushy_num, zigzag_num = 0, 0
        for idx, jo_str in enumerate(join_order_list):
            analyzer = plan_analysis.JoinOrderAnalyzer(join_order_str=jo_str)
            if analyzer.is_bushy() == True:
                bushy_num += 1
                bushy_list.append((idx, jo_str))
            else:
                zigzag_num += 1
                zigzag_list.append((idx, jo_str))

        return bushy_num, bushy_list, zigzag_num, zigzag_list


    def join_order_batch_analysis(self, meta_list, join_order_list):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        all_tables = meta_list[0][0]
        all_alias = [self.alias_mapping[t] for t in all_tables]
        alias_position_dict = {}
        for alias in all_alias:
            alias_position_dict[alias] = []

        for jo_str in join_order_list:
            jo_analyzer = plan_analysis.JoinOrderAnalyzer(join_order_str=jo_str)
            for k, v in jo_analyzer.level_dict.items():
                alias_position_dict[k].append(v)

        return alias_position_dict



    def get_join_position_status(self, table_subset):
        """
        获得join位置的状态
        
        Args:
            table_subset:
            arg2:
        Returns:
            score_list:
            res2:
        """
        _, _, _, zigzag_list = self.join_order_type_analysis(self.join_order_list)
        alias_subset = [self.alias_mapping[t] for t in table_subset]
        score_list = []

        for idx, jo_str in enumerate(zigzag_list):
            local_analyzer = plan_analysis.JoinOrderAnalyzer(jo_str)
            score = local_analyzer.table_subset_score(alias_subset=alias_subset)
            score_list.append(score)

        return score_list

    def restore_origin_names(self, sub_alias_list):
        """
        {Description}
        
        Args:
            sub_alias_list:
        Returns:
            sub_schema_list:
        """
        sub_schema_list = []

        for sub_alias in sub_alias_list:
            local_schema = []
            for alias in sub_alias:
                local_schema.append(self.alias_inverse[alias])
            # 排个序
            sub_schema_list.append(sorted(local_schema))
        
        return sub_schema_list
    
    def get_all_schema_combinations(self,):
        """
        获得所有schema的组合，利用PostgreSQL自动生成合法的子查询

        Args:
            None
        Returns:
            return1:
            return2:
        """
        # 构建一整个query的meta信息
        schema_list, filter_list = self.schema_total, []
        # query_meta = schema_list, filter_list
        # # 生成对应的query文本
        # query_text = query_construction.construct_origin_query(query_meta=query_meta, workload=self.workload)
        # self.query_ctrl.set_query_instance(query_text=query_text, query_meta=query_meta)
        # alias_combination_list = self.query_ctrl.get_all_sub_relations()

        # 新的获得sub_relations的方法
        try:
            alias_list = [self.alias_mapping[s] for s in schema_list]
        except KeyError as e:
            print(f"get_all_schema_combinations: meet Error. alias_mapping = {self.alias_mapping}. schema_list = {schema_list}.")
            raise e
        alias_combination_list = self.data_manager.get_subqueries(alias_list=alias_list)
        # print("get_all_schema_combinations: alias_combination_list = {}.".format(alias_combination_list))

        schema_combination_list = self.restore_origin_names(alias_combination_list)
        
        self.schema_combination_list = schema_combination_list
        self.schema_combination_dict = defaultdict(list)

        # 两表以上的场景
        for sub_schema in self.schema_combination_list:
            # 换成元组试一下
            self.schema_combination_dict[len(sub_schema)].append(tuple(sub_schema))

        # 单表的组合添加
        for schema in self.schema_total:
            self.schema_combination_dict[1].append(tuple([schema,]))

        return self.schema_combination_dict

    def show_subset_info(self,):
        """
        展示每一个subset可选的组合数
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        for k, v in self.schema_combination_dict.items():
            print("table_num = {}. candidates num = {}.".format(k, len(v)))

    def random_schema_subset(self, schema_num:int, target_num:int) -> list:
        """
        {Description}

        Args:
            schema_num: 目标的schema数目
            target_num: 产生schema组合的数目
        Returns:
            result_list:
        """
        result_list = []
        schema_list = self.schema_combination_dict[schema_num]
        # print("schema_list = {}.".format(schema_list))
        if target_num >= len(schema_list):
            result_list = schema_list
        else:
            result_list = []
            index_list = np.random.choice(range(len(schema_list)), size=target_num, replace=False)
            for idx in index_list:
                result_list.append(schema_list[idx])
        return result_list




    def workload_generation_under_spec(self, alias_set_list, total_num, \
            min_card = 1000, timeout = 10000, gen_mode = "normal", return_mode = "list"):
        """
        在指定table
    
        Args:
            alias_set_list: 别名集合组成的列表
            total_num: 总的查询数目
            min_card: 最小基数的限制
            timeout: 查询超时的限制

        Returns:
            query_list, meta_list, label_list: list模式下
            query_dict, meta_dict, label_dict: dict模式下
        """
        assert return_mode in ("list", "dict")
        sample_num = int(np.ceil(total_num / len(alias_set_list)))

        if return_mode == "list":
            query_list, meta_list, label_list = [], [], []
        elif return_mode == "dict":
            query_dict, meta_dict, label_dict = {}, {}, {}

        for alias_set in alias_set_list:
            table_subset = [self.alias_inverse[a] for a in alias_set]
            if gen_mode == "normal":
                local_generator = generator.QueryGenerator(schema_list = table_subset, 
                    dm_ref=self.data_manager, bins_builder=self.bins_builder, workload=self.workload)
            elif gen_mode == "complex":
                local_generator = generator.ComplexQueryGenerator(schema_list = table_subset, 
                    dm_ref=self.data_manager, bins_builder=self.bins_builder, workload=self.workload)
            else:
                raise ValueError(f"gen_mode = {gen_mode}. valid value = ['complex', 'normal']")
            
            query_local, meta_local = local_generator.generate_batch_queries(num=sample_num, with_meta=True)
            # print("query_local = {}.".format(query_local))
            # print("meta_local = {}.".format(meta_local))

            if return_mode == "list":
                query_list.extend(query_local)
                meta_list.extend(meta_local)
            elif return_mode == "dict":
                query_dict[alias_set] = query_local
                meta_dict[alias_set] = meta_local

        if return_mode == "list":
            # print("query_list = {}.".format(query_list))
            # 获得所有的label
            ts = time.time()
            label_list = self.db_conn.get_cardinalities(sql_list=query_list, timeout=timeout)
            te = time.time()
            print("workload_generation: delta_time = {}.".format(te - ts))

            query_list, meta_list, label_list = \
                self.eliminate_timeout_results(query_list, meta_list, label_list)
            
            query_list, meta_list, label_list = self.eliminate_invalid_card_results(\
                query_list, meta_list, label_list, min_card=min_card)
            
            return query_list, meta_list, label_list
        else:
            for alias_set in alias_set_list:
                query_local, meta_local = query_dict[alias_set], meta_dict[alias_set]
                label_local = self.db_conn.get_cardinalities(sql_list=query_local, timeout=timeout)

                query_local, meta_local, label_local = \
                    self.eliminate_timeout_results(query_local, meta_local, label_local)
                
                query_local, meta_local, label_local = self.eliminate_invalid_card_results(\
                    query_local, meta_local, label_local, min_card=min_card)
                
                query_dict[alias_set], meta_dict[alias_set], \
                    label_dict[alias_set] = query_local, meta_local, label_local
                
            return query_dict, meta_dict, label_dict

    def eliminate_invalid_card_results(self, query_list, meta_list, label_list, min_card = None, max_card = None):
        """
        {Description}
    
        Args:
            query_list: 
            meta_list: 
            label_list: 
            min_card:
            max_card
        Returns:
            query_res:
            meta_res:
            label_res:
        """
        if min_card is None:
            min_card = -1

        if max_card is None:
            max_card = 1e15

        assert min_card < max_card
        query_res, meta_res, label_res = [], [], []
        for query, meta, label in zip(query_list, meta_list, label_list):
            if min_card <= label <= max_card:
                query_res.append(query)
                meta_res.append(meta)
                label_res.append(label)

        return query_res, meta_res, label_res
    

    def eliminate_timeout_results(self, query_list, meta_list, label_list):
        """
        消除因超时而无法获得基数值的查询
        
        Args:
            query_list:
            meta_list:
            label_list:
        Returns:
            query_res:
            meta_res:
            label_res:
        """
        query_res, meta_res, label_res = [], [], []
        card_threhsold = 10
        for query, meta, label in zip(query_list, meta_list, label_list):
            if label is not None and label >= card_threhsold:
                # 将label是None的结果排除掉，并且排除基数结果小的查询
                query_res.append(query)
                meta_res.append(meta)
                label_res.append(label)

        return query_res, meta_res, label_res


    def single_query_generation(self, schema_subset: list):
        """
        产生单个查询
    
        Args:
            arg1:
            arg2:
        Returns:
            query_text:
            query_meta:
        """
        local_generator = generator.QueryGenerator(schema_list = schema_subset, 
            dm_ref=self.data_manager, bins_builder=self.bins_builder, workload=self.workload)
        query_text, query_meta = local_generator.generate_query(with_meta=True)
        return query_text, query_meta
    
    
    def workload_generation(self, table_num_dist, total_num, timeout = 10, gen_mode = "random"):
        """
        {Description}
    
        Args: 
            table_num_dist: 表数目的分布，用一个字典进行表示
            total_num: 总的查询数目
            timeout: 查询超时大小
        Returns:
            query_list:
            meta_list:
            label_list:
        """
        query_list, meta_list, label_list = [], [], []
        for table_num, ratio in table_num_dist.items():
            local_num = int(total_num * ratio)  # 总的生成查询的数目
            # 设置查询模板数目
            template_num = int(np.ceil(np.sqrt(local_num)))
            print("当前的table数目: {}. 生成的template总数: {}".format(table_num, template_num))

            local_result = self.random_schema_subset(schema_num=table_num, target_num=template_num)
            sample_num = int(np.ceil(local_num / template_num))  # 每个模板下的采样查询数目

            # print("workload_generation: local_result = {}.".format(local_result))
            for schema_subset in local_result:
                if gen_mode == "random":
                    local_generator = generator.QueryGenerator(schema_list = schema_subset, 
                        dm_ref=self.data_manager, bins_builder=self.bins_builder, workload=self.workload)
                elif gen_mode == "complex":
                    local_generator = generator.ComplexQueryGenerator(schema_list = schema_subset, 
                        dm_ref=self.data_manager, bins_builder=self.bins_builder, workload=self.workload)

                query_local, meta_local = local_generator.generate_batch_queries(num=sample_num, with_meta=True)
                # print("query_local = {}.".format(query_local))
                # print("meta_local = {}.".format(meta_local))
                query_list.extend(query_local)
                meta_list.extend(meta_local)

        # 获得所有的label
        ts = time.time()
        label_list = self.db_conn.get_cardinalities(sql_list=query_list, timeout=timeout)
        te = time.time()
        print("workload_generation: delta_time = {}.".format(te - ts))

        eliminate_timeout_results = self.eliminate_timeout_results
        print("before eliminate length = {}.".format(len(query_list)))
        query_list, meta_list, label_list = \
            eliminate_timeout_results(query_list, meta_list, label_list)
        print("after eliminate length = {}.".format(len(query_list)))
        return query_list, meta_list, label_list

    def save_to_historical_result(self, query_list, meta_list, label_list):
        """
        保存历史结果
        作为一个对象进行保存，路径为/src/plan/intermediate/query_obj，对象根据保存的时间来进行命名
    
        Args:
            query_list:
            meta_list:
            label_list:
        Returns:
            flag:
        """
        res = query_list, meta_list, label_list
        obj_name = "wkld_{}.pkl".format(time.strftime("%Y%m%d%H%M%S", time.localtime()))
        obj_dir_path = "/home/lianyuan/Research/CE_Evaluator/src/plan/intermediate/query_obj"
        os.makedirs(obj_dir_path, exist_ok=True)

        out_path = p_join(obj_dir_path, obj_name)
        utils.dump_pickle(res_obj = res, data_path = out_path)

        return True

    def load_from_historical_result(self, timestamp = "latest"):
        """
        加载历史结果，默认加载最新保存的结果
    
        Args:
            timestamp:
            arg2:
        Returns:
            query_list:
            meta_list: 
            label_list: 
        """
        in_path = ""
        obj_dir_path = "/home/lianyuan/Research/CE_Evaluator/src/plan/intermediate/query_obj"

        if timestamp == "latest":
            # 获得时间最近的对象名
            candidate_list = []
            for name in os.listdir(obj_dir_path):
                if 'wkld_' in name:
                    candidate_list.append(name)
            candidate_list.sort()
            # 选择排名最后一个对象
            in_path = p_join(obj_dir_path, candidate_list[-1])
        else:
            in_path = p_join(obj_dir_path, "wkld_{}.pkl".format(timestamp))

        res = utils.load_pickle(in_path)
        query_list, meta_list, label_list = res
        return query_list, meta_list, label_list


    def error_analysis(self, query_list, meta_list, label_list):
        """
        结果错误分析
        
        Args:
            query_list:
            meta_list:
            label_list:
        Returns:
            prediction_list:
            template_candidates:
        """
        prediction_list = []


# %%
def get_initializer_by_workload(schema_list = [], workload = "job", split_budget = 100) -> SearchInitilization:
    """
    {Description}
    
    Args:
        schema_list:
        workload:
    Returns:
        init_instance:
        res2:
    """
    internal_handler = ce_injection.PGInternalHandler(workload=workload)
    data_manager = data_management.DataManager(wkld_name=workload)
    query_ctrl = query_exploration.QueryController(workload=workload)

    init_instance = SearchInitilization(schema_total = schema_list, \
        ce_handler = internal_handler, workload = workload, data_manager_ref = \
        data_manager, query_ctrl_ref = query_ctrl, split_budget = split_budget)
    
    return init_instance

# %%

class WorkloadManager(object):
    """
    负载的管理类

    Members:
        field1:
        field2:
    """

    def __init__(self, workload, ce_handler = "internal"):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        self.workload = workload
        self.load_count = 0         # 加载查询的次数
        self.query_global, self.meta_global, self.label_global, self.estimation_global = {}, {}, {}, {}
        self.alias_mapping = query_construction.abbr_option[workload]
        self.alias_inverse = {}
        for k, v in self.alias_mapping.items():
            self.alias_inverse[v] = k

        # if ce_handler is None:
        #     # 获得内置的处理器
        #     self.ce_handler = ce_injection.get_internal_handler(workload=workload)
        # else:
        #     self.ce_handler = ce_handler

        if isinstance(ce_handler, str):
            self.ce_handler = ce_injection.get_ce_handler_by_name(\
                workload=workload, ce_type=ce_handler)
        else:
            self.ce_handler = ce_handler

        self.stats_ref_list = []

    def set_ce_handler(self, external_handler):
        """
        设置外部的handler
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        if isinstance(external_handler, str):
            # 输入是关键字的case
            # 关键字全部转成小写
            external_handler = external_handler.lower()
            if external_handler == "internal":
                self.ce_handler = ce_injection.PGInternalHandler(workload=self.workload)
            elif external_handler == "deepdb":
                raise NotImplementedError("external_handler == \"deepdb\"未实现")
            elif external_handler == "neurocard":
                raise NotImplementedError("external_handler == \"neurocard\"未实现")
        else:
            # 输入是ce_handler实例的情况
            self.ce_handler = external_handler


    def load_queries(self, query_list, meta_list, label_list, description = ""):
        """
        加载采样查询的信息，由此提取出候选的模板

        Args:
            query_list:
            meta_list:
            label_list:
            estimation_list:
        Returns:
            return1:
            return2:
        """

        self.load_count += 1
        curr_key = self.load_count
        self.query_global[curr_key] = query_list
        self.meta_global[curr_key] = meta_list
        self.label_global[curr_key] = label_list
        # self.estimation_global[curr_key] = estimation_list

        # 完成estimation_list的构造
        estimation_list = self.ce_handler.get_cardinalities(query_list=query_list)

        # print(f"load_queries: ce_handler's type = {type(self.ce_handler)}")
        # print(f"load_queries: label_list = {label_list}.")
        # print(f"load_queries: estimation_list = {estimation_list}.")
        self.estimation_global[curr_key] = estimation_list


    def find_fact_record(self, in_key, num = 3):
        """
        {Description}
    
        Args:
            in_key:
            arg2:
        Returns:
            return1:
            return2:
        """
        res_list = []

        # test_item = self.stats_ref_list[0]
        # print(f"find_fact_record: in_key = {in_key}, type = {type(in_key)}")
        # print(f"find_fact_record: test_item = {test_item[0]}, {test_item[1]}")

        for item in self.stats_ref_list:
            if isinstance(in_key, str) and item[0] == in_key:
                res_list.append(item)
            elif isinstance(in_key, (list, tuple)) and \
                item[0] == in_key[0] and item[1] == in_key[1]:
                res_list.append(item)

        res_list.sort(key=lambda a:a[3], reverse=True)
        # return [item[2:] for item in res_list[:min(num, len(res_list))]]

        res_meta_list = [item[2] for item in res_list[:min(num, len(res_list))]]
        error_list = [item[3] for item in res_list[:min(num, len(res_list))]]

        meta_idx_dict = self.meta_idx_dict
        # for idx, meta in enumerate(self.meta_global):
        #     print(f"find_fact_record: idx = {idx}. meta = {meta}")
        #     meta_idx_dict[str(meta)] = idx

        res_idx_list = [meta_idx_dict[str(meta)] for meta in res_meta_list]

        # print(f"find_fact_record: res_idx_list = {res_idx_list}.")
        # 获得具体的基数信息
        query_list = utils.list_index(self.query_curr, res_idx_list)
        estimation_list = utils.list_index(self.estimation_curr, res_idx_list)
        label_list = utils.list_index(self.label_curr, res_idx_list)

        # for error, query, est_card, true_card in \
        #     zip(error_list, query_list, estimation_list, label_list):
        #     print("find_fact_record: error = {:2f}. true_card = {}. est_card = {}. query_text = {}.".\
        #           format(error, true_card, est_card, query))

        return list(zip(res_meta_list, error_list))


    def select_schema_subset(self,):
        """
        选择table的组合
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        pass
    

    def select_column_combination(self,):
        """
        选择column的组合
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        pass


    def select_template_number(self, batch_id = "latest", strategy = "score", total_num = 8, min_num = 2):
        """
        {Description}
    
        Args:
            total_num:
            min_num:
        Returns:
            over_est_num:
            under_est_num:
        """
        over_est_num, under_est_num = 0.0, 0.0

        error_distance_list = []
        over_indicator_list = self.construct_metrics_list(batch_id=batch_id, 
            strategy=strategy, mode="over", aggr_level = "table")
        under_indicator_list = self.construct_metrics_list(batch_id=batch_id, 
            strategy=strategy, mode="under", aggr_level = "table")

        print(over_indicator_list)
        print(under_indicator_list)

        for over_est_candidate in range(min_num, total_num - min_num):
            under_est_candidate = total_num - over_est_candidate
            error_distance = np.abs(over_indicator_list[over_est_candidate - 1][-1] - under_indicator_list[under_est_candidate - 1][-1])
            error_distance_list.append((error_distance, over_est_candidate, under_est_candidate))

        error_distance_list.sort(key=lambda a:a[0])
        error_min, over_est_num, under_est_num = error_distance_list[0]
        print(f"select_template_number: error_min = {error_min:.2f}. total_num = {total_num}. over_est_num = {over_est_num}. under_est_num = {under_est_num}.")
        return over_est_num, under_est_num


    def select_potential_templates(self, num, batch_id = "latest", strategy = "max-greedy", \
            mode = "under", schema_duplicate = 2, existing_templates = [], return_inter = False):
        """
        从查询样例中选择有潜力的模板，这里也存在着多个需要考虑的问题
        TODO: 考虑随机查询结果谓词的问题，为dynamic_mv生成提供指导
    
        Args:
            num: 选择模版的数目
            batch_id: 加载查询的批次
            strategy: 使用策略
            mode: 选择模版的模式，目前支持的模式有["over", "under"]，分别代表over-estimation、under-estimation的情况。
                  后续可能考虑结合真实cardinality的指标，增加其科学性。
            schema_duplicate: 为了增加结果的多样性，限制schema相同的结果的数目
            existing_templates: 已经存在的模版，避免重复创建的问题
        Returns:
            result_list: 返回结果列表，每一个结果的表示为(query_meta, select_columns)，可以作为create_template的入参
            filtered_list: 中间结果列表(optional)
        """
        def duplicate_aware_selection(result_list):
            """
            考虑重复元素的选择
            
            Args:
                result_list:
            Returns:
                selected_list:
            """
            selected_list, index_list = [], []
            schema_subset_cnt = defaultdict(int)
            for idx, item in enumerate(result_list):
                template_repr, indicator = item
                schema_key, column_key = template_repr
                schema_repr = schema_key
                if schema_subset_cnt[schema_repr] < schema_duplicate:
                    schema_subset_cnt[schema_repr] += 1
                    selected_list.append(item)
                    index_list.append(idx)

                if len(selected_list) >= num:
                    break

            return selected_list, index_list

        def template_meta_match(meta1, meta2):
            # print("template_meta_match: meta1 = {}".format(meta1))
            # print("template_meta_match: meta2 = {}".format(meta2))
            return set(meta1[0]) == set(meta2[0]) and \
                set(meta1[1]) == set(meta2[1])
        
        def existing_remove(named_indicator_list, existing_templates):
            filtered_list = []
            for item in named_indicator_list:
                flag = True
                for tmpl in existing_templates:
                    flag = flag and not(template_meta_match(item[0], tmpl))
                if flag == True:
                    filtered_list.append(item)
            return filtered_list
        
        # 构建aggr的结果
        named_indicator_list = self.construct_metrics_list(batch_id=batch_id, strategy=strategy, mode=mode)
        print(f"select_potential_templates: mode = {mode}. strategy = {strategy}. named_indicator_list is following.")
        for item in named_indicator_list:
            print(item)

        # 删去已有的结果
        filtered_list = existing_remove(named_indicator_list, existing_templates)

        # selected_list = named_indicator_list[:num]
        selected_list, index_list = duplicate_aware_selection(result_list=filtered_list)
        filtered_new = utils.list_index(filtered_list, index_list)  # 筛选对应的filtered内容

        # print("selected_list(before) = {}.".format(selected_list))
        selected_list = list(zip(*selected_list))[0]   # 把结果里的metrics删除
        # print("selected_list(after) = {}.".format(selected_list))
        result_list = [self.construct_template_elements(schema_list=schema_list, column_list=column_list) for \
                        schema_list, column_list in selected_list]
        
        if return_inter == True:
            # 可能需要考虑后续结果的处理
            return result_list, filtered_new
        else:
            return result_list
    
    def construct_sub_workload(self, batch_id):
        """
        构建子负载
        
        Args:
            arg1:
            arg2:
        Returns:
            query_list, meta_list, label_list, estimation_list
        """

        if batch_id == "latest":
            # 选择最新的batch_id
            batch_id = max(self.query_global.keys())
        elif batch_id == "all":
            # 代表选择所有的batch
            batch_id = list(self.query_global.keys())
            
        def list_key_on_dict(data_dict, key_list):
            res_list = []
            for key in key_list:
                res_list.extend(data_dict[key])
            return res_list

        if isinstance(batch_id, int):
            query_list, meta_list, label_list, estimation_list = \
                self.query_global[batch_id], self.meta_global[batch_id], \
                self.label_global[batch_id], self.estimation_global[batch_id]   
            return query_list, meta_list, label_list, estimation_list     
        elif isinstance(batch_id, list):
            query_list = list_key_on_dict(self.query_global, batch_id)
            meta_list = list_key_on_dict(self.meta_global, batch_id)
            label_list = list_key_on_dict(self.label_global, batch_id)
            estimation_list = list_key_on_dict(self.estimation_global, batch_id)
            return query_list, meta_list, label_list, estimation_list     
        else:
            # raise ValueError("construct_sub_workload: unsupported batch_id type: {}.".format(type(batch_id)))
            pass
        return [], [], [], []
    

    def workload_stats(self, mode = "under", aggr_level = "column"):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        def one_side_q_error(true_card, estimation_card, mode = "over"):
            if mode == "over":
                return estimation_card / true_card
            elif mode == "under":
                return true_card / estimation_card
            else:
                raise ValueError("one_side_q_error: mode = {}".format(mode))
            
        batch_id = list(self.query_global.keys())
        query_list, meta_list, label_list, estimation_list = \
            self.construct_sub_workload(batch_id=batch_id)

        # 一侧的Q-error
        error_list = [one_side_q_error(i, j, mode=mode) for i, j in zip(label_list, estimation_list)]
        metrics_list = list(zip(label_list, error_list))

        result_dict = self.workload_aggregation(meta_list=meta_list, metrics_list=metrics_list)

        if aggr_level == "column":
            indicator_list_dict = {}
            for k, v in result_dict.items():
                for kk, vv in v.items():
                    indicator_list_dict[(k, kk)] = vv
        elif aggr_level == "table":
            indicator_list_dict = defaultdict(list)
            for k, v in result_dict.items():
                for kk, vv in v.items():
                    indicator_list_dict[k].extend(vv)
        else:
            raise ValueError(f"construct_metrics_list: Unsupported aggr_level({aggr_level}).")
        
        named_indicator_list = []

        for k, v in indicator_list_dict.items():
            # print("select_potential_templates: k = {}. v = {}.".format(k, v))
            val_max, val_90th = float(np.max(v)), float(np.quantile(v, 0.9))
            val_75th, val_median = float(np.quantile(v, 0.75)), float(np.quantile(v, 0.5))
            named_indicator_list.append((k, val_max, val_90th, val_75th, val_median))

        named_indicator_list.sort(key=lambda a: a[1], reverse=True)

        for item in named_indicator_list:
            tmpl_repr, val_max, val_90th, val_75th, val_median = item

            print("template = {}. max = {:.2f}. 90th = {:.2f}. 75th = {:.2f}. median = {:.2f}".\
                  format(tmpl_repr, val_max, val_90th, val_75th, val_median))
    
        return named_indicator_list


    def construct_metrics_list(self, batch_id = "latest", strategy = "max-greedy", mode = "under", aggr_level = "column"):
        """
        构建附带指标的列表
    
        Args:
            batch_id:
            strategy:
            mode:
        Returns:
            return1:
            return2:
        """
        def one_side_q_error(true_card, estimation_card, mode = "over"):
            if mode == "over":
                return estimation_card / true_card
            elif mode == "under":
                return true_card / estimation_card
            else:
                raise ValueError("one_side_q_error: mode = {}".format(mode))
            
        query_list, meta_list, label_list, estimation_list = self.construct_sub_workload(batch_id=batch_id)
        
        self.meta_idx_dict = {}
        self.query_curr, self.meta_curr, self.label_curr, self.estimation_curr = \
            query_list, meta_list, label_list, estimation_list
        
        for idx, meta in enumerate(meta_list):
            # if idx < 5:
            #     print(f"find_fact_record: idx = {idx}. meta = {meta}")
            self.meta_idx_dict[str(meta)] = idx

        # 一侧的Q-error
        error_list = [one_side_q_error(i, j, mode=mode) for i, j in zip(label_list, estimation_list)]
        metrics_list = list(zip(label_list, error_list))

        result_dict = self.workload_aggregation(meta_list=meta_list, metrics_list=metrics_list)
        if aggr_level == "column":
            indicator_list_dict = {}
            for k, v in result_dict.items():
                for kk, vv in v.items():
                    indicator_list_dict[(k, kk)] = vv
        elif aggr_level == "table":
            indicator_list_dict = defaultdict(list)
            for k, v in result_dict.items():
                for kk, vv in v.items():
                    indicator_list_dict[k].extend(vv)
        else:
            raise ValueError(f"construct_metrics_list: Unsupported aggr_level({aggr_level}).")

        named_indicator_list = []
        if "greedy" in strategy:
            #
            indicator = strategy.split("-")[0]
            if indicator == "max":
                # 取最大值作为评价指标
                for k, v in indicator_list_dict.items():
                    # print("select_potential_templates: k = {}. v = {}.".format(k, v))
                    named_indicator_list.append((k, float(np.max(v))))
            elif indicator == "median":
                # 取中位数作为评价指标
                for k, v in indicator_list_dict.items():
                    named_indicator_list.append((k, float(np.median(v))))
            else:
                raise ValueError("select_potential_templates: invalid indicator = {}.".format(indicator))
        elif "score" in strategy:
            # 基于打分的模版选择
            def score_func(metrics_list: list):
                val_max = np.max(metrics_list)
                val_90th = np.quantile(metrics_list, 0.9)
                val_75th = np.quantile(metrics_list, 0.75)
                score = float(val_max * val_90th * val_75th)
                # print(f"construct_metrics_list.score_func: score = {score:.2f}. "
                #       f"val_max = {val_max:.2f}. val_90th = {val_90th:.2f}. val_75th = {val_75th:.2f}.")
                # 怀疑有overflow的问题，考虑做log
                return val_max
                # return np.log(val_max * val_90th * val_75th + 1.0)

            for k, v in indicator_list_dict.items():
                named_indicator_list.append((k, score_func(v)))
        else:
            raise ValueError("select_potential_templates: unsupported strategy({}).".format(strategy))

        # 根据指标进行排序
        named_indicator_list.sort(key=lambda a: a[1], reverse=True)
        return named_indicator_list


    def construct_template_elements(self, schema_list, column_list):
        """
        构建模版计划初始化所用到的元素
        
        Args:
            schema_list: 所有table名组成的列表
            column_list: 所有column名组成的列表
        Returns:
            query_meta:
            selected_columns:
        """
        query_meta = schema_list, []
        selected_columns = []

        for alias_name, column_name in column_list:
            schema_name = self.alias_inverse[alias_name]
            # 添加选择的列
            selected_columns.append((schema_name, column_name)) 
            # 添加column到query_meta
            query_meta[1].append((alias_name, column_name, "placeholder", "placeholder"))

        return query_meta, selected_columns


    def workload_aggregation(self, meta_list, metrics_list) -> dict:
        """
        查询结果的聚合，我们暂时考虑两个level的聚合，其一是
        
        Args:
            meta_list: 元信息列表
            metrics_list: 评价指标列表
        Returns:
            result_dict:
            res2:
        """
        # 清空stats_ref_list
        self.stats_ref_list = []

        result_dict = defaultdict(lambda: defaultdict(list))
        for meta, metrics in zip(meta_list, metrics_list):
            # 把tuple作为dict的key，方便之后使用
            # print("meta = {}. metrics = {}.".format(meta, metrics))
            schema_key = tuple(sorted(meta[0]))    # 对所有schema进行排序
            column_list = []
            for item in meta[1]:
                column_list.append((item[0], item[1]))
            column_key = tuple(sorted(column_list))
            # print("workload_aggregation: schema_key = {}. column_key = {}.".format(schema_key, column_key))

            label, error = metrics
            result_dict[schema_key][column_key].append(error)

            # self.stats_ref_dict[(schema_key, column_key)].append((meta, error))
            self.stats_ref_list.append((schema_key, column_key, meta, error))

        return result_dict

    def show_workload_state(self, target = "all", aggr_level = "column"):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        named_indicator_list = self.construct_metrics_list(batch_id=target, aggr_level=aggr_level)
        # named_indicator_list.sort(key = lambda a: a[1])
        # print("show_workload_state: named_indicator_list = {}.".format(named_indicator_list))
        print("WorkloadManager.show_workload_state: ")
        for item in named_indicator_list:
            print(item)
        return named_indicator_list
# %%

class IterationConditionBuilder(object):
    """
    迭代条件的构造者

    Members:
        field1:
        field2:
    """

    def __init__(self, workload: str, bins_builder: grid_preprocess.BinsBuilder, split_budget = 100):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        self.workload = workload
        self.bins_builder = bins_builder
        self.split_budget = split_budget

        self.alias_mapping = self.bins_builder.data_manager.tbl_abbr
        self.alias_inverse = self.bins_builder.data_manager.abbr_inverse

    def construct_options_along_single_record(self, fact_record, column_list, iter_times = 3):
        """
        只根据一条记录(query_meta, error)进行构造iteration_options，采用如下简单的思路：
        首先
    
        Args:
            fact_record: 包含query_meta和error
            column_list:
            start_ratio:
            iter_times: 确保所有column迭代次数相同
        Returns:
            cond_bound_dict:
            return2:
        """
        def find_closest_value_index(sorted_list, target):
            index = np.searchsorted(sorted_list, target)
            if index == 0:
                return 0
            if index == len(sorted_list):
                return len(sorted_list) - 1
            before = sorted_list[index - 1]
            after = sorted_list[index]
            if after - target < target - before:
                return index
            else:
                return index - 1

        def find_nearest_idx_pair(column, start_val, end_val):
            # 
            bins_list = self.bins_local[column]
            start_val -= 1

            start_idx = find_closest_value_index(bins_list, start_val)
            end_idx = find_closest_value_index(bins_list, end_val)
            assert start_idx <= end_idx, f"find_nearest_idx_pair: start_idx = {start_idx}. end_idx = {end_idx}."
            if start_idx == end_idx:
                start_idx -= 1
            # print(f"find_nearest_idx_pair: start_idx = {start_idx}. end_idx = {end_idx}.")

            return start_idx, end_idx
        
        def index_iteration_pairs(column, start_init, end_init, iter_times):
            # 迭代column信息
            bins_list = self.bins_local[column]
            min_idx, max_idx = 0, len(bins_list) - 1
            start_idx_list = list(np.linspace(min_idx, start_init, iter_times, dtype=int))[::-1]
            end_idx_list = list(np.linspace(end_init, max_idx, iter_times, dtype=int))

            index_pair_list = list(zip(start_idx_list, end_idx_list))
            # print(f"index_iteration_pairs: column = {column}. start_init = {start_init}. end_init = {end_init}. iter_times = {iter_times}.")
            # print(f"index_iteration_pairs: index_pair_list = {index_pair_list}.")
            return index_pair_list
        
        self.bins_local = self.bins_builder.construct_bins_dict(column_list)
        self.reverse_local = self.bins_builder.construct_reverse_dict(self.bins_local)
        self.marginal_local = self.bins_builder.construct_marginal_dict(self.bins_local)

        cond_bound_dict = defaultdict(list)
        index_pair_dict= {}

        query_meta, error = fact_record
        schema_list, filter_list = query_meta

        for alias_name, column_name, start_val, end_val in filter_list:
            table_name = self.alias_inverse[alias_name]
            if (table_name, column_name) in column_list:
                col = (table_name, column_name)
                start_init, end_init = find_nearest_idx_pair(col, start_val, end_val)
                index_pair_dict[col] = start_init, end_init

        for col in column_list:
            # curr_ratio = start_ratio
            start_init, end_init = index_pair_dict[col]
            bins_list = self.bins_local[col]
            index_pair_list = index_iteration_pairs(col, start_init, end_init, iter_times)

            for start_idx, end_idx in index_pair_list:
                start_val, end_val = utils.predicate_transform(bins_list, start_idx, end_idx)
                cond_bound_dict[col].append((start_val, end_val))

            # while True:
            #     curr_iter += 1
            #     if curr_iter > iter_times:
            #         break

            #     # start_val, end_val = self.search_best_bound(column = col,
            #     #     benefit_array = benefit_arr, cost_array = cost_arr, data_ratio = curr_ratio)
            #     start_val, end_val = utils.predicate_transform(start_curr, end_curr)
            #     cond_bound_dict[col].append((start_val, end_val))
            #     start_curr, end_curr = index_iteration(col, start_curr, end_curr)

            #     # curr_ratio *= grow_rate
            #     # print("curr_iter = {}. curr_ratio = {:.2f}.".format(curr_iter, curr_ratio))

            #     # 考虑最后一次循环的问题
            #     # if 1.0 <= curr_ratio < 2.0 - 1e-5:
            #     #     curr_ratio = 1.0

        return cond_bound_dict
    
    
    def construct_condition_iteration_options(self, fact_record_list, column_list, start_ratio = 1.0 / 16, grow_rate = 2):
        """
        构造condition_iteration的迭代优化
        
        Args:
            fact_record_list: 参考事实记录列表
            column_list: 选择的列list
            start_ratio:
            grow_rate:
        Returns:
            cond_bound_dict:
            res2:
        """
        # print(f"construct_condition_iteration_options: column_list = {column_list}")

        self.bins_local = self.bins_builder.construct_bins_dict(column_list)
        self.reverse_local = self.bins_builder.construct_reverse_dict(self.bins_local)
        self.marginal_local = self.bins_builder.construct_marginal_dict(self.bins_local)

        cond_bound_dict = defaultdict(list)

        benefit_arr_dict = {}
        cost_arr_dict = {}

        benefit_interval_dict, cost_interval_dict = {}, {}

        processed_list = self.preprocess_record_list(fact_record_list, self.reverse_local)

        # 构造基本的计算依据
        for col in column_list:
            cond_value_list = [(item[0][col], item[1]) for item in processed_list]

            cost_arr_dict[col] = self.marginal_local[col]
            benefit_arr_dict[col] = self.assign_value_vector(\
                cond_value_list, len(cost_arr_dict[col]))
            
        # 
        inf = 1e15
        for col in column_list:
            benefit_diff_mat = self.construct_diff_mat(benefit_arr_dict[col])
            cost_diff_mat = self.construct_diff_mat(cost_arr_dict[col])
            # mask不合法的结果

            benefit_diff_mat[benefit_diff_mat < 1e-5] = -inf
            cost_diff_mat[cost_diff_mat < 1e-5] = inf

            benefit_interval_dict[col] = benefit_diff_mat
            cost_interval_dict[col] = cost_diff_mat

        # 计算最理想的边界
        max_iter_times = 10
        for col in column_list:
            benefit_arr, cost_arr = benefit_interval_dict[col], cost_interval_dict[col]

            curr_ratio = start_ratio
            curr_iter = 0
            while True:
                curr_iter += 1
                if curr_iter > max_iter_times or curr_ratio > 1.0:
                    break

                start_val, end_val = self.search_best_bound(column = col,
                    benefit_array = benefit_arr, cost_array = cost_arr, data_ratio = curr_ratio)
                cond_bound_dict[col].append((start_val, end_val))

                curr_ratio *= grow_rate
                # print("curr_iter = {}. curr_ratio = {:.2f}.".format(curr_iter, curr_ratio))

                # 考虑最后一次循环的问题
                if 1.0 <= curr_ratio < 2.0 - 1e-5:
                    curr_ratio = 1.0

        return cond_bound_dict
    
    def construct_diff_mat(self, value_arr):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        value_cumsum = np.cumsum(value_arr)
        value_cumsum = np.concatenate(([0], value_cumsum))

        # arr_len = len(value_cumsum)
        # diff_mat = value_cumsum.reshape(1, arr_len) @ \
        #            value_cumsum.reshape(arr_len, 1)

        diff_mat = np.subtract.outer(value_cumsum, value_cumsum)
        return diff_mat


    def preprocess_record_list(self, fact_record_list, reverse_dict):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        result_list = []

        for query_meta, error in fact_record_list:
            # fact_record中的每一条记录都是元信息和误差
            # print(f"preprocess_record_list: query_meta = {query_meta}.")
            _, fileter_list = query_meta
            column_idx_dict = {}

            for item in fileter_list:
                alias, column, start_val, end_val = item
                schema = self.alias_inverse[alias]

                if (schema, column) not in reverse_dict:
                    continue
                start_idx, end_idx = utils.predicate_location(\
                    reverse_dict[(schema, column)], start_val, end_val, schema, column)
                schema = self.alias_inverse[alias]
                column_idx_dict[(schema, column)] = start_idx, end_idx
            
            result_list.append((column_idx_dict, error))

        return result_list
     
    def assign_value_vector(self, cond_value_list, total_len):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        # 增加一些基础的收益，后续考虑优化
        res_arr = np.ones((total_len,), dtype=np.float64) 

        for (start_idx, end_idx), val in cond_value_list:
            res_arr[start_idx: end_idx + 1] += val

        return res_arr

    def search_best_bound(self, column, benefit_array: np.ndarray, cost_array: np.ndarray, data_ratio):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            start_val:
            end_val:
        """
        cost_threshold = data_ratio * np.sum(self.marginal_local[column])
        # print("search_best_bound: column = {}. array_shape = {}. cost_threshold = {:.2f}. cost_min = {:.2f}. cost_max = {:.2f}".\
        #       format(column, cost_array.shape, cost_threshold, np.min(cost_array), np.max(cost_array)))
        # print("search_best_bound: column = {}. array_shape = {}. cost_threshold = {:.2f}. cost_min = {:.2f}.".\
        #       format(column, cost_array.shape, cost_threshold, np.min(cost_array)))
         
        selected_idx = cost_threshold >= cost_array     # 选择尽可能多的结果

        # print(f"search_best_bound: selected_idx = {selected_idx}")
        if np.sum(selected_idx) > 0.5:  
            # 至少有一个结果
            max_benefit_idx = np.argmax(benefit_array[selected_idx])
            origin_idx = list(zip(*np.where(selected_idx == True)))[max_benefit_idx]
        else:
            # 直接选cost最小的结果
            origin_idx = np.argmin(cost_array)
            origin_idx = np.unravel_index(origin_idx, cost_array.shape)

        # print("search_best_bound: selected_cost = {:.2f}. selected_benefit = {:.2f}.".\
        #     format(cost_array[origin_idx], benefit_array[origin_idx]))

        
        end_idx, start_idx = origin_idx

        curr_bins_list = self.bins_local[column]
        start_val, end_val = curr_bins_list[start_idx] + 1, curr_bins_list[end_idx]
        # print("search_best_bound: start_idx = {}. end_idx = {}. start_val = {}. end_val = {}.".\
        #       format(start_idx, end_idx, start_val, end_val))
        # start_val, end_val = utils.predicate_transform(curr_bins_list, start_idx, end_idx)
        return start_val, end_val
    
# %%

