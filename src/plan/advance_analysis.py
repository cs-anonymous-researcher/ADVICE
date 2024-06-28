#!/usr/bin/env python
# -*- coding: utf-8 -*-

# %%

import time
from query import query_construction
from plan import plan_analysis, node_query
from estimation import plan_estimation, estimation_interface
from utility import utils
from utility.workload_parser import SQLParser
from copy import deepcopy
from plan.node_extension import ExtensionInstance
import numpy as np
from plan import node_query, node_extension
from collections import defaultdict
from workload import physical_plan_info

from utility.common_config import benefit_config

# %%

class PredicateController(object):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, query_instance_ref: node_query.QueryInstance, \
            extend_table: str, bins_dict: dict, marginal_dict: dict, column_list, mode):
        """
        目前只针对单个条件，后续考虑扩充

        Args:
            query_instance_ref:
            extend_table: 新拓展的表
            bins_local:
            marginal_local:
            column_list:
            mode:
        """
        self.workload = query_instance_ref.workload
        self.table_name, self.column_list = extend_table, column_list
        self.alias_name = query_construction.abbr_option[self.workload][extend_table]

        assert mode in ("under-estimation", "over-estimation")
        self.mode = mode
        self.query_instance = query_instance_ref
        self.bins_dict, self.marginal_dict = bins_dict, marginal_dict
        self.pred_candidates = defaultdict(lambda: dict())   # 候选的结果
        self.pred_existing = defaultdict(lambda: dict())     # 已经存在的结果

        self.column_size_dict = {}                           # 代表每个column总的记录数

    # @utils.timing_decorator
    def pred_generation(self, num):
        """
        生成谓词条件

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        # bins_len = len(self.bins_local)
        # 当前策略: 在每一个列上分别进行处理
        for col in self.column_list:
            bins_local, marginal_local = self.bins_dict[col], self.marginal_dict[col]
            self.column_size_dict[col] = np.sum(marginal_local)
            bins_len = len(bins_local)

            for pred_idx in range(num):
                # 确保随机产生的两个值不等
                idx1, idx2 = np.random.choice(bins_len, size=2, replace=False)  
                start_idx, end_idx = min(idx1, idx2), max(idx1, idx2)
                start_val, end_val = utils.predicate_transform(\
                    bins_local, start_idx, end_idx)
                range_val = utils.get_marginal_range(\
                    marginal_local, start_idx, end_idx)
                
                self.pred_candidates[col][pred_idx] = {
                    "column": col,
                    "idx_pair": (start_idx, end_idx),
                    "val_pair": (start_val, end_val),
                    "range_size": range_val
                }
        # check column_size_dict
        # print(f"pred_generation: column_size_dict = {self.column_size_dict}.")
        return self.pred_candidates

    # @utils.timing_decorator
    def card_estimation_adjust(self, subquery_dict: dict, single_table_dict: dict, ratio: float):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        subquery_res, single_table_res = {}, {}

        for k, v in subquery_dict.items():
            # print(f"card_estimation_adjust: v = {v}. ratio = {ratio}.")
            if self.alias_name in k:
                subquery_res[k] = int(v * ratio)
            else:
                subquery_res[k] = v


        for k, v in single_table_dict.items():
            if k == self.alias_name:
                single_table_res[k] = int(v * ratio)
            else:
                single_table_res[k] = v

        return subquery_res, single_table_res


    def pred_selection_by_reference(self, ref_card_list, out_num):
        """
        通过reference选择相应的predicates，重点是row estimation尽可能的接近
        
        Args:
            ref_card_list: 引用实例列表
            arg2:
        Returns:
            res1:
            res2:
        """
        def dist_score(in_card, ref_card_list, total_card, alpha = 1.0):
            # 计算in_card和ref_card_list间的距离
            # print(f"pred_selection_by_reference.dist_score: ref_card_list = {ref_card_list}. total_card = {total_card}.")
            normalize_card_list = [card / total_card for card in ref_card_list]
            in_card = in_card / total_card
            dist_arr = np.abs(np.array(normalize_card_list) - in_card)
            score_arr = np.exp(-1 * alpha * dist_arr)
            return np.sum(score_arr)

        idx_list, result_list = [], []

        instance = self.query_instance
        curr_meta = instance.query_meta
        # print(f"pred_selection_by_reference: ref_card_list = {ref_card_list}.")

        # 遍历所有的predicate，然后找距离最近的
        total_list = []
        for col in self.column_list:
            item_list = list(self.pred_candidates[col].items())
            # print(f"pred_selection_by_reference: col = {col}. item[0] = {item_list[0]}.")
            # 计算score
            total_card = self.column_size_dict[col]
            score_list = [dist_score(item[1]['range_size'], ref_card_list, total_card, alpha=5.0) for item in item_list]
            # print(f"pred_selection_by_reference: col = {col}. score_list = {utils.list_round(score_list, 2)}.")
            total_list.extend(list(zip(item_list, score_list)))

        # print(f"pred_selection_by_reference: total_list = {total_list}.")
        total_list.sort(key=lambda a: a[1], reverse=True)

        for (k, v), score in total_list[:out_num]:
            col = v['column']
            print(f"pred_selection_by_reference: score = {score:.2f}. col = {col}. v = {v}.")
            range_dict = {
                (self.table_name, col): v['val_pair']
            }
            meta_local = instance.add_new_table_meta(\
                curr_meta, self.table_name, range_dict)
            result_list.append(meta_local)
            idx_list.append((col, k))

        return idx_list, result_list

    # @utils.timing_decorator
    def pred_random_selection(self, ):
        """
        在一开始随机选择谓词
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        idx_list, result_list = [], []
        instance = self.query_instance
        curr_meta = instance.query_meta

        for col in self.column_list:
            item_list = list(self.pred_candidates[col].items())
            selected_idx = np.random.choice(len(item_list))

            k, v = item_list[selected_idx]
            range_dict = {
                (self.table_name, col): v['val_pair']
            }
            meta_local = instance.add_new_table_meta(\
                curr_meta, self.table_name, range_dict)

            result_list.append(meta_local)
            idx_list.append((col, k))
        return idx_list, result_list
    
    # @utils.timing_decorator
    def pred_intelligent_selection(self, ):
        """
        谓词探索的智能选择
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        idx_list, result_list = [], []
        instance = self.query_instance
        curr_meta = instance.query_meta
        for col in self.column_list:
            local_candidates = self.pred_candidates[col]
            local_existence = self.pred_existing[col]

            valid_flag = False
            for pred_idx, v in local_candidates.items():
                if pred_idx in local_existence:
                    continue

                start_idx, end_idx = v['idx_pair']
                start_val, end_val = v['val_pair']

                subquery_local, single_table_local = \
                    self.pred_inference(start_idx, end_idx, col)
                
                # 当前的元信息
                # meta_local = (), ()
                range_dict = {(self.table_name, col): (start_val, end_val)}
                meta_local = instance.add_new_table_meta(curr_meta, self.table_name, range_dict)

                flag, plan_target, plan_actual = self.query_instance.plan_verification(\
                    meta_local, self.table_name, subquery_local, single_table_local)

                if self.mode == "under-estimation":
                    if flag == True:
                        valid_flag = True
                        result_list.append(meta_local)
                        idx_list.append((col, pred_idx))
                elif self.mode == "over-estimation":
                    if flag == False:
                        valid_flag = True
                        result_list.append(meta_local)
                        idx_list.append((col, pred_idx))

                if valid_flag == True:
                    break

        return idx_list, result_list

    # @utils.timing_decorator
    def distance_measure(self, start_idx1, end_idx1, start_idx2, end_idx2, marginal_list):
        """
        sample之间的距离估计，这里我们考虑两方面的距离，一是coverage，即重叠的幅度。二是size，即大小。
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        size1 = utils.get_marginal_range(marginal_list, start_idx1, end_idx1)
        size2 = utils.get_marginal_range(marginal_list, start_idx2, end_idx2)

        size_dist = 1.0 - (min(size1, size2) / max(size1, size2))

        if end_idx1 <= start_idx2 or end_idx2 <= start_idx1:
            coverage_dist = 1.0
        else:
            start_inter, end_inter = max(start_idx1, start_idx2), min(end_idx1, end_idx2)
            size_inter = utils.get_marginal_range(marginal_list, start_inter, end_inter)
            coverage_dist = 1.0 - (size_inter / np.sqrt(size1 * size2))

        return size_dist + coverage_dist

    # @utils.timing_decorator
    def pred_inference(self, start_idx, end_idx, column_name):
        """
        利用已有数据推断，找到最接近的
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        bins_local, marginal_local = self.bins_dict[column_name], \
            self.marginal_dict[column_name]

        min_idx, min_dist, min_idx_pair = None, 1000, (-1, -1)

        for pred_idx, v in self.pred_existing[column_name].items():
            start_ref, end_ref = v['idx_pair']
            curr_dist = self.distance_measure(start_idx, \
                end_idx, start_ref, end_ref, marginal_local)

            # print(f"pred_inference: pred_idx = {pred_idx}. curr_dist = {curr_dist}. min_dist = {min_dist}.")
            if curr_dist < min_dist:
                min_idx, min_dist, min_idx_pair = pred_idx, curr_dist, (start_ref, end_ref)

        
        target_range = utils.get_marginal_range(marginal_local, start_idx, end_idx)
        start_ref, end_ref = min_idx_pair
        reference_range = utils.get_marginal_range(marginal_local, start_ref, end_ref)
        ratio = target_range / reference_range

        # print(f"pred_inference: final. pred_idx = {pred_idx}. curr_dist = {curr_dist}. min_dist = {min_dist}.")
        subquery_dict, single_table_dict = self.pred_existing[column_name][min_idx]['card_dict_pair']

        subquery_res, single_table_res = self.card_estimation_adjust(subquery_dict, single_table_dict, ratio)
        return subquery_res, single_table_res
        

    # @utils.timing_decorator
    def pred_load(self, pred_idx, column_name, subquery_dict, single_table_dict):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        candidate_info = self.pred_candidates[column_name][pred_idx]
        self.pred_existing[column_name][pred_idx] = {
            "card_dict_pair": (subquery_dict, single_table_dict), 
            "idx_pair": candidate_info['idx_pair'], 
            "val_pair": candidate_info['val_pair']
        }

        return self.pred_existing

    

# %%

class AdvanceAnalyzer(plan_analysis.InstanceAnalyzer):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, query_instance: node_query.QueryInstance, mode: str, \
        save_intermediate = False, split_budget = 100):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        super(AdvanceAnalyzer, self).__init__(query_instance=query_instance, split_budget=split_budget)

        # 批量的估计查询执行器
        self.batch_executor = node_query.BatchExecutor(query_instance.ce_handler)

        assert mode in ("over-estimation", "under-estimation")
        self.mode = mode

        self.save_intermediate = save_intermediate
        self.record_list = []

    def get_analyze_records(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        return self.record_list

    @utils.timing_decorator
    def init_all_actions(self, table_subset, mode = "random"):
        """
        初始化所有的动作，并且计算获得预期的收益
        
        Args:
            table_subset: 所有数据表的子集
            mode: 初始化应用模式
        Returns:
            action_list: 动作列表 
            value_list: 对应的收益列表
        """
        assert mode in ("random", "multi-loop", "multi_loop")

        instance = self.instance
        action_list, value_list = [], []
        table_list = instance.fetch_candidate_tables(table_subset=table_subset)    # 获取所有当前可以join的表

        # 结果保存字典
        result_dict = {}
        t1 = time.time()
        for table in table_list:
            # 同一table的情况下评测多个meta信息，随机生成5个metas
            if mode == "random":
                # 使用新的批处理系统
                meta_list, query_list, result_batch, card_dict_list = self.multi_plans_evaluation_under_meta_list(\
                    new_table=table, meta_num=10, split_budget=self.split_budget)
            elif mode == "multi-loop" or mode == "multi_loop":
                try:
                    temp = self.multi_plans_evaluation_under_multi_loop(new_table=table, 
                        meta_num=3, mode=self.mode, split_budget=self.split_budget)
                    meta_list, query_list, result_batch, card_dict_list = temp
                except ValueError as e:
                    print(f"init_all_actions: item[0] = {temp[0]}.")
                    print(f"init_all_actions: item[1] = {temp[1]}.")
                    print(f"init_all_actions: item[2] = {temp[2]}.")
                    raise e
            
            # if self.save_intermediate == True:
            #     # 保存历史结果
            #     for meta, query, card_dict in zip(meta_list, query_list, card_dict_list):
            #         self.record_list.append((meta, query, card_dict))

            # if len(result_batch) > 0:
            #     print(f"AdvanceAnalyzer.init_all_actions: table = {table}. result_batch = {result_batch}. result_item = {result_batch[0]}.")

            benefit, candidate_list, card_dict_candidate = self.result_filter(\
                query_list, meta_list, result_batch, card_dict_list, target_table=table)

            # 添加benefit和action，如果结果不优直接暂时把action禁用了
            if benefit > 1e-5:
                result_dict[table] = candidate_list, card_dict_candidate
                value_list.append(benefit)
                action_list.append(table)

        t2 = time.time()
        # raise ValueError("init_all_actions test")
        self.action_result_dict = result_dict
        print(f"AdvanceAnalyzer.init_all_actions: action_list = {action_list}. value_list = {utils.list_round(value_list)}")
        print(f"AdvanceAnalyzer.init_all_actions: delta_time = {t2 - t1:.2f}. len(action_list) = {len(action_list)}. len(table_list) = {len(table_list)}. ")

        return action_list, value_list
    


    def exploit(self, table_name, config = {}):
        """
        选取若干个表进行拓展，最后选一个收益最大的meta信息创建新的query_instance
        
        相较原函数做了如下的优化：
        1. 存储了之前的estimation cardinality，减小探索代价
        2. 

        Args:
            table_name: 表的名字
            config: 探索的配置
        Returns:
            new_query_instance:
            benefit:
            actual_cost: 
            expected_cost: 
            actual_card: 
            expected_card:
        """
        # instance = self.instance
        candidate_list, card_dict_list = self.action_result_dict[table_name]
        new_query_instance = None

        extension_list = []
        clip_factor = config.get('clip_factor', 10.0)     # 截断变量
        alpha = config.get('alpha', 0.5)
        # 
        time_limit = config.get('timeout', -1)

        def benefit_calculate(cost_true, cost_estimation, card_true, card_estimation):
            """
            计算节点拓展的收益情况
            
            Args:
                arg1:
                arg2:
            Returns:
                res1:
                res2:
            """
            benefit1, benefit2 = 0.0, 0.0   # [0, 1]之间的值
            # 计算plan的增益
            if cost_true * clip_factor <= cost_estimation:
                benefit1 = 1.0
            else:
                benefit1 = (cost_estimation - cost_true) / (cost_true * (clip_factor - 1))

            # 计算cardinality带来的增益
            if config['mode'] == "over-estimation":
                if card_estimation <= card_true:
                    benefit2 = 0.0
                elif card_true * clip_factor <= card_estimation:
                    benefit2 = 1.0
                else:
                    benefit2 = (card_estimation - card_true) / (cost_true * (clip_factor - 1))

            elif config['mode'] == "under-estimation":
                try:
                    if card_estimation >= card_true:
                        benefit2 = 0.0
                    else:
                        benefit2 = (card_true - card_estimation) / card_true
                except TypeError as e:
                    print(f"benefit_calculate: TypeError. card_est = {card_estimation}. card_true = {card_true}.")
                    raise e

            return alpha * benefit1 + (1 - alpha) * benefit2

        for (query_text, query_meta), card_dict in zip(candidate_list[:1], card_dict_list[:1]):
            # 处理配一个候选的配置，记录最优的结果
            print("InstanceAnalyzer.exploit: query_meta = {}. query_text = {}.".format(query_meta, query_text))
            local_extension = self.create_extension_instance(\
                ext_query_meta=query_meta, ext_query_text=query_text, card_dict=card_dict)
            
            flag, cost1, cost2 = local_extension.true_card_plan_verification(\
                time_limit=time_limit, with_card_dict=True)     # P-Error相关素材，cost1 <= cost2，使用hint_sql优化plan

            # 针对flag的处理，考虑超时的情况
            if flag == False:
                # 两个查询计划不等的情况
                top_query_estimation, top_query_true = local_extension.top_query_estimation() 
                # Q-Error相关素材
                benefit = benefit_calculate(cost1, cost2, top_query_true, top_query_estimation)
            elif flag == True:
                # 两个查询计划相等的情况
                if cost1 == -1 and cost2 == -1:
                    top_query_estimation, top_query_true = 1, 1     # 伪造一个基数
                    benefit = 0.0
                else:
                    top_query_estimation, top_query_true = local_extension.top_query_estimation()               # Q-Error相关素材
                    benefit = benefit_calculate(cost1, cost2, top_query_true, top_query_estimation)

            # 在原有的基础上添加额外的内容
            extension_list.append((local_extension, benefit, cost1, cost2, top_query_true, top_query_estimation))
        
        # print("exploit: len(extension_list) = {}. len(candidate_list) = {}. len(card_dict_list) = {}.".\
        #       format(len(extension_list), len(candidate_list), len(card_dict_list)))
        extension_list.sort(key=lambda a:a[1], reverse=True)
        selected_extension: node_extension.ExtensionInstance = extension_list[0][0]
        benefit = extension_list[0][1]      # 收益情况
        actual_cost, expected_cost = extension_list[0][3], extension_list[0][2]
        actual_card, expected_card = extension_list[0][5], extension_list[0][4]

        # 修改入参
        new_query_instance = selected_extension.construct_query_instance(query_instance_ref=self.instance)  
        return new_query_instance, benefit, actual_cost, expected_cost, actual_card, expected_card


    @utils.timing_decorator
    def multi_plans_evaluation_under_multi_loop(self, new_table, column_num = 1, meta_num = 3, \
            total_num = 20, loop_num = 3, split_budget = 100, target_num = 3, mode = "under-estimation"):
        """
        采用多轮的策略探索condition
        
        Args:
            new_table: 
            column_num: 
            meta_num:
            total_num:
            loop_num:
            split_budget:
            target_num:
            mode:
        Returns:
            meta_global:
            query_global: 
            result_global:
            card_dict_global:
        """

        assert mode in ("under-estimation", "over-estimation")
        bins_builder = self.instance.bins_builder

        total_column_list = self.instance.data_manager.\
            get_valid_columns(schema_name=new_table)
        total_column_num = len(total_column_list)     # 总的列个数
        
        selected_columns = [(new_table, column) for column in total_column_list]

        bins_origin = bins_builder.construct_bins_dict(selected_columns, split_budget)
        marginal_origin = bins_builder.construct_marginal_dict(bins_origin)

        bins_local = {k[1]: v for k, v in bins_origin.items()}
        marginal_local = {k[1]: v for k, v in marginal_origin.items()}

        if meta_num <= total_column_num:
            column_list = np.random.choice(total_column_list, meta_num)
        else:
            column_list = total_column_list

        pred_ctrl = PredicateController(self.instance, new_table, \
                    bins_local, marginal_local, column_list, mode)
        pred_ctrl.pred_generation(num=total_num)

        query_global, meta_global, result_global, card_dict_global = [], [], [], []

        for iter_idx in range(loop_num):
            # 生成candidate_meta
            if iter_idx == 0:
                idx_list, meta_list = pred_ctrl.pred_random_selection()
            else:
                idx_list, meta_list = pred_ctrl.pred_intelligent_selection()

            # print(f"multi_plans_evaluation_under_multi_loop: idx_list = {idx_list}.")
            # print(f"multi_plans_evaluation_under_multi_loop: meta_list = {meta_list}.")

            if len(meta_list) == 0:
                #
                return [], [], [], []
            
            # evaluate结果
            result_list, card_dict_list = self.plan_list_evaluation(\
                meta_list, new_table, with_card_dict = True)

            result_wrapped = [(item, None) for item in result_list]

            # print(f"multi_plans_evaluation_under_multi_loop: result_list = {result_list}.")

            for idx, (res_item, card_dict) in enumerate(zip(result_list, card_dict_list)):
                assert res_item[0] in (True, False)
                # if (mode == "over-estimation" and res_item[0] == False) or\
                #    (mode == "under-estimation" and res_item[0] == True): 
                # 20240309: 对于over-estimation的结果，直接加入候选集
                if (mode == "over-estimation") or (mode == "under-estimation" and res_item[0] == True):  
                    # 添加结果
                    query_meta = meta_list[idx]
                    query_text = query_construction.construct_origin_query(\
                        query_meta, self.instance.workload)
                    query_global.append(query_text), meta_global.append(query_meta)
                    result_global.append(result_wrapped[idx])
                    card_dict_global.append(card_dict)

            # 如果遇到满足条件的结果，直接返回
            if len(query_global) >= target_num:
                break

            # 添加结果
            for (col_name, pred_idx), result in zip(idx_list, card_dict_list):
                subquery_local, single_table_local = result['subquery'], result['single_table']
                pred_ctrl.pred_load(pred_idx, col_name, subquery_local, single_table_local)

        # return , meta_global, result_global
        params = meta_global, query_global, result_global, card_dict_global
        # print(f"multi_plans_evaluation_under_multi_loop: len(params) = {len(params)}")
        return params

    @utils.timing_decorator
    def multi_plans_evaluation_under_meta_list(self, new_table, column_num = 1, \
        meta_num = 3, split_budget = 100):
        """
        从属于同一个table下的多个meta信息
        
        Args:
            new_table:
            column_num:
            meta_num:
            adjust_num:
            split_budget:
        Returns:
            meta_list:
            query_list:
            result_batch:
        """
        time_start = time.time()
        instance = self.instance

        # 生成column的相关值
        column_list = instance.predicate_selection(table_name = new_table, \
            column_num = column_num)
        
        bins_dict, reverse_dict = instance.construct_bins_dict(column_list = column_list, \
            split_budget = split_budget)

        # 一批随机的column值
        column_values_batch = instance.random_on_columns_batch(bins_dict = bins_dict, num = meta_num)
        curr_query_meta = instance.query_meta

        # 生成一批meta信息
        meta_list = [instance.add_new_table_meta(curr_query_meta, new_table, column_range_dict = \
            column_values) for column_values in column_values_batch]
        
        query_list = [query_construction.construct_origin_query(meta_info, workload = instance.workload) \
            for meta_info in meta_list]

        # print("meta_list = {}.".format(meta_list))
        result_batch, card_dict_list = self.plan_list_evaluation(\
            meta_list, new_table, with_card_dict=True)

        # 加工结果，使其和之前的格式相匹配
        result_wrapped = [(item, None) for item in result_batch]
        time_end = time.time()

        print(f"multi_plans_evaluation_under_meta_list: new_table = {new_table}. column_num = {column_num}. delta_time = {time_end - time_start:.2f}")

        # return zip(meta_list, result_batch)
        # return meta_list, query_list, result_batch
        return meta_list, query_list, result_wrapped, card_dict_list
    

    @utils.timing_decorator
    def plan_list_evaluation(self, query_meta_list, new_table, with_card_dict = False):
        """
        {Description}
    
        Args:
            query_meta_list:
            new_table:
        Returns:
            result_list:
            card_dict_list(optional):
        """
        instance = self.instance
        self.batch_executor.clean_cache()

        idx_list = []
        for query_meta in query_meta_list:
            subquery_dict, single_table_dict = instance.get_query_card_dict(query_meta)
            idx = self.batch_executor.add_instance(subquery_dict, single_table_dict)
            idx_list.append(idx)

        instance_result = self.batch_executor.execute_all_instance_list()
        result_list = []
        card_dict_list = []
        for idx, query_meta in zip(idx_list, query_meta_list):
            subquery_ext, single_table_ext = instance_result[idx]

            subquery_full, single_table_full = \
                instance.load_query_card_dict(query_meta, subquery_ext, single_table_ext)
            
            flag, plan_target, plan_actual = instance.plan_verification(query_meta=query_meta, \
                new_table=new_table, subquery_dict=subquery_full, single_table_dict=single_table_full)
            origin_result = flag, plan_target, plan_actual

            # 添加相关结果
            result_list.append(origin_result)
            card_dict_list.append({
                "subquery": subquery_full,
                "single_table": single_table_full
            })

        if with_card_dict == True:
            return result_list, card_dict_list
        else:
            return result_list
    
    def expect_benefit_estimation(self, ):
        """
        预期收益估计，判断
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        pass


    def exploit_async(self, table_name, config = {}):
        """
        生成异步探索的节点
    
        Args:
            arg1:
            arg2:
        Returns:
            estimate_benefit: 
            local_extension: 
            local_query_instance:
            est_expected_cost:
            est_actual_cost:
        """
        
        candidate_list, card_dict_list = self.action_result_dict[table_name]

        try:
            # 直接选取第一个节点进行探索
            (query_text, query_meta), card_dict = candidate_list[0], card_dict_list[0]
        except Exception as e:
            print(f"exploit_async: meet IndexError. candidate_list = {candidate_list}. "\
                  f"card_dict_list = {card_dict_list}. action_result_dict = {self.action_result_dict}.")
            raise e

        # 生成extension实例
        local_extension = self.create_extension_instance(ext_query_meta=query_meta, \
                            ext_query_text=query_text, card_dict=card_dict)
        local_query_instance = local_extension.construct_query_instance(\
            query_instance_ref=self.instance)  
        
        # 2024-03-30: 先估计一下预期收益，此时不考虑join order的问题
        estimate_cost_pair = estimation_interface.estimate_plan_benefit(local_extension, self.mode, 
            table_name, card_est_spec = "graph_corr_based", plan_sample_num=benefit_config['plan_sample_num'],
            restrict_order = False)
        
        estimate_benefit = utils.benefit_calculate(estimate_cost_pair[0], \
                            estimate_cost_pair[1], clip_factor=10.0)

        # 再返回实例，用于任务启动
        return estimate_benefit, local_extension, local_query_instance, estimate_cost_pair[0], estimate_cost_pair[1]


    def create_extension_instance(self, ext_query_meta, ext_query_text, card_dict):
        """
        创建extension的实例
    
        Args:
            ext_query_meta: 外部的query_meta
            ext_query_text: 外部的query_text，包含了一部分foreign_mapping的信息，所以需要导入进去
        Returns:
            extension_res:
            return2:
        """
        instance = self.instance
        # query_meta的合法性检测
        extra_meta = instance.get_extra_schema_meta(ext_query_meta)
        # print(f"create_extension_instance: extra_meta = {extra_meta}. ext_query_meta = {ext_query_meta}")
        if len(extra_meta[0]) != 1:
            # 额外的table数目不为1
            raise ValueError("create_extension_instance: ext_query_meta = {}. self.query_meta = {}.".\
                    format(ext_query_meta, instance.query_meta))

        # 更新全局主外键映射
        # print(f"create_extension_instance: ext_query_text = {ext_query_text}. ext_query_meta = {ext_query_meta}.")
        curr_parser = SQLParser(sql_text=ext_query_text, workload=instance.workload)
        curr_parser.update_global_foreign_mapping()

        # 将估计的基数根据query_meta的实例进行补全
        subquery_dict, single_table_dict = card_dict['subquery'], card_dict['single_table']

        # print(f"create_extension_instance: query_meta = {ext_query_meta[0]}. subquery_dict = {subquery_dict}. single_table_dict = {single_table_dict}")

        # 创建Extension实例
        extension_res = ExtensionInstance(query_text=ext_query_text, query_meta = ext_query_meta, \
            query_ctrl = instance.query_ctrl, subquery_estimation = subquery_dict, \
            single_table_estimation = single_table_dict, subquery_true = deepcopy(instance.true_card_dict), \
            single_table_true = deepcopy(instance.true_single_table), external_info = {})

        # 返回结果
        return extension_res
    
    @utils.timing_decorator
    def plan_error_estimation(self, query_text, query_meta, card_dict, target_table = None, restrict_order = True):
        """
        添加真实基数估计的action收益预测
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        # print(f"plan_error_estimation: query_meta = {query_meta}. query_text = {query_text}.")
        # 生成extension实例
        local_extension = self.create_extension_instance(ext_query_meta=query_meta, \
                            ext_query_text=query_text, card_dict=card_dict)
        
        # # 先估计一下预期收益
        # estimate_cost_pair, eval_result_list = local_estimator.cost_pair_integration()
        estimate_cost_pair = estimation_interface.estimate_plan_benefit(local_extension, self.mode, 
            target_table, card_est_spec = "graph_corr_based", plan_sample_num=benefit_config["plan_sample_num"],
            restrict_order = restrict_order)
        
        estimate_benefit = utils.benefit_calculate(estimate_cost_pair[0], \
                            estimate_cost_pair[1], clip_factor=10.0)

        estimate_p_error = estimate_cost_pair[1] / estimate_cost_pair[0]
        # 再返回实例，用于任务启动
        return estimate_benefit, estimate_p_error, estimate_cost_pair

    @utils.timing_decorator
    def result_filter(self, query_list, meta_list, result_batch, \
            card_dict_list, candidate_num = 3, target_table = None):
        """
        针对结果的过滤
    
        Args:
            query_list:
            meta_list:
            result_batch:
            card_dict_list:
            candidate_num:
            target_table:
        Returns:
            benefit: 预估收益
            candidate_list: 候选列表
            card_dict_candidate: 候选估计基数
        """
        mode = self.mode
        assert mode in ("under-estimation", "over-estimation")
        assert len(query_list) == len(meta_list) == len(result_batch)

        query_list_str = '\n'.join([str(item) for item in query_list])
        meta_list_str = '\n'.join([str(meta) for meta in meta_list])

        # print(f"AdvanceAnalyzer.result_filter: query_list = \n{query_list_str}.")
        # print(f"AdvanceAnalyzer.result_filter: meta_list = \n{meta_list_str}.")
        # 2024-03-09: 确保所有input list长度相同
        assert len(query_list) == len(meta_list) == len(result_batch) == len(card_dict_list), \
            f"AdvanceAnalyzer.result_filter: query_list = {len(query_list)}. meta_list = {len(meta_list)}. "\
            f"result_batch = {len(result_batch)}. card_dict_list = {len(card_dict_list)}."

        if len(query_list) == 0:
            # 如果没有结果的话，直接返回空集
            return -1.0, [], []

        benefit, candidate_list, card_dict_candidate = 0, [], []

        # debug = True
        debug = False

        for query, meta, result, card_dict in zip(query_list, meta_list, result_batch, card_dict_list):
            if debug:
                # 用于debug
                plan1: physical_plan_info.PhysicalPlan = result[0][1]       # 目标物理计划
                plan2: physical_plan_info.PhysicalPlan = result[0][2]       # 实际物理计划
                
                # # 打印查询计划以进行比较
                # plan1.show_plan(), plan2.show_plan()
                # 只打印join_order
                print("plan1 jo: ", end="")
                plan1.show_join_order()
                print("plan2 jo: ", end="")
                plan2.show_join_order()
                print()

            mode_dict = {
                "under-estimation": True,
                "over-estimation": False
            }

            flag = mode_dict[mode] == result[0][0]
            # print(f"plan_match: result[0][0] = {result[0][0]}. mode_dict[mode] = {mode_dict[mode]}. flag = {flag}.")

            if flag:
                # 如果查询计划符合预期，加入到结果集中
                candidate_list.append((query, meta))
                card_dict_candidate.append(card_dict)

            if self.save_intermediate == True:
                # 保存历史结果
                self.record_list.append((meta, query, card_dict, target_table, flag))
            
        benefit_list, out_list, card_dict_out = [], [], []

        for (query, meta), card_dict in zip(candidate_list, card_dict_candidate):
            benefit, p_error, _ = self.plan_error_estimation(query, meta, 
                card_dict, target_table, restrict_order=False)

            # print(f"result_filter: p_error = {p_error:.2f}. target_table = {target_table}")
            # if p_error > 1 + 1e-5:
            benefit_list.append(benefit)
            out_list.append((query, meta))
            card_dict_out.append(card_dict)

        if len(out_list) == 0:
            print(f"result_filter: no valid action. meta_list = {meta_list}.")
            return 0.0, [], []
        else:
            # print(f"result_filter: benefit_list = {benefit_list}.")
            combined_lists = sorted(zip(benefit_list, out_list, card_dict_out), \
                                    key=lambda a: a[0], reverse=True)
            benefit_sorted, sorted_list, card_dict_sorted = zip(*combined_lists)

            # print(f"result_filter: benefit = {benefit_sorted[0]:.2f}. sorted_list = {utils.list_round(benefit_sorted)}.")
            return benefit_sorted[0], sorted_list, card_dict_sorted
    
# %%
