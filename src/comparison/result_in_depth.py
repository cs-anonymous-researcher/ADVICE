#!/usr/bin/env python
# -*- coding: utf-8 -*-

# %%
import numpy as np
from utility import utils
import os
from os.path import join as p_join
# %%

from result_analysis import case_analysis, execution_evaluation
from comparison import result_construction, result_base
from utility import workload_parser
from copy import deepcopy
from data_interaction import postgres_connector
import pickle

# %%

class CaseSpecificConstructor(result_construction.ReportConstructor):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, workload: str, intermediate_dir: str, result_dir: str, new_config: dict = None, ssh_info: dict = {}):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        self.new_config = new_config
        self.ssh_info = ssh_info 
        super().__init__(workload, intermediate_dir, result_dir)


    def find_representative_cases(self, config: dict, topk = None, error_threshold = 5.0):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            result_dict: item=(query_list, card_list)
            reference_dict: item=error_list
        """
        result_dict, reference_dict = {}, {}
        out_dir, val_dict = self.parse_config(config, \
            out_format=('query', 'result', 'card_dict'))
        val_dict: dict = val_dict
        
        for method_key, (query_list, result_list, card_list) in val_dict.items():
            error_list = [item[0] for item in result_list]
            index_sorted = np.argsort(error_list)[::-1]     # 根据error大小进行排序
            # 
            # print(f"find_representative_cases: error_list = {utils.list_round(error_list, 2)}")
            # print(f"find_representative_cases: max_error = {error_list[index_sorted[0]]: .2f}. "\
            #       f"min_error = {error_list[index_sorted[-1]]: .2f}.")

            query_local, card_local, error_local = [], [], []
            if topk is None:
                topk = len(index_sorted)
                index_selected = index_sorted[:topk]
            elif isinstance(topk, tuple):
                start, end = topk
                index_selected = index_sorted[start: end]
            elif isinstance(topk, int):
                index_selected = index_sorted[:topk]
            else:
                raise TypeError(f"find_representative_cases: type(topk) = {type(topk)}")

            for idx in index_selected:
            # for idx in index_sorted[:topk]:
                if error_list[idx] < error_threshold and len(query_local) > 0:
                    # 比threshold小，直接退出
                    break
                query_local.append(query_list[idx])
                card_local.append(card_list[idx])
                error_local.append(error_list[idx])

            # result_dict[method] = (query_local, card_local)
            # reference_dict[method] = error_local
            result_dict[method_key] = (query_local, card_local)
            reference_dict[method_key] = error_local

        return result_dict, reference_dict

    def eval_execution_performance(self, candidate_dict: dict, reference_dict: dict, start_mode, search_method_subset, ce_method_subset):
        """
        评估实际执行的表现情况
    
        Args:
            candidate_dict:
            arg2:
        Returns:
            out_dict:
            return2:
        """
        evaluator = execution_evaluation.ActualExecutionEvaluation(self.workload, self.new_config, self.ssh_info)
        out_dict = {}
        # print(f"eval_execution_performance: candidate_dict.keys = {candidate_dict.keys()}.")
        for method_key, (query_list, card_dict_list) in candidate_dict.items():
            p_error_list = reference_dict[method_key]
            if isinstance(method_key, tuple):
                ce_method, search_method = method_key
                # print(f"eval_execution_performance: ce_method = {ce_method}. search_method = {search_method}.")
                if ce_method_subset is not None and ce_method not in ce_method_subset:
                    continue
            else:
                search_method = method_key

            if search_method_subset is not None and search_method not in search_method_subset:
                continue
            
            print(f"eval_execution_performance: evaluator.result_construction. method_key = {method_key}.")
            out_dict[method_key] = evaluator.result_construction(query_list, card_dict_list, p_error_list, start_mode)
        return out_dict
    
    def construct_execution_result(self, config: dict, topk = None, error_threshold = 5.0, 
            start_mode = "warm", search_method_subset = None, ce_method_subset = None):
        """
        {Description}
    
        Args:
            config:
            topk:
            error_threshold:
        Returns:
            out_dict:
            reference_dict:
        """
        # 2024-04-01: 打印函数相关参数        
        candidate_dict, reference_dict = \
            self.find_representative_cases(config, topk, error_threshold)

        print(f"construct_execution_result: topk = {topk}. error_threshold = {error_threshold:.2f}. start_mode = {start_mode}. "
              f"search_method_subset = {search_method_subset}. ce_method_subset = {ce_method_subset}.")
        
        out_dict = self.eval_execution_performance(candidate_dict, reference_dict, \
            start_mode, search_method_subset, ce_method_subset)
        return out_dict, reference_dict


# %%
from query import ce_injection
from plan import node_utils
 
class EstCardMutationAnalyzer(result_construction.ReportConstructor):
    """
    替换估计基数

    Members:
        field1:
        field2:
    """

    def __init__(self, workload: str, ce_str: str, intermediate_dir: str, \
                 result_dir: str, config_dir: str = "/home/lianyuan/Research/CE_Evaluator/evaluation/config"):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        self.ce_handler = ce_injection.get_ce_handler_by_name(workload, ce_str)
        super().__init__(workload, intermediate_dir, result_dir, config_dir)
        self.query_list, self.meta_list, self.card_dict_list, \
            self.result_list = [], [], [], []

        self.expt_res_dict, self.card_new_dict = {}, {}     # 不同次实验的结果字典
        self.card_queries_list, self.card_new_list = [], []
    

    def add_instance_external(self, query, meta, card_dict, result):
        """
        直接从外部加载结果实例
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        self.query_list.append(query), self.meta_list.append(meta)
        self.card_dict_list.append(card_dict), self.result_list.append(result)

    def clean_all_results(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        self.query_list, self.meta_list = [], []
        self.card_dict_list, self.result_list = [], []

    def update_result(self, config: dict, search_subset = None, estimation_subset = None):
        """
        利用已有结果更新pickle（注意保存旧的结果）
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        try:
            result_id_list = config["result_list"]
        except KeyError as e:
            print(f"load_result: meet KeyError. config = {config.keys()}.")
            raise e

        for result_id in result_id_list:
            instance_meta = self.result_meta_dict[result_id]
            obj_path = instance_meta['obj_path']
            est_method = instance_meta['estimation_method']
            expl_method = instance_meta['search_method']

            # 提前退出
            if search_subset is not None and expl_method not in search_subset:
                continue

            if estimation_subset is not None and est_method not in estimation_subset:
                continue
            
            result_obj, out_path = self.load_object(obj_path, with_path=True)
            
            # 2024-03-28: 筛掉不合法的结果
            # result_obj = self.filter_wrap_func(result_obj, True, False, True, est_method)
            # print(len(result_obj[0]))
            if len(result_obj) == 5:
                # 有time信息的情况
                # result_obj = result_obj[:4]
                result_obj, time_info = result_obj[:4], result_obj[4]
            else:
                time_info = None

            expected_num = len(result_obj[0])
            result_obj, valid_index = self.filter_wrap_func(result_obj, True, False, False, est_method, return_index = True)
            query_local, meta_local, result_local, card_dict_local = result_obj

            query_ref, meta_ref, result_ref, card_dict_ref = self.expt_res_dict[(expl_method, est_method)]

            assert len(card_dict_ref) == len(card_dict_local) == len(query_local) == len(meta_local)
            assert query_local == query_ref
            # assert meta_local == meta_ref
            # assert card_dict_local == card_dict_ref

            # 考虑计算新的result
            result_replace = []
            for query_text, query_meta, card_dict in zip(query_local, meta_local, card_dict_ref):
                subquery_true, single_table_true, subquery_est, single_table_est = utils.extract_card_info(card_dict)
                if utils.card_dict_valid_check(subquery_true, single_table_true) == False:
                    result_replace.append((0.0, -1, -1))
                else:
                    analyzer = case_analysis.CaseAnalyzer(query_text, \
                        query_meta, (1.0, 10.0, 10.0), card_dict, self.workload)
                    result_replace.append((analyzer.p_error, analyzer.estimation_cost, analyzer.true_cost))
            
            print(f"est_method = {est_method}. expl_method = {expl_method}. result_replace = {result_replace}.")
            if time_info is not None:
                # 新的时间信息
                time_new = result_base.update_time_info(time_info, valid_index, expected_num)
                result_new = query_local, meta_local, result_replace, card_dict_ref, time_new
            else:
                result_new = query_local, meta_local, result_replace, card_dict_ref

            print(f"out_path = {out_path}.")
            utils.dump_pickle(result_new, out_path)
            # if len(result_obj) == 4:
            # elif len(result_obj) == 5:
            #     query_local, meta_local, result_local, card_dict_local, time_info = result_obj
            # else:
            #     continue
        return True
    
    def load_external_time_info(self, query_ref, result_path):
        """
        {Description}
    
        Args:
            query_list:
            result_path:
        Returns:
            return1:
            return2:
        """
        result_old = self.load_object(result_path)
        time_list, time_start, time_end = result_old[-1]
        query_list = result_old[0]
        valid_index = [idx for idx, query in enumerate(query_list) if query in query_ref]

        print(f"load_external_time_info: len(query_list) = {len(query_list)}. len(valid_index) = {len(valid_index)}.")
        time_selected = utils.list_index(time_list, valid_index)
        return time_selected, time_start, time_end

    def truncate_result(self, config: dict, search_subset = None, estimation_subset = None):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        try:
            result_id_list = config["result_list"]
        except KeyError as e:
            print(f"load_result: meet KeyError. config = {config.keys()}.")
            raise e

        for result_id in result_id_list:
            instance_meta = self.result_meta_dict[result_id]
            obj_path = instance_meta['obj_path']
            est_method = instance_meta['estimation_method']
            expl_method = instance_meta['search_method']

            # 提前退出
            if search_subset is not None and expl_method not in search_subset:
                continue

            if estimation_subset is not None and est_method not in estimation_subset:
                continue
            
            result_obj, out_path = self.load_object(obj_path, with_path=True)
            
            # 2024-03-28: 筛掉不合法的结果
            # assert len(result_obj) == 5, f"len(result_obj) = {len(result_obj)}."
            if len(result_obj) == 5:
                # 有time信息的情况
                result_obj, time_info = result_obj[:4], result_obj[4]
            else:
                old_path = out_path.replace("/experiment_obj/history_pickle/", \
                    "/experiment_obj(backup)/history_pickle(backup)/")
                time_info = self.load_external_time_info(result_obj[0], old_path)

            valid_index, time_new = result_base.truncate_time_info(time_info, adjust_factor = 2.5)
            result_truncated = utils.list_index_batch(result_obj, valid_index)
            result_out = result_truncated + [time_new,]
            result_out = tuple(result_out)
            print(f"out_path = {out_path}.")
            utils.dump_pickle(result_out, out_path)
        return True
    

    def load_result(self, config: dict, load_mode = "list", \
                    search_subset = None, estimation_subset = None):
        """
        加载结果
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        try:
            result_id_list = config["result_list"]
        except KeyError as e:
            print(f"load_result: meet KeyError. config = {config.keys()}.")
            raise e

        assert load_mode in ("list", "dict"), f"load_result: load_mode = {load_mode}"

        for result_id in result_id_list:
            instance_meta = self.result_meta_dict[result_id]
            obj_path = instance_meta['obj_path']
            est_method = instance_meta['estimation_method']
            expl_method = instance_meta['search_method']

            # 提前退出
            if search_subset is not None and expl_method not in search_subset:
                continue

            if estimation_subset is not None and est_method not in estimation_subset:
                continue
            
            result_obj = self.load_object(obj_path)

            # 2024-03-28: 筛掉不合法的结果
            if len(result_obj) == 5:
                # 有time信息的情况
                result_obj = result_obj[:4]

            result_obj = self.filter_wrap_func(result_obj, True, False, False, est_method)
            
            if len(result_obj) == 4:
                query_local, meta_local, result_local, card_dict_local = result_obj
            elif len(result_obj) == 5:
                query_local, meta_local, result_local, card_dict_local, time_info_local = result_obj
            else:
                continue
                # raise ValueError(f"load_result: len(result_obj) = {len(result_obj)}")
            
            if load_mode == "list":
                self.query_list.extend(query_local)
                self.meta_list.extend(meta_local)
                self.result_list.extend(result_local)
                self.card_dict_list.extend(card_dict_local)
            elif load_mode == "dict":
                # print("load_result")
                self.expt_res_dict[(expl_method, est_method)]= \
                    (query_local, meta_local, result_local, card_dict_local)

        assert len(self.query_list) == len(self.meta_list) == len(self.result_list) == len(self.card_dict_list)
        # print(f"load_result: len(query_list) = {len(self.query_list)}")
        return True
    
    def load_intermediate(self, in_path, mode = "list"):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        if mode == "list":
            self.card_queries_list, self.card_new_list = utils.load_pickle(in_path)
        elif mode == "dict":
            raise ValueError("")
        
    def replace_old_cards(self, card_dict_list: list, card_new_list: list, mode = "true"):
        """
        实现真实基数的替换
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        assert len(card_dict_list) == len(card_new_list)
        assert mode in ("true", "estimation")
        replaced_list = []
        for card_item, (subquery_new, single_table_new) in zip(card_dict_list, card_new_list):
            subquery_true, single_table_true, subquery_estimation, \
                single_table_estimation = utils.extract_card_info(card_item)

            if mode == "true":
                card_new = utils.pack_card_info(subquery_new, single_table_new, \
                    subquery_estimation, single_table_estimation)
            elif mode == "estimation":
                card_new = utils.pack_card_info(subquery_true, single_table_true, \
                    subquery_new, single_table_new)
            replaced_list.append(card_new)

        return replaced_list
    

    def load_external_card(self, obj_path, mode, card_type):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        assert mode in ("list", "dict")
        assert card_type in ("true", "estimation")
        if mode == "list":
            with open(obj_path, "rb") as f_in:
                card_obj: list = pickle.load(f_in)

            # self.assemble_result()
            if card_type == "true":
                pass
            elif card_type == "estimation":
                pass
        elif mode == "dict":
            with open(obj_path, "rb") as f_in:
                card_obj: dict = pickle.load(f_in)

            for (expl_method, ce_method), flatten_card_list in card_obj.items():
                print(f"load_external_card: expl_method = {expl_method}. ce_method = {ce_method}.")
                query_list, meta_list, result_list, card_dict_list = self.expt_res_dict[(expl_method, ce_method)]

                card_queries_list = self.construct_estimation_queries(query_list, card_dict_list)
                out_query_list = self.flatten_queries(card_queries_list)
                # card_list = self.ce_handler.get_cardinalities(out_query_list)
                assert len(out_query_list) == len(flatten_card_list), \
                    f"load_external_card: len(out_query_list) = {len(out_query_list)}. len(flatten_card_list) = {len(flatten_card_list)}."

                card_new_list = self.assemble_result(flatten_card_list, card_queries_list)
                # self.card_new_dict[(expl_method, ce_method)] = card_new_list

                if card_type == "true":
                    card_dict_list = self.replace_old_cards(card_dict_list, card_new_list, mode="true")
                    self.expt_res_dict[(expl_method, ce_method)] = \
                        query_list, meta_list, result_list, card_dict_list
                elif card_type == "estimation":
                    self.card_new_dict[(expl_method, ce_method)] = card_new_list

        return card_obj
    

    def dump_intermediate(self, out_path):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        utils.dump_pickle((self.card_queries_list, self.card_new_list), out_path)

    def q_error_comparison(self,):
        """
        比较前后q_error

        Args:
            arg1:
            arg2:
        Returns:
            q_error_old_list:
            q_error_new_list:
        """
        # q_error = lambda card1, card2: max((card1 + 1) / (card2 + 1), (card2 + 1) / (card1 + 1))
        def q_error(card1, card2):
            if card1 is None or card2 is None:
                return 1.0
            else:
                return max((card1 + 1) / (card2 + 1), (card2 + 1) / (card1 + 1))
            
        card_true_flatten, card_new_flatten, card_old_flatten = [], [], []  
        
        for subquery_dict, single_table_dict in self.card_new_list:
            for k in sorted(subquery_dict.keys()):
                card_new_flatten.append(subquery_dict[k])

            for k in sorted(single_table_dict.keys()):
                card_new_flatten.append(single_table_dict[k])

        for card_dict in self.card_dict_list:
            subquery_true, single_table_true, subquery_est, \
                single_table_est = utils.extract_card_info(card_dict)
            
            for k in sorted(subquery_est.keys()):
                card_old_flatten.append(subquery_est[k])
                card_true_flatten.append(subquery_true[k])

            for k in sorted(single_table_est.keys()):
                card_old_flatten.append(single_table_est[k])
                card_true_flatten.append(single_table_true[k])

        assert len(card_true_flatten) == len(card_new_flatten) == len(card_old_flatten)
        q_error_old_list = [q_error(card1, card2) for card1, card2 in zip(card_true_flatten, card_old_flatten)]
        q_error_new_list = [q_error(card1, card2) for card1, card2 in zip(card_true_flatten, card_new_flatten)]

        return q_error_old_list, q_error_new_list


    def p_error_comparison(self, mode):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        assert mode in ("list","dict")
        if mode == "list":
            return self.p_error_comparison_on_list(self.query_list, 
                self.meta_list, self.card_new_list, self.card_dict_list)
        elif mode == "dict":
            result_dict = {}
            for (expl_method, search_method), (query_list, meta_list, result_list, \
                card_dict_list) in self.expt_res_dict.items():
                card_new_list = self.card_new_dict[(expl_method, search_method)]
                result_dict[(expl_method, search_method)] = self.p_error_comparison_on_list(
                    query_list, meta_list, card_new_list, card_dict_list
                )
            return result_dict

    def p_error_comparison_on_list(self, query_list, meta_list, card_new_list, card_dict_list):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        assert len(query_list) == len(meta_list) == len(card_new_list) == len(card_dict_list), \
            f"query_list = {len(query_list)}. meta_list = {len(meta_list)}. card_new_list = {len(card_new_list)}. card_dict_list = {len(card_dict_list)}."
        
        p_error_old_list, p_error_new_list = [], []

        # for query, meta, (subquery_new, single_table_new), card_dict_old in \
        #     zip(self.query_list, self.meta_list, self.card_new_list, self.card_dict_list):
        for query, meta, (subquery_new, single_table_new), card_dict_old in \
            zip(query_list, meta_list, card_new_list, card_dict_list):
            if self.verify_card_complete(meta, card_dict_old) == False:
                continue
            subquery_true, single_table_true, subquery_est, single_table_est = \
                utils.extract_card_info(card_dict_old)
            
            card_dict_new = utils.pack_card_info(deepcopy(subquery_true), \
                deepcopy(single_table_true), subquery_new, single_table_new)
            analyzer_old = case_analysis.CaseAnalyzer(query, meta, (), card_dict_old, self.workload)
            analyzer_new = case_analysis.CaseAnalyzer(query, meta, (), card_dict_new, self.workload)
            
            print(f"p_error_comparison_on_list: p_error_old = {analyzer_old.p_error:.2f}. p_error_new = {analyzer_new.p_error:.2f}")
            p_error_old_list.append(analyzer_old.p_error)
            p_error_new_list.append(analyzer_new.p_error)

        return p_error_old_list, p_error_new_list
    

    def eval_remarkable_cases(self, mode = "list", topk = 1):
        """
        重点评测P-Error大的case的正确性，验证实验结果
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        def select_func(query_list, meta_list, result_list, card_dict_list):
            assert len(query_list) == len(meta_list) == len(result_list) == len(card_dict_list), \
                f"select_func: len1 = {len(query_list)}, len2 = {len(meta_list)}, "\
                f"len3 = {len(result_list)}, len4 = {len(card_dict_list)}"
            
            cmp_list = []

            if topk is not None:
                error_list = [item[0] for item in result_list]
                idx_list = np.argsort(error_list)[::-1][:topk]
            else:
                # 表示所有结果
                idx_list = np.arange(len(query_list))

            query_subset, meta_subset, result_subset, card_dict_subset = utils.list_index_batch(
                [query_list, meta_list, result_list, card_dict_list], idx_list
            )

            for query, meta, result, card_dict in zip(query_subset, \
                    meta_subset, result_subset, card_dict_subset):
                print(f"eval_remarkable_cases: result = {result}.")
                # 2024-03-28: 暂时只验证真实基数
                report_dict = self.result_verifier.verify_instance(query, 
                    meta, result, card_dict, mode="true")
                cmp_list.append(report_dict)
            return cmp_list
        
        if mode == "list":
            cmp_list = select_func(self.query_list, self.meta_list, self.result_list, self.card_dict_list)
        elif mode == "dict":
            cmp_list_dict = {}
            for (expl_method, search_method), (query_list, meta_list, result_list, \
                card_dict_list) in self.expt_res_dict.items():
                cmp_local = select_func(query_list, meta_list, result_list, card_dict_list)
                cmp_list_dict[(expl_method, search_method)] = cmp_local

        if mode == "list":
            return cmp_list
        elif mode == "dict":
            return cmp_list_dict

    def show_remarkable_cases(self, mode = "list", topk = 1, plot_figure = False, use_new_card = True, out_dir = "./compare_figures"):
        """
        {Description}
    
        Args:
            mode:
            use_new_card:
        Returns:
            return1:
            return2:
        """
        def select_func(query_list, meta_list, card_new_list, card_dict_list):
            analyzer_list = []
            assert len(query_list) == len(meta_list) == len(card_new_list) == len(card_dict_list), \
                f"select_func: len1 = {len(query_list)}, len2 = {len(meta_list)}, len3 = {len(card_new_list)}, len4 = {len(card_dict_list)}"
            print(f"select_func: len(query_list) = {len(query_list)}.")

            for query, meta, (subquery_new, single_table_new), card_dict_old in \
                zip(query_list, meta_list, card_new_list, card_dict_list):
                if self.verify_card_complete(meta, card_dict_old) == False:
                    continue

                subquery_true, single_table_true, _, _ = \
                    utils.extract_card_info(card_dict_old)
                
                card_dict_new = utils.pack_card_info(deepcopy(subquery_true), \
                    deepcopy(single_table_true), subquery_new, single_table_new)
                analyzer = case_analysis.CaseAnalyzer(query, meta, (), card_dict_new, self.workload)
                analyzer.db_conn.conn.close()
                analyzer_list.append(analyzer)

            analyzer_sorted = sorted(analyzer_list, key=lambda item: item.p_error, reverse=True)
            test_analyzer = analyzer_sorted[0]
            result = test_analyzer.plot_plan_comparison()
            # result[0].render("out_true")
            # result[1].render("out_est")
            return result, test_analyzer
        
        if mode == "list":
            if use_new_card == True:
                result, test_analyzer = select_func(self.query_list, \
                    self.meta_list, self.card_new_list, self.card_dict_list)
            else:
                card_old_list = [utils.extract_card_info(card_dict)[2:4] \
                    for card_dict in self.card_dict_list]
                result, test_analyzer = select_func(self.query_list, \
                    self.meta_list, card_old_list, self.card_dict_list)
                
            result[0].render(p_join(out_dir, "out_true"))
            result[1].render(p_join(out_dir, "out_est"))
        elif mode == "dict":
            analyzer_dict = {}
            for (expl_method, search_method), (query_list, meta_list, result_list, \
                card_dict_list) in self.expt_res_dict.items():
                print(f"show_remarkable_cases: ({expl_method}, {search_method}): len(query_list) = {len(query_list)}.")

                if use_new_card == True:
                    card_new_list = self.card_new_dict[(expl_method, search_method)]
                    result, test_analyzer = select_func(query_list, \
                        meta_list, card_new_list, card_dict_list)
                else:
                    card_old_list = [utils.extract_card_info(card_dict)[2:4] \
                        for card_dict in card_dict_list]
                    result, test_analyzer = select_func(query_list, \
                        meta_list, card_old_list, card_dict_list)
                prefix = f"{self.workload}_{expl_method}_{search_method}"
                result[0].render(p_join(out_dir, f"{prefix}_out_true"))
                result[1].render(p_join(out_dir, f"{prefix}_out_est"))
                analyzer_dict[(expl_method, search_method)] = test_analyzer
        
        if mode == "list":
            return test_analyzer
        elif mode == "dict":
            return analyzer_dict

    def construct_estimation_queries(self, query_list, card_dict_list):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        # 确保不包含历史记录

        # for query_text, card_dict in zip(self.query_list, self.card_dict_list):
        card_queries_list = []

        for query_text, card_dict in zip(query_list, card_dict_list):
            subquery_true, single_table_true, subquery_est, \
                single_table_est = utils.extract_card_info(card_dict)
            subquery_keys, single_table_keys = \
                sorted(subquery_est.keys()), sorted(single_table_est.keys())
            
            query_parser = workload_parser.SQLParser(query_text, self.workload)
            subquery_list, single_table_list = node_utils.get_diff_queries(\
                query_parser, subquery_keys, single_table_keys)
            subquery_out = {k: v for k, v in zip(subquery_keys, subquery_list)}
            single_table_out = {k: v for k, v in zip(single_table_keys, single_table_list)}
            # self.card_queries_list.append((subquery_out, single_table_out))
            card_queries_list.append((subquery_out, single_table_out))

        # return self.card_queries_list
        return card_queries_list


    def flatten_queries(self, card_queries_list) -> list[str]:
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        query_total = []
        # for subquery_dict, single_table_dict in self.card_queries_list:
        for subquery_dict, single_table_dict in card_queries_list:
            for k in sorted(subquery_dict.keys()):
                query_total.append(subquery_dict[k])

            for k in sorted(single_table_dict.keys()):
                query_total.append(single_table_dict[k])

        return query_total
    
    def assemble_result(self, result_list, card_queries_list) -> list[str]:
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        idx = 0
        card_new_list = []
        # for subquery_dict, single_table_dict in self.card_queries_list:
        for subquery_dict, single_table_dict in card_queries_list:
            subquery_local, single_table_local = {}, {}
            for k in sorted(subquery_dict.keys()):
                subquery_local[k] = result_list[idx]
                idx += 1

            for k in sorted(single_table_dict.keys()):
                single_table_local[k] = result_list[idx]
                idx += 1

            # self.card_new_list.append((subquery_local, single_table_local))
            card_new_list.append((subquery_local, single_table_local))
        # return self.card_new_list
        return card_new_list
    

    def get_cardinalities_in_segment(self, total_query_list, split_size = 1000):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        # 将列表按指定大小分割
        split_list = [total_query_list[i:i+split_size] for i in range(0, len(total_query_list), split_size)]

        # 对每个子列表应用给定函数
        processed_list = [self.ce_handler.get_cardinalities(sub_lst) for sub_lst in split_list]

        # 拼接结果
        result_list = []
        for item in processed_list:
            result_list.extend(item)

        return result_list
    
    def get_true_cardinalities(self, total_query_list, timeout = None):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        db_conn = postgres_connector.connector_instance(self.workload)
        true_card_list = db_conn.get_cardinalities(total_query_list, timeout=timeout)
        return true_card_list
    

    def construct_new_estimation(self, mode = "list", true_card = False, update_member = True):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        print("construct_new_estimation: call this function")
        assert mode in ("list", "dict")
        if mode == "list":
            assert len(self.card_new_list) == 0, f"construct_estimation_queries: len(self.card_new_list) = {len(self.card_new_list)}"
            assert len(self.card_queries_list) == 0, f"construct_estimation_queries: len(self.card_queries_list) = {len(self.card_queries_list)}"

            card_queries_list = self.construct_estimation_queries(self.query_list, self.card_dict_list)
            out_query_list = self.flatten_queries(card_queries_list)
            # card_list = self.ce_handler.get_cardinalities(out_query_list)
            
            if true_card == False:
                card_list = self.get_cardinalities_in_segment(out_query_list)
            else:
                card_list = self.get_true_cardinalities(out_query_list)

            card_new_list = self.assemble_result(card_list, card_queries_list)

            if update_member == True:
                self.card_new_list = card_new_list
            return card_new_list
        elif mode == "dict":
            # self.card_new_dict = {}
            for (expl_method, search_method), (query_list, meta_list, result_list, card_dict_list) in self.expt_res_dict.items():                
                card_queries_list = self.construct_estimation_queries(query_list, card_dict_list)
                out_query_list = self.flatten_queries(card_queries_list)
                # card_list = self.ce_handler.get_cardinalities(out_query_list)

                if true_card == False:
                    card_list = self.get_cardinalities_in_segment(out_query_list)
                else:
                    card_list = self.get_true_cardinalities(out_query_list)

                card_new_list = self.assemble_result(card_list, card_queries_list)
                self.card_new_dict[(expl_method, search_method)] = card_new_list

            return self.card_new_dict
        
# %%

