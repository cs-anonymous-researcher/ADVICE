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
from plan import advance_analysis
from plan.node_query import QueryInstance
from workload import physical_plan_info
from utility import utils
from utility.common_config import eval_config
from estimation import estimation_interface, state_estimation
from query import query_construction

# %%
benefit_mode = "local"

def set_global_model(new_mode):
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    global benefit_mode
    if new_mode in ("local", "global"):
        benefit_mode = new_mode
    else:
        print(f"set_global_model: invalid new_mode = {new_mode}")

# %%

class StatefulAnalyzer(advance_analysis.AdvanceAnalyzer):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, query_instance: QueryInstance, mode: str, exploration_dict: dict = {}, split_budget = 100, 
        all_distinct_states = set()):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        super(StatefulAnalyzer, self).__init__(query_instance, mode, save_intermediate = True, split_budget=split_budget)
        self.exploration_dict = exploration_dict
        self.schema_list = query_instance.query_meta[0]
        self.all_distinct_states = all_distinct_states

        # print(f"StatefulAnalyzer.__init__: schema_list = {self.schema_list}. \nall_distinct_states = "\
        #       f"{all_distinct_states}. exploration_dict = {exploration_dict}")

    def set_node_info(self, template_id = -1, root_id = -1, node_id = -1):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        self.template_id, self.root_id, self.node_id = template_id, root_id, node_id

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
        # 2024-03-31: 用于测试
        # if 'state_manager' in self.exploration_dict and 'path_list' in self.exploration_dict:
        #     self.current_benefit_reference()

        assert mode in ("random", "multi-loop", "multi_loop", "reference")

        instance = self.instance
        action_list, value_list = [], []
        table_list = instance.fetch_candidate_tables(table_subset=table_subset)    # 获取所有当前可以join的表

        try:
            # 根据已有路径，完成table_list的过滤，提高探索的效率
            path_list = self.exploration_dict['path_list']
            path_dict = utils.prefix_aggregation(self.schema_list, path_list)
            print(f"StatefulAnalyzer.init_all_actions: table_list = {table_list}. schema_order = {self.schema_list}. "\
                  f"path_list = {path_list}. path_dict = {path_dict}.")
            table_filter = [t for t in table_list if t in path_dict]

            # 2024-03-30: 如果新的状态不存在于state_manager中，同样考虑生成进行探索
            for t in table_list:
                if t not in table_filter:
                    new_list: list = self.schema_list + (t,)
                    if new_list not in self.all_distinct_states:
                        # print(f"StatefulAnalyzer.init_all_actions: new_list = {new_list}.")
                        table_filter.append(t)
                        
        except TypeError as e:
            print(f"StatefulAnalyzer.init_all_actions: meet TypeError. path_list = {path_list}. exploration_dict = {self.exploration_dict}.")
            raise e
        except KeyError as e:
            print(f"StatefulAnalyzer.init_all_actions: meet KeyError. table_list = {table_list}.")
            table_filter = table_list

        # 结果保存字典
        result_dict = {}
        t1 = time.time()

        for table in table_filter:
            # 同一table的情况下评测多个meta信息，随机生成5个metas
            if mode == "random":
                # 使用新的批处理系统
                meta_list, query_list, result_batch, card_dict_list = self.multi_plans_evaluation_under_meta_list(\
                    new_table=table, meta_num=10, split_budget=self.split_budget)
            elif mode == "multi-loop" or mode == "multi_loop":
                try:
                    # temp = self.multi_plans_evaluation_under_multi_loop(new_table=table, 
                    #     meta_num=3, mode=self.mode, split_budget=self.split_budget)
                    temp = self.multi_plans_evaluation_under_multi_loop(new_table=table, 
                        meta_num=eval_config['meta_num'], loop_num=eval_config['loop_num'],
                        mode=self.mode, split_budget=self.split_budget)
                    meta_list, query_list, result_batch, card_dict_list = temp
                except ValueError as e:
                    print(f"StatefulAnalyzer.init_all_actions: meet ValueError. error = {e}")
                    print(f"init_all_actions: item[0] = {temp[0]}.")
                    print(f"init_all_actions: item[1] = {temp[1]}.")
                    print(f"init_all_actions: item[2] = {temp[2]}.")
                    raise e
            elif mode == "reference":
                new_key = (*self.schema_list, table)
                if 'ref_index_dict' not in self.exploration_dict.keys() or new_key not in self.exploration_dict['ref_index_dict'].keys():
                    temp = self.multi_plans_evaluation_under_multi_loop(new_table=table, 
                        meta_num=eval_config['meta_num'], loop_num=eval_config['loop_num'],
                        mode=self.mode, split_budget=self.split_budget)
                    meta_list, query_list, result_batch, card_dict_list = temp                
                else:
                    # 2024-03-22: 加入reference case影响谓词的选择
                    external_index_dict = self.exploration_dict['ref_index_dict']
                    print(f"StatefulAnalyzer.init_all_actions: mode = reference and external_index_dict = {external_index_dict}.")
                    print(f"StatefulAnalyzer.init_all_actions: schema_list = {self.schema_list}. new_table = {table}.")
                    ref_case_list = [(new_key, case_id) for case_id in external_index_dict[new_key]]
                    try:
                        temp = self.multi_plans_evaluation_under_reference(table, mode=self.mode, 
                            split_budget=self.split_budget, ref_case_list=ref_case_list)
                        meta_list, query_list, result_batch, card_dict_list = temp                
                    except Exception as e:
                        print(f"StatefulAnalyzer.init_all_actions: meet ValueError. error = {e}.")
                        # print(f"init_all_actions: item[0] = {temp[0]}.")
                        # print(f"init_all_actions: item[1] = {temp[1]}.")
                        # print(f"init_all_actions: item[2] = {temp[2]}.")
                        raise e
            if len(query_list) == 0:
                # 2024-03-09: 无论在何时，如果找不到结果，就直接退出
                # print(f"StatefulAnalyzer.init_all_actions: len(query_list) = 0. table = {table}. table_filter = {table_filter}.")
                continue

            benefit, candidate_list, card_dict_candidate = self.result_filter(\
                query_list, meta_list, result_batch, card_dict_list, target_table=table)

            # 添加benefit和action，如果结果不优直接暂时把action禁用了
            # if benefit > 1e-5:
            if len(candidate_list) > 0:
                result_dict[table] = candidate_list, card_dict_candidate
                value_list.append(benefit)
                action_list.append(table)

        t2 = time.time()
        self.action_result_dict = result_dict
        # print(f"StatefulAnalyzer.init_all_actions: action_list = {action_list}. value_list = {utils.list_round(value_list)}")
        print(f"StatefulAnalyzer.init_all_actions: delta_time = {t2 - t1:.2f}. len(action_list) = "\
              f"{len(action_list)}. len(table_list) = {len(table_filter)}.")
        
        return action_list, value_list

    def get_available_paths(self, target_table):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        path_list = self.exploration_dict['path_list']
        path_dict = utils.prefix_aggregation(self.schema_list, path_list)
        # print(f"get_available_paths: target_table = {target_table}. path_dict = {path_dict}. self.schema_list = {self.schema_list}. path_list = {path_list}.")
        local_list: list = deepcopy(path_dict[target_table])
        # return path_dict[target_table]
        return local_list
    

    def current_benefit_reference(self,):
        """
        生成当前收益的参考，用于future_benefit_simulation的debug
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        instance = self.instance
        query = instance.query_text
        meta = instance.query_meta
        card_dict = utils.pack_card_info(instance.true_card_dict, instance.true_single_table, 
            instance.estimation_card_dict, instance.estimation_single_table)
                                         
        state_manager: state_estimation.StateManager = self.exploration_dict['state_manager']
        path_list = self.exploration_dict['path_list']

        if 'ref_index_dict' in self.exploration_dict:
            external_index_dict = self.exploration_dict['ref_index_dict']
            tree_result = state_manager.infer_new_init_under_index_ref(meta, card_dict, 
                path_list, external_index_dict=external_index_dict)
        else:
            tree_result = state_manager.infer_spec_new_init(meta, card_dict, path_list)

        max_local = state_estimation.tree_result_max(tree_result)
        print(f"current_benefit_reference: template_id = {self.template_id}. root_id = {self.root_id}."\
              f" node_id = {self.node_id}. max_local = {max_local: .2f}")
        return max_local


    def future_benefit_simulation(self, query, meta, card_dict, target_table):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        state_manager: state_estimation.StateManager = self.exploration_dict['state_manager']
        local_extension = self.create_extension_instance(ext_query_meta=meta, \
                            ext_query_text=query, card_dict=card_dict)
        
        # 补全查询计划的基数
        card_dict_list = estimation_interface.complement_plan_cards(\
            local_extension, self.mode, target_table, plan_sample_num=10)
        
        local_paths = self.get_available_paths(target_table)
        max_list = []
        
        current_path = self.schema_list + (target_table,)
        final_benefit = 0.0
        # if len(current_path) == 4:
        #     print(f"get_available_paths(root mode): local_paths = {local_paths}. current_path = {current_path}.")
        # else:
        #     print(f"get_available_paths: local_paths = {local_paths}. current_path = {current_path}.")

        # 考虑当前的收益
        if current_path in local_paths:
            local_paths.remove(current_path)
        
        # 当前的拓展不需要考虑join_order
        current_benefit = estimation_interface.integrate_spec_cards(local_extension, 
            self.mode, target_table, card_dict_list, restrict_order=False)
        # else:
        #     # 不包含当前收益
        #     current_benefit = 0.0

        # 考虑长远的收益
        if len(local_paths) > 0:
            print(f"future_benefit_simulation: exploration_dict.keys = {self.exploration_dict.keys()}")
            for card_dict in card_dict_list:
                if "ref_index_dict" not in self.exploration_dict.keys():
                    tree_result = state_manager.infer_spec_new_init(meta, card_dict, local_paths, current_benefit)
                else:
                    external_index_dict = {k: self.exploration_dict["ref_index_dict"][k] for k in \
                        local_paths if k in self.exploration_dict["ref_index_dict"]}
                    tree_result = state_manager.infer_new_init_under_index_ref(meta, card_dict, 
                        local_paths, external_index_dict, )

                # print(f"future_benefit_simulation: tree_result = {tree_result}.")
                max_local = state_estimation.tree_result_max(tree_result)
                max_list.append(max_local)
            # return max(max_list)        # 考虑最大收益
            long_benefit = np.max(max_list)   # 考虑平均收益
            print(f"future_benefit_simulation: template_id = {self.template_id}. root_id = {self.root_id}. node_id = {self.node_id}. "\
                  f"current_benefit = {current_benefit:.2f}. long_benefit = {long_benefit:.2f}. max_list = {utils.list_round(max_list)}.")
            final_benefit = max(long_benefit, current_benefit)
        else:
            final_benefit = current_benefit
        
        return final_benefit


    def result_filter(self, query_list, meta_list, result_batch, \
            card_dict_list, candidate_num = 3, target_table = None):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        flag1 = "state_manager" in self.exploration_dict
        flag2 = "path_list" in self.exploration_dict
        flag = flag1 and flag2

        # if flag == True:
        #     print(f"StatefulAnalyzer.result_filter: state_manager = {flag1}. path_list_in = {flag2}.")
        # else:
        #     print(f"StatefulAnalyzer.result_filter: state_manager = {flag1}. path_list_in = {flag2}.")
            
        debug = False

        if benefit_mode == "global" and flag:
            # 利用未来节点估算全局收益，来选择当前动作
            mode = self.mode
            assert mode in ("under-estimation", "over-estimation")
            assert len(query_list) == len(meta_list) == len(result_batch)
            if len(query_list) == 0:
                # 如果没有结果的话，直接返回空集
                return -1.0, [], []

            benefit, candidate_list, card_dict_candidate = 0, [], []

            # join order的正确性验证
            for query, meta, result, card_dict in zip(query_list, meta_list, result_batch, card_dict_list):
                if debug:
                    # 用于debug
                    plan1: physical_plan_info.PhysicalPlan = result[0][1] 
                    plan2: physical_plan_info.PhysicalPlan = result[0][2]
                    
                    # 打印查询计划以进行比较
                    # 只打印join_order
                    print("plan1 jo: ", end=""); plan1.show_join_order()
                    print("plan2 jo: ", end=""); plan2.show_join_order()
                    print()
                mode_dict = { "under-estimation": True, "over-estimation": False }

                flag = mode_dict[mode] == result[0][0]
                if flag:
                    # 如果查询计划符合预期，加入到结果集中
                    candidate_list.append((query, meta))
                    card_dict_candidate.append(card_dict)

                if self.save_intermediate == True:
                    # 保存历史结果
                    self.record_list.append((meta, query, card_dict, target_table, flag))
                
            benefit_list, out_list, card_dict_out = [], [], []

            # 未来收益的验证
            for (query, meta), card_dict in zip(candidate_list, card_dict_candidate):
                # 验证未来的收益
                benefit = self.future_benefit_simulation(query, meta, card_dict, target_table)
                benefit_list.append(benefit)
                out_list.append((query, meta))
                card_dict_out.append(card_dict)

            if len(out_list) == 0:
                print(f"result_filter: no valid action. meta_list = {meta_list}.")
                return 0.0, [], []
            else:
                # print(f"result_filter: benefit_list = {benefit_list}.")
                assert len(benefit_list) == len(out_list) == len(card_dict_out)
                
                combined_lists = sorted(zip(benefit_list, out_list, card_dict_out), \
                                        key=lambda a: a[0], reverse=True)
                benefit_sorted, sorted_list, card_dict_sorted = zip(*combined_lists)

                print(f"StatefulAnalyzer.result_filter: schema_list = {meta[0]}. benefit = {benefit_sorted[0]:.2f}. "\
                      f"sorted_list = {utils.list_round(benefit_sorted)}.")
                return benefit_sorted[0], sorted_list, card_dict_sorted
        else:
            # 考虑局部收益
            return super().result_filter(query_list, meta_list, 
                result_batch, card_dict_list, candidate_num, target_table)


    @utils.timing_decorator
    def multi_plans_evaluation_under_reference(self, new_table, column_num = 1, meta_num = 5, \
            total_num = 20, split_budget = 100, mode = "under-estimation", ref_case_list = []):
        """
        在reference case下生成多个plans
        
        Args:
            new_table: 
            column_num: 
            meta_num: 
            total_num:
            split_budget:
            mode:
            ref_case_list: 
        Returns:
            meta_global:
            query_global: 
            result_global:
            card_dict_global:
        """
        assert len(ref_case_list) > 0, f"multi_plans_evaluation_under_reference: ref_case_list = []"
        # 
        print(f"multi_plans_evaluation_under_reference: ref_case_list = {ref_case_list}. exploration_dict = {self.exploration_dict}.")
        state_manager_ref: state_estimation.StateManager = self.exploration_dict['state_manager']
        new_alias = utils.abbr_option[self.instance.workload][new_table]
        # 通过ref_case_list获得具体的case
        card_list = []
        for schema_key, case_id in ref_case_list:
            # 找出对应的query_meta以及单表的cardinality
            single_stable = state_manager_ref.state_dict[schema_key]
            info_dict = single_stable.instance_list[case_id]
            new_card = info_dict['card_dict']['true']['single_table'][new_alias]
            print(f"multi_plans_evaluation_under_reference: schema_key = {schema_key}. case_id = {case_id}. new_table = {new_table}. new_card = {new_card}.")
            card_list.append(new_card)
        
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

        #
        pred_ctrl = advance_analysis.PredicateController(self.instance, new_table, \
                    bins_local, marginal_local, column_list, mode)
        
        pred_ctrl.pred_generation(num=total_num)
        idx_list, meta_list = pred_ctrl.pred_selection_by_reference(card_list, meta_num)

        query_global, meta_global, result_global, card_dict_global = [], [], [], []

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

        # 添加结果
        for (col_name, pred_idx), result in zip(idx_list, card_dict_list):
            subquery_local, single_table_local = result['subquery'], result['single_table']
            pred_ctrl.pred_load(pred_idx, col_name, subquery_local, single_table_local)

        # return , meta_global, result_global
        params = meta_global, query_global, result_global, card_dict_global
        # print(f"multi_plans_evaluation_under_multi_loop: len(params) = {len(params)}")
        return params

# %%
