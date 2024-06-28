#!/usr/bin/env python
# -*- coding: utf-8 -*-


# %%
import time
from experiment import parallel_exploration
import psycopg2 as pg
from asynchronous import construct_input, task_management, state_inspection
from collections import defaultdict
import numpy as np
# from comparison import aggregation_func
from plan import plan_template, advance_search, node_query
from utility import utils

# %%

class TemplateEvaluator(object):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, test_explorer: parallel_exploration.ParallelForestExploration):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        self.test_explorer = test_explorer
        self.template_list = self.get_template_list()
        self.template_result_dict = defaultdict(lambda: \
            {"query_list":[], "meta_list":[], "card_dict_list":[]})
        self.template_metrics_dict = defaultdict(list)

        workload = test_explorer.workload
        self.db_conn = pg.connect(**construct_input.conn_dict(workload))
        # self.inspector = state_inspection.WorkloadInspector(workload=workload, db_conn=self.db_conn)        
        # self.agent = task_management.TaskAgent(workload=workload)

        self.result_dict = {}

    def result_aggregation(self, template_id, agg_config = {"topk":5}):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            max_p_error: 
            topk_p_error: 
            median_p_error: 
            mean_p_error:
        """
        metrics_list = self.template_metrics_dict[template_id]
        p_error_list = [item[1]/item[2] for item in metrics_list]

        print("result_aggregation: len(p_error_list) = {}.".format(len(p_error_list)))

        if len(p_error_list) > 0:
            median_p_error = np.median(p_error_list)
            mean_p_error = np.mean(p_error_list)
            max_p_error = np.max(p_error_list)
            topk_p_error = np.sort(p_error_list)[::-1][min(agg_config['topk'], len(p_error_list) - 1)]

            print("result_aggregation: max = {:.2f}. topk = {:.2f}. median = {:.2f}. mean = {:.2f}.".\
                format(max_p_error, topk_p_error, median_p_error, mean_p_error))

            return max_p_error, topk_p_error, median_p_error, mean_p_error
        else:
            print("result_aggregation: warning! no result generated")
            return 0.0, 0.0, 0.0, 0.0
    
    def show_existing_templates(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        print(self.template_list)

    def show_template_result(self,):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        for k, v in self.template_eval_result.items():
            print("template_id = {}. eval_result = {}.".format(k, v))

    def get_template_list(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        return self.test_explorer.template_id_list

    def add_batch_result(self, template_id, query_list, \
        meta_list, card_dict_list):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        self.template_result_dict[template_id]['query_list'].extend(query_list)
        self.template_result_dict[template_id]['meta_list'].extend(meta_list)
        self.template_result_dict[template_id]['card_dict_list'].extend(card_dict_list)

    def add_batch_metrics(self, template_id, result_list):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        self.template_metrics_dict[template_id].extend(result_list)

    def evaluate_single_template(self, template_id, total_time, save_result = True):
        """
        {Description}

        Args:
            template_id: 
            total_time:
            save_result:
        Returns:
            res_metrics:
            return2:
        """
        if isinstance(template_id, int):
            template_id = str(template_id)

        self.test_explorer.reset_search_state()     # 重置状态信息
        
        # 固定到一个template上面
        self.test_explorer.set_search_config(template_id_list=[template_id,])   
        query_list, meta_list, result_list, card_dict_list =\
            self.test_explorer.parallel_workload_generation(total_time=\
                total_time, save_result=save_result)

        self.add_batch_result(template_id, \
            query_list, meta_list, card_dict_list)
        self.add_batch_metrics(template_id, result_list=result_list)

        res_metrics = self.result_aggregation(template_id=template_id)
        return res_metrics
    
    def evaluate_multi_templates(self, template_id_list, method, total_time, save_result = True):
        """
        {Description}
    
        Args:
            template_id_list:
            method:
            total_time:
            save_result:
        Returns:
            res_metrics:
            return2:
        """
        print(f"evaluate_multi_templates: method = {method}. template_id_list = {template_id_list}")
        
        available_list = ['polling_based', 'epsilon_greedy', 'correlated_MAB']
        assert method in available_list, f"available_list = {available_list}"

        self.test_explorer.reset_search_state()     # 重置状态信息
        pesudo_id = ",".join(sorted(template_id_list))

        # 固定到一个template上面
        self.test_explorer.set_search_config(template_id_list=template_id_list)

        # 调整探索策略
        if method == "polling_based":
            query_list, meta_list, result_list, card_dict_list = \
                self.test_explorer.polling_based_workload_generation(total_time=total_time)
        elif method == "epsilon_greedy":
            query_list, meta_list, result_list, card_dict_list = \
                self.test_explorer.Epsilon_Greedy_workload_generation(total_time=total_time)
        elif method == "correlated_MAB":
            query_list, meta_list, result_list, card_dict_list = \
                self.test_explorer.Correlated_MAB_workload_generation(total_time=total_time)
                    
        self.add_batch_result(pesudo_id, \
            query_list, meta_list, card_dict_list)
        self.add_batch_metrics(pesudo_id, result_list=result_list)
        res_metrics = self.result_aggregation(template_id=pesudo_id)
        return res_metrics

    def evaluate_all_templates(self, total_time, save_result: bool = False):
        """
        评估所有模版的效果
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        template_list = self.get_template_list()
        result_dict = {}

        for template_id in template_list:
            result_dict[template_id] = self.evaluate_single_template(template_id = template_id,\
                total_time=total_time, save_result = save_result)

        return result_dict
    
    
    def evaluate_estimation_quality(self, template_id, iter_num = 10, init_strategy = "multi-loop"):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        template_instance: plan_template.TemplatePlan = \
            self.test_explorer.get_template_by_id(id=template_id)
        
        template_instance.set_ce_handler(external_handler=self.test_explorer.ce_handler)
        mode = template_instance.mode

        local_map = { "under-estimation": "under", "over-estimation": "over" }
        
        init_info = {
            "target": local_map[mode], "num": 100,      # 增加探索实例
            "min_card": 5000, "max_card": 5000000,
        }

        workload = self.test_explorer.workload
        external_dict = self.test_explorer.external_dict
        selected_tables = self.test_explorer.selected_tables

        result_list = []
        for iter_idx in range(iter_num):
            selected_query, selected_meta, true_card, estimation_card = \
                template_instance.explore_init_query(external_info = init_info)

            root_query = node_query.get_query_instance(workload=workload, query_meta=selected_meta, \
                                                       external_dict=external_dict)

            subquery_dict, single_table_dict = \
                template_instance.get_plan_cardinalities(in_meta=selected_meta, query_text=selected_query)

            # 加载真实基数
            root_query.add_true_card(subquery_dict, "subquery")
            root_query.add_true_card(single_table_dict, "single_table")

            tree_info = {
                "query_instance": root_query,
                "selected_tables": selected_tables,
                "max_depth": 5,
                "timeout": 60000
            }

            test_tree = advance_search.AdvanceTree(external_info=tree_info, max_step=10, \
                    template_id=0, mode=template_instance.mode, init_strategy=init_strategy)
            
            test_root = test_tree.root
            # test_tree.expand_node()
            action_list = test_root.get_available_actions()

            result_dict = {}
            # 枚举所有的action
            for action in action_list:
                res_local = test_root.verify_action(action=action)
                result_dict[action] = res_local
            
            result_list.append(result_dict)
            
        return result_list

    def evaluate_estimation_all(self, iter_num = 10):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        template_list = self.get_template_list()
        result_dict = {}

        for template_id in template_list:
            result_dict[template_id] = self.evaluate_estimation_quality(\
                template_id=template_id, iter_num=iter_num)
            # break   # 提前中止，用于测试
        
        return result_dict
    

    def evaluate_init_actions_quality(self, template_id, iter_num = 10, init_strategy = "multi-loop"):
        """
        评测template在创建root以后init_actions的质量，主要考虑两个问题：

        1. 加上真实基数的估计以后，能否提升探索的效率
        2. 能否利用estimation card adjust的方法，减小估计基数探索的次数
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        template_instance: plan_template.TemplatePlan = \
            self.test_explorer.get_template_by_id(id=template_id)
        
        template_instance.set_ce_handler(external_handler=self.test_explorer.ce_handler)
        mode = template_instance.mode

        local_map = {
            "under-estimation": "under",
            "over-estimation": "over"
        }
        
        init_info = {
            "target": local_map[mode],
            "min_card": 1000,
            "max_card": 5000000,
            "num": 100      # 增加探索实例
        }

        workload = self.test_explorer.workload
        external_dict = self.test_explorer.external_dict
        selected_tables = self.test_explorer.selected_tables

        for iter_idx in range(iter_num):
            selected_query, selected_meta, true_card, estimation_card = \
                template_instance.explore_init_query(external_info = init_info)

            root_query = node_query.get_query_instance(workload=workload, query_meta=selected_meta, \
                                                       external_dict=external_dict)

            subquery_dict, single_table_dict = \
                template_instance.get_plan_cardinalities(in_meta=selected_meta, query_text=selected_query)

            # 加载真实基数
            root_query.add_true_card(subquery_dict, "subquery")
            root_query.add_true_card(single_table_dict, "single_table")

            print(f"evaluate_init_actions_quality: estimaiton_subquery = {utils.dict_str(root_query.estimation_card_dict)}.")
            print(f"evaluate_init_actions_quality: true_subquery = {utils.dict_str(root_query.true_card_dict)}.")
            print(f"evaluate_init_actions_quality: estimaiton_single_table = {utils.dict_str(root_query.estimation_single_table)}.")
            print(f"evaluate_init_actions_quality: true_single_table = {utils.dict_str(root_query.true_single_table)}.")

            tree_info = {
                "query_instance": root_query,
                "selected_tables": selected_tables,
                "max_depth": 5,
                "timeout": 60000
            }

            test_tree = advance_search.AdvanceTree(external_info=tree_info, max_step=10, \
                    template_id=0, mode=template_instance.mode, init_strategy=init_strategy)
            
            test_root = test_tree.root
            # test_tree.expand_node()
            action_list = test_root.get_available_actions()

            # 枚举所有的action
            for action in action_list:
                test_root.take_action(action=action, sync=True)


    def evaluate_init_actions_all(self, iter_num = 10, init_strategy = "multi-loop"):
        """
        评测所有template在创建root以后init_actions的质量
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        template_list = self.get_template_list()
        result_dict = {}

        for template_id in template_list:
            # result_dict[template_id] = self.evaluate_single_template(template_id = template_id,\
            #     iter_num=iter_num, save_result = save_result)
            result_dict[template_id] = self.evaluate_init_actions_quality(template_id = template_id,\
                 iter_num=iter_num, init_strategy=init_strategy)
            # break
        
        return result_dict
    

    def evaluate_init_query_quality(self, template_id, iter_num, save_result = True):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        template_instance: plan_template.TemplatePlan = \
            self.test_explorer.get_template_by_id(id=template_id)
        mode = template_instance.mode


        local_map = {
            "under-estimation": "under",
            "over-estimation": "over"
        }
        
        external_info = {
            "target":  local_map[mode],
            "min_card": 5000,
            "max_card": 5000000
        }

        result_dict = {}

        for grid_plan_id in template_instance.get_id_list():
            # 选择一个grid_plan进行探索
            template_instance.bind_grid_plan(grid_plan_id)
            template_instance.set_ce_handler(external_handler=self.test_explorer.ce_handler)
            result_list = []

            for iter_idx in range(iter_num):
                selected_query, selected_meta, true_card, estimation_card = \
                    template_instance.explore_init_query(external_info = external_info)

                if mode == "under-estimation":
                    q_error = true_card / estimation_card
                elif mode == "over-estimation":
                    q_error = estimation_card / true_card
                else:
                    raise ValueError(f"evaluate_init_query_quality: template_instance.mode = {mode}")
                
                result_list.append((selected_query, selected_meta, \
                                    true_card, estimation_card, q_error))
                
                print(f"init_query_quality: template_id = {template_id}. mode = {mode}. "
                    f"iter_idx = {iter_idx}. q_error = {q_error:.2f}.")
            
            result_dict[grid_plan_id] = result_list

        return result_dict
    

    def evaluate_init_query_all(self, iter_num, save_result = True):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        template_list = self.get_template_list()
        result_dict = {}

        for template_id in template_list:
            result_dict[template_id] = self.evaluate_init_query_quality(template_id = template_id,\
                iter_num=iter_num, save_result = save_result)

        return result_dict
# %%

