#!/usr/bin/env python
# -*- coding: utf-8 -*-

# %%

import numpy as np
import pandas as pd
from os.path import join as p_join
from matplotlib import pyplot as plt
import json, os, time, re, math
import pickle
from data_interaction import postgres_connector, data_management   # 直接从数据库获得结果
from data_interaction.mv_management import conditions_apply
from query import query_construction, ce_injection, query_exploration
from utility import workload_parser, utils
from workload import physical_plan_info


# %%


class ResultDisplay(object):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self,):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        self.query_list, self.meta_list, self.result_list, self.card_dict_list = [], [], [], []

    def load_workload(self, workload_tuple):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        self.query_list, self.meta_list, self.result_list, self.card_dict_list = workload_tuple


    def workload_transform(self, query_list, meta_list, result_list, card_dict_list):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            instance_list:
            return2:
        """
        instance_list = [(query, meta, result, card_dict) for query, meta, result, card_dict in \
            zip(query_list, meta_list, result_list, card_dict_list)]

        return instance_list

    def show_workload(self,):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        for query_text, query_meta in zip(self.query_list, self.meta_list):
            print("sql = {}. meta = {}.".format(query_text, query_meta))

# %%

class ResultVerifier(object):
    """
    基数估计正确性的验证器

    Members:
        field1:
        field2:
    """

    def __init__(self, workload = "job", ce_type = "internal"):
        """
        {Description}

        Args:
            workload:
            arg2:
        """
        self.workload = workload
        self.ce_handler = ce_injection.get_ce_handler_by_name(workload=workload, ce_type=ce_type)
        self.db_conn: postgres_connector.Connector = postgres_connector.connector_instance(workload=workload)
        self.query_ctrl = query_exploration.QueryController(db_conn=self.db_conn, workload=self.workload)

        self.query_text, self.query_meta = None, None
        self.query_parser = None
        self.data_manager = data_management.DataManager(wkld_name=workload)

    def reset_ce_handler(self, ce_type):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        self.ce_handler = ce_injection.get_ce_handler_by_name(\
            workload=self.workload, ce_type=ce_type)

    def load_external_result(self, signature, result_dir = "/home/lianyuan/Research/CE_Evaluator/result"):
        """
        {Description}
        
        Args:
            signature:
            result_dir:
        Returns:
            instance_list:
            res2:
        """
        result_path = p_join(result_dir, "{}.pkl".format(signature))
        query_list, meta_list, result_list, card_dict_list = utils.load_pickle(result_path)

        self.instance_list = list(zip(query_list, meta_list, result_list, card_dict_list))
        print("len(instance_list) = {}.".format(len(self.instance_list)))
        return self.instance_list
    

    def get_valid_instances(self, threshold = 1 + 1e-5):
        """
        获得p_error大的实例
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        selected_instances = [item for item in self.instance_list if item[2][0] > threshold]
        return selected_instances

    def get_invalid_instances(self, threshold = 1 + 1e-5):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        selected_instances = [item for item in self.instance_list if item[2][0] <= threshold]
        return selected_instances
    
    def verbose_external_result(self, num = -1):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            report_list:
        """
        if num < 0:
            num = len(self.instance_list)

        selected_list = self.instance_list[:num]
        result = self.verbose_workload(instance_list=selected_list)
        return result


    def verify_workload(self, instance_list = None):
        """
        {Description}
    
        Args:
            instance_list:
            arg2:
        Returns:
            return1:
            return2:
        """
        if instance_list is None:
            instance_list = self.instance_list

        verify_res_list = []
    
        for query_text, query_meta, result, card_dict in instance_list:
            report_local = self.verify_instance(query_text, query_meta, result, card_dict)
            verify_res_list.append(report_local)

        return verify_res_list
    

    def set_instance(self, query_text, query_meta, result, card_dict):
        """
        设置实例
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        # print(f"set_instance: query_text = {query_text}.") 
        # print(f"set_instance: query_meta = {query_meta}.")

        self.query_text, self.query_meta = query_text, query_meta
        self.result, self.card_dict = result, card_dict
        self.query_parser = workload_parser.SQLParser(\
            sql_text=self.query_text, workload=self.workload)
        self.query_ctrl.set_query_instance(query_text, query_meta)

    def verify_card_complete(self, query_meta, subquery_true, single_table_true, \
                             subquery_estimation, single_table_estimation):
        """
        {Description}
    
        Args:
            query_meta:
            subquery_true: 
            single_table_true: 
            subquery_estimation: 
            single_table_estimation: 
        Returns:
            missing_list:
        """
        subquery_repr_list = self.data_manager.get_meta_subqueries(query_meta)
        schema_list = query_meta[0]
        single_table_repr_list = [self.data_manager.tbl_abbr[s] for s in schema_list]

        missing_list = []
        # 处理subquery
        for k in subquery_repr_list:

            if k not in subquery_true or subquery_true[k] is None or subquery_true[k] < -1e-5:
                missing_list.append((k, "true"))

            if k not in subquery_estimation or \
                subquery_estimation[k] is None or subquery_estimation[k] < -1e-5:
                missing_list.append((k, "estimation"))

        # 处理single_table
        for k in single_table_repr_list:
            if k not in single_table_true or single_table_true[k] is None or single_table_true[k] < -1e-5:
                missing_list.append((k, "true"))

            if k not in single_table_estimation or \
                single_table_estimation[k] is None or single_table_estimation[k] < -1e-5:
                missing_list.append((k, "estimation"))

        # print(f"verify_card_complete: missing_list = {missing_list}")
        return missing_list
    
    def construct_physical_plans(self, query_text, query_meta, card_dict):
        """
        {Description}
    
        Args:
            query_text:
            query_meta:
            card_dict:
        Returns:
            true_physical:
            estimation_physical:
        """
        self.query_ctrl.set_query_instance(query_text=query_text, query_meta=query_meta)
        subquery_true, single_table_true, subquery_estimation, \
            single_table_estimation = utils.extract_card_info(card_dict)

        true_plan: dict = self.query_ctrl.get_plan_by_external_card(subquery_true, single_table_true)
        estimation_plan: dict = self.query_ctrl.get_plan_by_external_card(subquery_estimation, single_table_estimation)
        db_conn = self.query_ctrl.db_conn

        true_physical = physical_plan_info.PhysicalPlan(query_text = query_text, \
            plan_dict = true_plan, db_conn = db_conn)
        estimation_physical = physical_plan_info.PhysicalPlan(query_text = query_text, \
            plan_dict = estimation_plan, db_conn = db_conn)
        
        return true_physical, estimation_physical
    
    def verify_instance(self, query_text, query_meta, result, card_dict, mode = "all"):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        mode_dict = {
            "all": {    
                "card_dict_complete": True, "true_card": True,
                "est_card": True, "hint_plan": True
            },
            "estimation": {
                "card_dict_complete": True, "true_card": False,
                "est_card": True, "hint_plan": False
            },
            "true": {
                "card_dict_complete": True, "true_card": True,
                "est_card": False, "hint_plan": False
            },
            "plan": {
                "card_dict_complete": True, "true_card": False,
                "est_card": False, "hint_plan": True            
            }
        }

        assert mode in mode_dict.keys(), f"verify_instance: mode = {mode}. available_list = {list(mode_dict.keys())}."
        curr_conf = mode_dict[mode]
        print("verify_instance: query_text = {}. query_meta = {}.".format(query_text, query_meta))

        subquery_true, single_table_true = \
            card_dict["true"]["subquery"], card_dict["true"]["single_table"]
        subquery_estimation, single_table_estimation = \
            card_dict["estimation"]["subquery"], card_dict["estimation"]["single_table"]

        if curr_conf["card_dict_complete"] == True:
            missing_list = self.verify_card_complete(query_meta, subquery_true, single_table_true,
                                subquery_estimation, single_table_estimation)

        # 基数的验证
        self.query_parser = workload_parser.SQLParser(sql_text=query_text, workload=self.workload)
        self.query_ctrl.set_query_instance(query_text=query_text, query_meta=self.query_parser.generate_meta())

        estimation_error_list, true_error_list = self.verify_cardinalities(self.query_parser, \
                    subquery_true, single_table_true, subquery_estimation, single_table_estimation,
                    verify_true=curr_conf['true_card'], verify_estimation=curr_conf['est_card'])

        if curr_conf['hint_plan'] == True:
            # 查询计划的验证
            # 根据基数获得查询计划
            true_plan: dict = self.query_ctrl.get_plan_by_external_card(subquery_true, single_table_true)
            estimation_plan: dict = self.query_ctrl.get_plan_by_external_card(subquery_estimation, single_table_estimation)
            db_conn = self.query_ctrl.db_conn

            # 验证真实基数
            self.verify_plan_card(plan_dict = true_plan, subquery_ref = subquery_true, \
                                single_table_ref=single_table_true)
            
            # 验证估计基数
            self.verify_plan_card(plan_dict = estimation_plan, subquery_ref=subquery_estimation, \
                                single_table_ref=single_table_estimation)

            true_physical = physical_plan_info.PhysicalPlan(query_text = query_text, \
                plan_dict = true_plan, db_conn = db_conn)
            estimation_physical = physical_plan_info.PhysicalPlan(query_text = query_text, \
                plan_dict = estimation_plan, db_conn = db_conn)
            
            p_error, cost2, cost1 = result
            flag, cost1_actual, cost2_actual = self.verify_cost(subquery_dict = subquery_true, single_table_dict = single_table_true,
                plan1 = true_physical, plan2 = estimation_physical, cost1 = cost1, cost2 = cost2)

        if mode == "all":
            report_dict = {
                "cardinality": {
                    "estimation_error_list": estimation_error_list,
                    "true_error_list": true_error_list
                },
                "cost": {
                    "flag": flag,
                    "cost1_result": cost1, "cost1_actual": cost1_actual,
                    "cost2_result": cost2, "cost2_actual": cost2_actual
                },
                "metrics": {
                    "p_error": p_error
                }
            }
        elif mode == "estimation":
            report_dict = {
                "cardinality": {
                    "estimation_error_list": estimation_error_list,
                },
                "card_dict_complete": missing_list
            }
        else:
            report_dict = {}

        return report_dict


    def get_card_hint_query(self, query_text, subquery_dict, single_table_dict, out_mode = "verbose"):
        """
        {Description}
        
        Args:
            query_text:
            subquery_dict:
            single_table_dict:
            out_mode:
        Returns:
            res1:
            res2:
        """
        hint_sql_text = self.db_conn.inject_cardinalities_sql(\
            sql_text=query_text, subquery_dict=subquery_dict, single_table_dict=single_table_dict)
        
        if out_mode == "verbose":
            hint_sql_text = hint_sql_text.replace("", "")
        elif out_mode == "json":
            hint_sql_text = hint_sql_text.replace("JSON", "")
        else:
            raise ValueError("get_card_hint_query: out_mode = {}.".format(out_mode))
        
        return hint_sql_text


    # def get_full_hint_query(self, query_text, subquery_dict, single_table_dict, \
    #                         leading, physical_operator_dict, out_mode = "verbose"):
    def get_card_hint_query(self, query_text, subquery_dict, single_table_dict, out_mode = "verbose"):
        """
        {Description}
        
        Args:
            query_text:
            card_dict:
            leading:
            physical_operator_dict:
            out_mode:
        Returns:
            res1:
            res2:
        """
        hint_sql_text = self.db_conn.inject_cardinalities_sql(\
            sql_text=query_text, subquery_dict=subquery_dict, single_table_dict=single_table_dict)
        
        return hint_sql_text

    def verify_plan_card(self, plan_dict, subquery_ref, single_table_ref):
        """
        根据获得的plan_dict和card_dict，判断基数是否注入成功
        
        Args:
            plan_dict:
            card_dict:
        Returns:
            res1:
            res2:
        """
        subquery_plan, single_table_plan = postgres_connector.parse_all_subquery_cardinality(plan_dict=plan_dict)
        # print("card_plan = {}.".format(card_plan))
        error_list = []
        
        for k, v in subquery_plan.items():
            if v != subquery_ref[k]:
                error_list.append((k, v, subquery_ref[k]))
            # else:
            #     pass

        for k, v in single_table_plan.items():
            if v != single_table_ref[k]:
                error_list.append((k, v, single_table_ref[k]))

        return len(error_list) == 0, error_list


    def verify_plan_operator(self, plan_dict: dict, ref_leading: str, ref_physical_operator_dict: dict):
        """
        根据获得的plan_dict，判断物理算子是否注入成功
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        leading, join_ops, scan_ops = query_exploration.get_physical_plan(plan=plan_dict)
        actual_plan = leading, join_ops, scan_ops
        ref_plan = ref_leading, ref_physical_operator_dict['join_ops'], \
            ref_physical_operator_dict['scan_ops']
        # jo_cmp_res = leading == ref_leading

        cmp_res = physical_plan_info.physical_comparison(actual_plan, ref_plan)
        return cmp_res

    def verify_cost(self, subquery_dict: dict, single_table_dict:dict, plan1: \
        physical_plan_info.PhysicalPlan, plan2: physical_plan_info.PhysicalPlan, cost1: float, cost2: float):
        """
        验证cost是否正确
        
        Args:
            subquery_dict: 真实子查询基数
            single_table_dict: 真实单表基数
            plan1: 查询计划1
            plan2: 查询计划2
            cost1: 结果代价1
            cost2: 结果代价2
        Returns:
            flag:
            cost1_actual: 
            cost2_actual:
        """
        cost1_actual = plan1.get_plan_cost(subquery_dict = subquery_dict, single_table_dict = single_table_dict)
        cost2_actual = plan2.get_plan_cost(subquery_dict = subquery_dict, single_table_dict = single_table_dict)

        cost1_result, cost2_result = cost1, cost2
        flag = (cost1_actual == cost1_result) and (cost1_result == cost2_result)
        print("verify_cost: cost1_actual = {}. cost1_result = {}. cost2_actual = {}. cost2_result = {}.".\
            format(cost1_actual, cost1_result, cost2_actual, cost2_result))
        
        return flag, cost1_actual, cost2_actual

    def verify_cardinalities(self, query_parser, subquery_true, single_table_true, \
                             subquery_estimation, single_table_estimation, \
                             verify_true = True, verify_estimation = True):
        """
        比较基数结果是否正确
    
        Args:
            query_parser:
            subquery_true:
            single_table_true:
            subquery_estimation:
            single_table_estimation:
            verify_true:
            verify_estimation:
        Returns:
            estimation_error_list: 
            true_error_list:
        """
        current_meta = query_parser.generate_meta()
        # self.query_ctrl.set_query_instance()
        subquery_repr_list = self.query_ctrl.get_all_sub_relations()
        single_table_repr_list = self.query_ctrl.get_all_single_relations()

        subquery_sql_list, single_table_sql_list = [], []

        for alias_list in subquery_repr_list:
            local_query = query_parser.construct_PK_FK_sub_query(alias_list=alias_list)
            subquery_sql_list.append(local_query)

        for alias in single_table_repr_list:
            local_query = query_parser.get_single_table_query(alias=alias)
            single_table_sql_list.append(local_query)

        def list_pair_dict_comparison(key_list, value_list, card_old_dict, out_list, mode="true_card"):
            for k, card_new in zip(key_list, value_list):
                try:
                    card_old = card_old_dict[k]
                    if (card_new != card_old):
                        print("{} unmatch. repr = {}. card_new = {}. card_old = {}.".\
                            format(mode, k, card_new, card_old))
                        out_list.append((k, card_new, card_old))
                except KeyError:
                    print("current_meta = {}. selected_key = {}. card_old_dict = {}.".\
                        format(current_meta, k, card_old_dict))

        estimation_error_list, true_error_list = [], []

        if verify_estimation == True:
            # 验证估计基数的正确性
            subquery_estimation_list = self.ce_handler.get_cardinalities(query_list=subquery_sql_list)
            single_table_estimation_list = self.ce_handler.get_cardinalities(query_list=single_table_sql_list)

            list_pair_dict_comparison(subquery_repr_list, subquery_estimation_list, \
                                      subquery_estimation, estimation_error_list, mode="estimation_card")
            list_pair_dict_comparison(single_table_repr_list, single_table_estimation_list, \
                                      single_table_estimation, estimation_error_list, mode="estimation_card")

        if verify_true == True:
            # 验证真实基数的正确性
            subquery_true_list = self.db_conn.get_cardinalities(subquery_sql_list)
            single_table_true_list = self.db_conn.get_cardinalities(single_table_sql_list)

            print(f"subquery_true_list = {subquery_true_list}.")
            print(f"single_table_true_list = {single_table_true_list}.")
            list_pair_dict_comparison(subquery_repr_list, subquery_true_list, \
                                      subquery_true, true_error_list, mode="true_card")
            list_pair_dict_comparison(single_table_repr_list, single_table_true_list, \
                                      single_table_true, true_error_list, mode="true_card")
        
        return estimation_error_list, true_error_list

    def construct_get_card_queries(self,):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            subquery_res: 
            single_table_res:
        """
        query_parser = self.query_parser
        card_dict = self.card_dict

        current_meta = query_parser.generate_meta()
        subquery_repr_list = self.query_ctrl.get_all_sub_relations()
        single_table_repr_list = self.query_ctrl.get_all_single_relations()

        subquery_sql_list, single_table_sql_list = [], []

        for alias_list in subquery_repr_list:
            local_query = query_parser.construct_PK_FK_sub_query(alias_list=alias_list)
            subquery_sql_list.append(local_query)

        for alias in single_table_repr_list:
            local_query = query_parser.get_single_table_query(alias=alias)
            single_table_sql_list.append(local_query)

        subquery_res, single_table_res = {}, {}

        for k, v in zip(subquery_repr_list, subquery_sql_list):
            #
            # subquery_res[k] = (subquery_card[k], v)
            subquery_res[k] = v
        for k, v in zip(single_table_repr_list, single_table_sql_list):
            #
            # single_table_res[k] = (single_table_card[k], v)
            single_table_res[k] = v

        return subquery_res, single_table_res
    

    def construct_card_hint_queries(self, subquery_card, single_table_card):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        query_text, query_meta = self.query_text, self.query_meta
        card_hint_query = self.db_conn.inject_cardinalities_sql(sql_text=\
            query_text, subquery_dict=subquery_card, single_table_dict=single_table_card)
        return card_hint_query
    
    def construct_plan_hint_queries(self, subquery_card, single_table_card, scan_ops, join_ops, leading):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        query_text, query_meta = self.query_text, self.query_meta

        full_hint_query = self.db_conn.get_query_under_complete_hint(sql_text=query_text,
            subquery_dict=subquery_card, single_table_dict=single_table_card, join_ops=join_ops, 
            leading_hint=leading, scan_ops=scan_ops)

        return full_hint_query
    

    
    def construct_mannual_queries(self,):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        card_dict = self.card_dict
        subquery_res, single_table_res = self.construct_get_card_queries()

        subquery_true, single_table_true = card_dict['true']['subquery'], card_dict['true']['single_table']
        subquery_est, single_table_est = \
            card_dict['estimation']['subquery'], card_dict['estimation']['single_table']

        #
        subquery_integration = {k: (subquery_true[k], subquery_est[k], v) for k, v in subquery_res.items()} 
        single_table_integration = {k: (single_table_true[k], \
            single_table_est[k], v) for k, v in single_table_res.items()}
        # 
        true_card_hint_query = self.construct_card_hint_queries(subquery_card=subquery_true, \
            single_table_card=single_table_true)
        est_card_hint_query = self.construct_card_hint_queries(subquery_card=subquery_est,\
            single_table_card=single_table_est)

        true_plan_dict = self.query_ctrl.get_plan_by_external_card(subquery_true, single_table_true)
        est_plan_dict = self.query_ctrl.get_plan_by_external_card(subquery_est, single_table_est)

        # 
        true_leading, true_join_ops, true_scan_ops = query_exploration.\
            get_physical_plan(plan=true_plan_dict)
        est_leading, est_join_ops, est_scan_ops = query_exploration.\
            get_physical_plan(plan=est_plan_dict)

        true_plan_hint_query = self.construct_plan_hint_queries(subquery_card=subquery_true, 
            single_table_card=single_table_true, scan_ops=true_scan_ops, join_ops=true_join_ops, leading=true_leading)
        est_plan_hint_query = self.construct_plan_hint_queries(subquery_card=subquery_true,
            single_table_card=single_table_true, scan_ops=est_scan_ops, join_ops=est_join_ops, leading=est_leading)

        result_dict = {
            "info": {
                "top_query": self.query_text,
                "top_meta": self.query_meta,
                "cost_true": self.result[1], 
                "cost_est": self.result[2]
            },
            "get_card_queries":{
                # "subquery": subquery_res,
                # "single_table": single_table_res
                "subquery": subquery_integration,
                "single_table": single_table_integration
            },
            "card_hint_queries": {
                "true": true_card_hint_query,
                "estimation": est_card_hint_query
            },
            "plan_hint_queries": {
                "true": true_plan_hint_query,
                "estimation":est_plan_hint_query
            }
        }
        return result_dict
    
    def print_mannual_dict(self, result_dict: dict):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        print(f"top_query = {result_dict['info']['top_query']}.")
        disable_command = "SET max_parallel_workers_per_gather = 0;"
        print(f"disable_parallel command: {disable_command}.")

        print("print get subquery card command:")
        for k, v in result_dict['get_card_queries']['subquery'].items():
            # print(f"print_mannual_dict: k = {k}. v = {v}.")
            true_card, est_card, query_text = v
            print(f"query_repr = {k}. true_card = {true_card}. est_card = {est_card}.")
            print(f"query_text = {query_text}")

        print("print get single_table card command:")
        for k, v in result_dict['get_card_queries']['single_table'].items():
            # print(f"print_mannual_dict: k = {k}. v = {v}.")
            true_card, est_card, query_text = v
            print(f"query_repr = {k}. true_card = {true_card}. est_card = {est_card}.")
            print(f"query_text = {query_text}")

        print("print true_card_hint query:")
        print(f"{result_dict['card_hint_queries']['true']}")

        print("print estimation_card_hint query:")
        print(f"{result_dict['card_hint_queries']['estimation']}")

        print(f"print true_plan_hint query: cost = {result_dict['info']['cost_true']}")
        print(f"{result_dict['plan_hint_queries']['true']}")

        print(f"print estimation_plan_hint query: cost = {result_dict['info']['cost_est']}")
        print(f"{result_dict['plan_hint_queries']['estimation']}")

# %%
