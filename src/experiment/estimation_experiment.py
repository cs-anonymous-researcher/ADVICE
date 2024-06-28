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
from estimation import template_estimation

# %%

class MultiQueryEstimationTest(object):
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
        self.test_estimator = template_estimation.TemplateEstimator(workload=workload)
        # self.template_id_list = self.test_estimator.
        self.template_id_list = []
    
    def test_on_multi_templates(self, template_num, query_num, extend_num):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        if template_num == "all":
            template_num = len(self.template_id_list)
        else:
            template_num = min(template_num, len(self.template_id_list))

        result_dict = {}
        for template_id in self.template_id_list[:template_num]:
            explore_res_list = self.test_on_multi_queries(template_id, query_num, extend_num)
            result_dict[template_id] = explore_res_list

        return result_dict

    def test_on_multi_queries(self, template_id, query_num, extend_num, config="equal_diff"):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            explore_res_list:
            return2:
        """
        self.test_estimator.load_template_instance(template_id=template_id)

        explore_res_list = []
        for query_id in range(query_num):
            base_query_text, base_query_meta, subquery_dict, single_table_dict = \
                self.test_estimator.generate_query_instance(set_query=True)
            local_res_list = []
            for extend_id in range(extend_num):
                selected_table, query_text, query_meta, result = \
                        self.test_estimator.extend_random_table(config=config, with_verify=True)
                local_res_list.append((selected_table, query_text, query_meta, result))
            # explore_res_list.append((base_query_text, base_query_meta, \
            #                          subquery_dict, single_table_dict, local_res_list))
            explore_res_list.append((base_query_text, base_query_meta, local_res_list))

        return explore_res_list
    
    def template_result_aggregation(self, template_res_list):
        """
        针对结果进行聚合，给出实验效果
        
        Args:
            arg1:
            arg2:
        Returns:
            query_summary_dict:
            res2:
        """
        query_summary_dict = {}
        for base_query_text, base_query_meta, query_res_list in template_res_list:
            metrics_dict = self.query_result_aggregation(query_res_list)
            query_summary_dict[str(base_query_meta)] = metrics_dict
        return query_summary_dict


    def query_result_aggregation(self, query_res_list):
        """
        聚合一个查询的拓展结果，目前考虑以下几个指标:
        P_error的平均差距
        真实基数下的Plan不一致概率
        真实基数下和混合基数下查询计划一致的概率
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        summary_dict = {}

        p_error_diff_list = []
        plan_consistent_list = []
        optimal_plan_match_list = []

        for selected_table, query_text, query_meta, result in query_res_list:
            actual_benefit, estimated_benefit, comparison_res_list = result
            p_error_diff_list.append(max(actual_benefit / estimated_benefit, \
                                         estimated_benefit / actual_benefit))
            plan_consistent_list.append(actual_benefit < (1.0 + 1e-5))

            print("query_result_aggregation: comparison_res_list[:3] = {}.".format(comparison_res_list[:3]))
            # 打印comparison中元素的类型
            for item in comparison_res_list:
                print(" ".join([str(type(j)) for j in item]))

            optimal_plan_match_list.append(np.sum([item[1][0] for item in \
                comparison_res_list]) / len(comparison_res_list))

        summary_dict = {
            "average_error_diff": np.average(p_error_diff_list),
            "plan_consistent_rate": np.average(plan_consistent_list),
            "optimal_match_rate": np.average(optimal_plan_match_list)
        }
        return summary_dict

    def multi_templates_aggregation(self, template_res_dict: dict):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """

        template_summary_dict = {}
        for template_idx, template_res in template_res_dict.items():
            template_summary_dict[template_idx] = self.template_result_aggregation(template_res)

        return template_summary_dict

    def output_result(self,):
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

