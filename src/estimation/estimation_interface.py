#!/usr/bin/env python
# -*- coding: utf-8 -*-

# %%
from utility.common_config import option_collections
from plan import node_extension
from estimation import plan_estimation, external_card_estimation
from utility import utils
import numpy as np

# %%
card_est_input = "graph_corr_based"

def set_global_card_estimator(new_est):
    """
    设置新的基数估计器

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    global card_est_input
    if new_est not in option_collections.keys():
        raise ValueError(f"set_global_card_estimator: new_est = {new_est}. avaialble_list = {option_collections.keys()}.")
    card_est_input = new_est
    return True


@utils.timing_decorator
def estimate_plan_benefit(query_ext: node_extension.ExtensionInstance, mode: str, 
    target_table = None, card_est_spec = None, plan_sample_num = 100, restrict_order = True):
    """
    估计plan的收益

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    if card_est_spec is not None:
        local_estimator = plan_estimation.PlanBenefitEstimator(\
            query_ext, card_est_spec, mode, target_table)
    else:
        local_estimator = plan_estimation.PlanBenefitEstimator(\
            query_ext, card_est_input, mode, target_table)

    estimate_cost_pair, eval_result_list = local_estimator.cost_pair_integration(\
        plan_sample_num = plan_sample_num, restrict_order = restrict_order)
    return estimate_cost_pair


def complement_plan_cards(query_ext: node_extension.ExtensionInstance, \
        mode: str, target_table = None, card_est_spec = None, plan_sample_num = 10):
    """
    利用PlanBenefitEstimator获得card_dict

    Args:
        arg1:
        arg2:
    Returns:
        card_dict_list:
        return2:
    """
    if card_est_spec is not None:
        local_estimator = plan_estimation.PlanBenefitEstimator(\
            query_ext, card_est_spec, mode, target_table)
    else:
        local_estimator = plan_estimation.PlanBenefitEstimator(\
            query_ext, card_est_input, mode, target_table)
        
    card_dict_list = local_estimator.generate_card_sample(plan_sample_num)
    return card_dict_list

def integrate_spec_cards(query_ext: node_extension.ExtensionInstance, \
        mode: str, target_table = None, card_dict_list = [], restrict_order = True):
    """
    在特定的基数信息上进行收益求和

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    eval_result_list = []
    for card_dict in card_dict_list:
        subquery_true, single_table_true, subquery_est, \
            single_table_est = utils.extract_card_info(card_dict)
        if mode == "under-estimation":
            flag, cost1, cost2, plan1, plan2 = query_ext.two_plan_comparison(
                subquery_dict1=subquery_true, single_table_dict1=single_table_true, 
                subquery_dict2=subquery_est, single_table_dict2=single_table_est,
                keyword1="mixed", keyword2="estimation"
            )
        elif mode == "over-estimation":
            plan_flag, table_flag, cost1, cost2, plan1, plan2 = \
                query_ext.two_plan_verification_under_constraint(
                    subquery_dict1=subquery_true, single_table_dict1=single_table_true, 
                    subquery_dict2=subquery_est, single_table_dict2=single_table_est,
                    keyword1="mixed", keyword2="estimation", 
                    target_table=target_table, return_plan=True)
            
            # flag = (plan_flag or (not table_flag))
            flag = plan_flag and table_flag

            if table_flag == False and restrict_order == True:
                # 为了适应API，调整cost
                cost1 = cost2
        
        eval_result_list.append((flag, cost1, cost2, plan1, plan2))
    
    cost_pair_list = [(item[1], item[2]) for item in eval_result_list]
    reward = 0.0

    reward_mode = "error-oriented"
    if reward_mode == "error-oriented":
        error_list = [item[1]/item[0] for item in cost_pair_list]
        reward = np.average(error_list)
    elif reward_mode == "cost-oriented":
        total_cost1 = np.sum([item[0] for item in cost_pair_list])
        total_cost2 = np.sum([item[1] for item in cost_pair_list])
        reward = total_cost2 / total_cost1
    else: 
        err_str = f"integrate_spec_cards: reward_mode = {reward_mode}."
        raise ValueError(err_str)
    return reward

def upload_complete_instance(workload, query_text, query_meta, card_dict, url = "http://101.6.96.160:30007"):
    """
    {Description}

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    subquery_true, single_table_true, subquery_est, \
        single_table_est = utils.extract_card_info(card_dict)
    card_estimator = external_card_estimation.ExternalEstimator(workload, url=url)
    card_estimator.set_instance(query_text, query_meta)
    card_estimator.set_existing_card_dict(subquery_true, \
        single_table_true, subquery_est, single_table_est)
    card_estimator.upload_plan_instance()

    return True

# 待使用的函数
def upload_complete_extension(query_ext: node_extension.ExtensionInstance, url = "http://101.6.96.160:30007"):
    """
    {Description}

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    card_dict = utils.pack_card_info(query_ext.subquery_true, query_ext.single_table_true,
        query_ext.subquery_estimation, query_ext.single_table_estimation)
    upload_complete_instance(query_ext.workload, query_ext.query_text, \
        query_ext.query_meta, card_dict, url)


# %%
