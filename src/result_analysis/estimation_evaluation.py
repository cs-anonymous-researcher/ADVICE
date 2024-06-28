#!/usr/bin/env python
# -*- coding: utf-8 -*-


# %%

from query import query_exploration, query_construction
from plan import node_extension
from estimation import plan_estimation
from result_analysis import res_verification
from collections import defaultdict
import numpy as np

# %%


class EstimationEvaluator(object):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, workload, result_verifier: res_verification.ResultVerifier, external_estimator = "equal_diff"):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        self.workload = workload
        self.alias_mapping = query_construction.abbr_option[workload]
        # self.estimator = plan_estimation.PlanBenefitEstimator()
        self.result_verifier = result_verifier
        self.external_estimator = external_estimator
        self.query_ctrl = query_exploration.QueryController(workload=workload)


    def load_result(self, signature):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        instance_list = self.result_verifier.load_external_result(signature=signature)
        self.invalid_instances = self.result_verifier.get_invalid_instances()
        self.valid_instances = self.result_verifier.get_valid_instances()
        self.instance_list = instance_list

    def verify_valid_cases(self, case_num = 1, mask_range = [1]):
        """
        {Description}

        Args:
            case_num:
            mask_range:
        Returns:
            reward_pair_list:
            return2:
        """
        return self.verify_in_list(self.valid_instances, case_num, mask_range)


    def verify_invalid_cases(self, case_num = 1, mask_range = [1]):
        """
        {Description}

        Args:
            case_num:
            mask_range:
        Returns:
            reward_pair_list:
            return2:
        """
        return self.verify_in_list(self.invalid_instances, case_num, mask_range)
    
    def verify_in_list(self, instance_list, case_num = 1, mask_range = [1]):
        """
        {Description}
        
        Args:
            instance_list:
            case_num:
            mask_range:
        Returns:
            reward_pair_list:
            res2:
        """
        reward_pair_list = []

        for instance in instance_list[:case_num]:
            query_text, query_meta, result, card_dict = instance
            for mask_num in mask_range:
                true_reward, est_reward = self.instance_analysis(\
                    query_text, query_meta, result, card_dict, mask_num)
                reward_pair_list.append((true_reward, est_reward, mask_num))

        return reward_pair_list

    def mask_true_card(self, subquery_true, selected_table = None, mask_num = 1):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        top_alias_tuple = max(subquery_true.keys(), key=lambda a:len(a))

        if selected_table is not None:
            selected_alias = self.alias_mapping[selected_table]
        else:
            selected_alias = np.random.choice(top_alias_tuple)

        subquery_masked = {}
        alias_num_dict = defaultdict(list)

        for k in subquery_true:
            alias_num_dict[len(k)].append(k)

        mask_list = []

        # 装填mask_list
        left_num = mask_num

        # for k, tuple_list in alias_num_dict.items():
        for k in sorted(alias_num_dict.keys(), reverse=True):
            tuple_list = alias_num_dict[k]
            for tuple_alias in tuple_list:
                if selected_alias in tuple_alias:
                    mask_list.append(tuple_alias)
                    left_num -= 1
                    if left_num <= 0:
                        break
            if left_num <= 0:
                break

        print(f"mask_true_card: selected_alias = {selected_alias}. mask_list = {mask_list}.")
        subquery_masked = {k: v for k, v in subquery_true.items() if k not in mask_list}
        return subquery_masked
    

    def instance_analysis(self, query_text, query_meta, result, card_dict, mask_num = 1):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        subquery_true, single_table_true = \
            card_dict['true']["subquery"], card_dict["true"]["single_table"]
        subquery_estimation, single_table_estimation = \
            card_dict["estimation"]["subquery"], card_dict["estimation"]["single_table"]

        subquery_masked = self.mask_true_card(subquery_true, mask_num=mask_num)    #
        extension_instance: node_extension.ExtensionInstance = node_extension.\
            ExtensionInstance(query_text=query_text, query_ctrl=self.query_ctrl, external_info={}, \
            query_meta=query_meta, subquery_estimation=subquery_estimation, single_table_estimation=\
            single_table_estimation, subquery_true=subquery_masked, single_table_true=single_table_true)
        
        local_estimator = plan_estimation.PlanBenefitEstimator(\
            query_extension=extension_instance, card_est_input=self.external_estimator)

        reward, eval_result_list = local_estimator.benefit_integration()
        
        return result[0], reward
# %%
