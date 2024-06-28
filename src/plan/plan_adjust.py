#!/usr/bin/env python
# -*- coding: utf-8 -*-

# %%

from utility import workload_spec, utils
from result_analysis import case_analysis
from grid_manipulation import grid_preprocess
from plan import node_query

# %%

class PlanTuner(object):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, workload, ce_method):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        self.workload = workload
        self.ce_method = ce_method
        self.alias_mapping = workload_spec.abbr_option[workload]
        self.query_meta, self.pseudo_card_dict, self.actual_card_dict = None, None, None
        self.bins_builder = grid_preprocess.get_bins_builder_by_workload(workload)

    def load_plan_instance(self, query_meta: tuple, pseudo_card_dict: dict, actual_card_dict: dict):
        """
        加载计划实例
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        self.query_meta, self.pseudo_card_dict, self.actual_card_dict = \
            query_meta, pseudo_card_dict, actual_card_dict

        self.pseudo_analyzer = case_analysis.construct_case_instance(\
            self.query_meta, pseudo_card_dict, self.workload)
        self.actual_analyzer = case_analysis.construct_case_instance(\
            self.query_meta, actual_card_dict, self.workload)

        print(f"pseudo join_order: {self.pseudo_analyzer.get_plan_join_order('estimation')}")
        print(f"actual join_order: {self.actual_analyzer.get_plan_join_order('estimation')}")


    def adjust_multi_tables(self, table_list, factor_list):
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
    

    def adjust_table(self, table_name, factor):
        """
        {Description}

        Args:
            table_name:
            factor:
        Returns:
            out_card_dict:
        """
        # 
        table_alias = self.alias_mapping[table_name]
        subquery_true, single_table_true, subquery_est, single_table_est = \
            utils.extract_card_info(self.actual_card_dict, dict_copy=True)

        for alias_tuple in subquery_true.keys():
            if table_alias in alias_tuple:
                subquery_true[alias_tuple] *= factor

        single_table_true[table_alias] *= factor

        for alias_tuple in subquery_est.keys():
            if table_alias in alias_tuple:
                subquery_est[alias_tuple] *= factor
            
        single_table_est[table_alias] *= factor
        out_card_dict = utils.pack_card_info(subquery_true, \
            single_table_true, subquery_est, single_table_est, dict_copy=True)

        return out_card_dict
    

    def verify_card_dict(self, card_dict_in):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        local_analyzer = case_analysis.construct_case_instance(self.query_meta, card_dict_in, self.workload)
        print(local_analyzer.get_plan_join_order("estimation")[1])
        print(local_analyzer.p_error)
        print(local_analyzer.estimation_physical.get_physical_spec())
        print(self.pseudo_analyzer.estimation_physical.get_physical_spec())
        print(local_analyzer.estimation_cost, self.pseudo_analyzer.estimation_cost)
        print(local_analyzer.true_cost, self.pseudo_analyzer.true_cost)


    def adjust_plan(self,):
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
    

    def adjust_meta_by_factor(self, table_name, factor):
        """
        {Description}
    
        Args:
            table_name:
            factor:
        Returns:
            return1:
            return2:
        """
        pass

    def find_target_predicate(self, table_name, init_cond, init_card, target_card):
        """
        {Description}

        Args:
            table_name:
            init_cond:
            init_card:
            target_card:
        Returns:
            target_cond:
            return2:
        """
        target_meta = ([], [])
        target_card = 0
    
        return target_meta, target_card


    def new_meta_verification(self, new_meta):
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
