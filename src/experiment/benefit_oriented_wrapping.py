#!/usr/bin/env python
# -*- coding: utf-8 -*-

# %%

from experiment import stateful_wrapping, stateful_exploration, benefit_oriented_exploration
from experiment.stateful_wrapping import default_sample_config, default_template_config
from utility import common_config

# %%

class BenefitOrientedSearcher(stateful_wrapping.StatefulParallelSearcher):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, schema_total, workload, tmpl_meta_path, time_limit = 60000, max_step = 100, ce_type:str = "internal", 
            expl_estimator = "dummy", resource_config = stateful_exploration.default_resource_config, 
            card_est_input = "graph_corr_based", action_selection_mode = "local", root_selection_mode = "normal", 
            noise_parameters = None, sample_config = default_sample_config, template_config = default_template_config, 
            split_budget = 100):
        """
        {Description}

        Args:
            schema_total:
            workload:
            tmpl_meta_path:
            time_limit: 
            max_step: 
            ce_type: 
            expl_estimator: 
            resource_config: 
            card_est_input: 
            action_selection_mode: 
            root_selection_mode: 
            noise_parameters: 
            sample_config: 
            template_config:
            split_budget:
        """
        super().__init__(schema_total, workload, tmpl_meta_path, time_limit, max_step, ce_type, expl_estimator, 
            resource_config, card_est_input, action_selection_mode, root_selection_mode, noise_parameters, sample_config, 
            template_config, split_budget)
        
        # 设置新的forest_explorer
        self.forest_explorer = benefit_oriented_exploration.BenefitOrientedExploration(workload, 
            expt_config=self.expt_config, expl_estimator=expl_estimator, resource_config=resource_config, 
            max_expl_step=max_step, tmpl_meta_path=tmpl_meta_path, card_est_input=card_est_input, 
            action_selection_mode=action_selection_mode, root_selection_mode=root_selection_mode, 
            noise_parameters=noise_parameters, warm_up_num=common_config.warm_up_num, 
            tree_config=common_config.tree_config_dict[workload], init_strategy=common_config.init_strategy)

    def func_name1(self,):
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

    def func_name2(self,):
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
