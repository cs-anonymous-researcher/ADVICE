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

from plan import stateful_analysis, node_extension, node_query
from utility import utils

# %%

class BenefitOrientedAnalyzer(stateful_analysis.StatefulAnalyzer):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, query_instance: node_query.QueryInstance, mode: str, exploration_dict: dict = {}, 
            split_budget = 100, all_distinct_states = set()):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        super().__init__(query_instance, mode, exploration_dict, split_budget, all_distinct_states)

    @utils.timing_decorator
    def init_all_actions(self, table_subset, mode = "random"):
        """
        重写初始化所有动作的函数，返回的结果从action_list, value_list升级到
        构建新的BenefitOrientedNode需要的元素

        Args:
            table_subset:
            mode:
        Returns:
            node_info_list: 由(table_name, benefit, query_instance, extension_instance)这样的tuple结构组成
            return2:
        """
        action_list, value_list = super().init_all_actions(table_subset, mode)
        node_info_list = []

        for table_name, benefit in zip(action_list, value_list):
            candidate_list, card_dict_list = self.action_result_dict[table_name]
            (query_text, query_meta), card_dict = candidate_list[0], card_dict_list[0]

            # 用于创建node实例
            extension_instance: node_extension.ExtensionInstance = \
                self.create_extension_instance(query_meta, query_text, card_dict)
            query_instance: node_query.QueryInstance = extension_instance.construct_query_instance(self.instance)

            # 
            assert isinstance(query_instance, node_query.QueryInstance), \
                f"BenefitOrientedAnalyzer.init_all_actions: query_instance = {query_instance}."
            assert isinstance(extension_instance, node_extension.ExtensionInstance), \
                f"BenefitOrientedAnalyzer.init_all_actions: extension_instance = {extension_instance}."
            
            node_info_list.append((table_name, benefit, query_instance, extension_instance))

        return node_info_list



# %%
