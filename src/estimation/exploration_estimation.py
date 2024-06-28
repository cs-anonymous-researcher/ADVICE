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

from sklearn.linear_model import LinearRegression

# %%

class AbstractEstimator(object):
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
        pass
    
    def load_query_info(self, query, cost, exec_time):
        raise NotImplementedError("load_query_info")

    def load_task_info(self, benefit):
        raise NotImplementedError("load_task_info")

    def get_estimator_name(self, ):
        raise NotImplementedError("get_estimator_name")
    
    def get_expected_benefit(self,):
        raise NotImplementedError("get_expected_benefit")

    def get_learned_factor(self,):
        raise NotImplementedError("get_learned_factor")

# %%

class DummyEstimator(AbstractEstimator):
    """
    一个仿制的估计器

    Members:
        field1:
        field2:
    """

    def __init__(self,):
        super().__init__()

    def load_query_info(self, query, cost, exec_time):
        pass

    def load_query_batch(self, query_list, cost_list, exec_time_list):
        pass

    def load_task_info(self, benefit):
        pass

    def get_estimator_name(self, ):
        return "Dummy"
    
    def get_expected_benefit(self,):
        return 1.2

    def get_learned_factor(self,):
        return 0.01

# %%

class LinearEstimator(AbstractEstimator):
    """
    用来探索中的查询时间估计以及短任务的收益估计

    Members:
        field1:
        field2:
    """

    def __init__(self, ):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        self.query_list, self.cost_list = [], []
        self.exec_time_list = []
        self.benefit_list = []

    def load_query_info(self, query, cost, exec_time):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        self.query_list.append(query)
        self.cost_list.append(cost)
        self.exec_time_list.append(exec_time)

    def load_task_info(self, benefit):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        self.benefit_list.append(benefit)
    
    def get_expected_benefit(self,):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        if len(self.benefit_list) > 0:
            return np.average(self.benefit_list)
        else:
            print("get_expected_benefit: benefit_list is empty.")
            return 1.0

    def get_learned_factor(self,):
        """
        获得学习到的factor，如果cost_list是空的，
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        if len(self.cost_list) > 0:
            data_X = np.array(self.cost_list).reshape(-1, 1)
            data_y = np.array(self.exec_time_list)
            # print(f"get_learned_factor: data_X.shape = {data_X.shape}. data_y.shape = {data_y.shape}.")
            model = LinearRegression(fit_intercept=False)
            model.fit(data_X, data_y)
            return model.coef_[0]
        else:
            # print("get_learned_factor: cost_list is empty")
            return 1e-6

    def get_estimator_name(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        return "Linear"

# %%
name_mapping = {
    "dummy": DummyEstimator(),
    "linear": LinearEstimator()
}

