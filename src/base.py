#!/usr/bin/env python
# -*- coding: utf-8 -*-


# %%

import numpy as np
import pandas as pd
from os.path import join as p_join
from matplotlib import pyplot as plt
import json, os, time, re, math
import pickle

from query import query_construction
# %%


class BaseCaller(object):
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

    def initialization(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        raise NotImplementedError("")

    def get_estimation_in_batch(self, query_list, label_list = None):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        raise NotImplementedError("")


    def get_estimation_by_query(self, subquery_dict, single_table_dict):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        raise NotImplementedError("")
    
    def transform_query(self, query):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        raise NotImplementedError("")

    def transform_workload(self, query_list):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        raise NotImplementedError("")

    

# %%
class BaseHandler(object):
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
        self.result_fetcher = None

    def initialization(self,):
        self.result_fetcher.initialize_caller()

    def get_cardinalities(self, query_list):
        """
        不实现基数估计的方法

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        raise NotImplementedError("get_cardinalities has not implemented")


    def estimate_subqueries(self, meta_list):
        """
        {Description}
        
        Args:
            meta_list:
        Returns:
            cardinality_list:
        """
        cardinality_list = []
        subquery_list = self.construct_subqueries(meta_list)
        cardinality_list = self.get_cardinalities(subquery_list)
        return cardinality_list


    def construct_subqueries(self, meta_list):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        subquery_list = []
        for query_meta in meta_list:
            subquery_list.append(self.generate_query(query_meta))
        return subquery_list


    def pseudo_dataframe(self,):
        """
        获得一个默认的DataFrame
        
        Args:
            None
        Returns:
            out_df:
        """
        return pd.DataFrame([])

    def generate_query(self, query_meta):
        """
        {Description}
        
        Args:
            query_meta:
        Returns:
            res1:
            res2:
        """
        single_df = self.pseudo_dataframe()
        single_wrapper = query_construction.SingleWrapper(\
            single_df, query_meta, mode="Default")
        return single_wrapper.generate_current_query()

# %%