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

# %%

class ResultManager(object):
    """
    读入元信息，对于结果进行批量的验证

    Members:
        field1:
        field2:
    """

    def __init__(self, meta_path = ""):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        self.meta_path = meta_path

    def load_meta_info(self, meta_path):
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


    def verify_result_by_conditions(self, filter_dict):
        """
        根据过滤器来进行结果的验证

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        pass
    
    def get_all_workloads(self,):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """

    def get_all_methods(self,):
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
