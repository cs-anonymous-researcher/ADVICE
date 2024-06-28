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

from utility import utils

# %%

class ResultSummarizer(object):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, result_dir = "/home/lianyuan/Research/CE_Evaluator/result"):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        self.result_dir = result_dir
        self.exec_res_dict = {}
        self.meta_dict = self.load_meta()

    def load_meta(self, meta_name = "meta.json"):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        meta_path = p_join(self.result_dir, meta_name)
        self.meta_path = meta_path
        return utils.load_json(meta_path)

    def dump_meta(self, ):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        utils.dump_json(self.meta_dict, self.meta_path)


    def analyze_result(self,):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        key_list = sorted(list(self.exec_res_dict.keys()))

        # for k, v in self.exec_res_dict.items():
        for k in key_list:
            v = self.exec_res_dict[k]
            print("f_name = {}. record_num = {}.".format(k, len(v[0])))


    def load_result(self,):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        for f_name in os.listdir(self.result_dir):
            if f_name.endswith(".pkl") == False:
                continue
            f_path = p_join(self.result_dir, f_name)

            with open(f_path, "rb") as f_in:
                result = pickle.load(f_in)
            query_list, meta_list, result_list, card_dict_list = result
            self.exec_res_dict[f_name] = query_list, meta_list, result_list, card_dict_list

        return self.exec_res_dict
    

    def save_result(self, name, query_list, meta_list, result_list, card_dict_list, extra_info = {}):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        pickle_obj = query_list, meta_list, result_list, card_dict_list
        result_path = p_join(self.result_dir, f"{name}.pkl")
        utils.dump_pickle(pickle_obj, result_path)
        self.meta_dict[name] = {
            "result_path": result_path,
            "instance_length": len(query_list),
            "extra_info": extra_info
        }
        self.dump_meta()


# %%
