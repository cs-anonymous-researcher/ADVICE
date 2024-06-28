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
import requests
from utility.common_config import http_timeout
# %%

class ExternalCaseEstimator(object):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, workload, url = "http://101.6.96.160:30007"):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        self.workload, self.url = workload, url

    def meta_stringify(self, meta_pair_list):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        return [(str(meta1), str(meta2)) for meta1, meta2 in meta_pair_list]

    def upload_query_pair(self, meta_pair_list, distance_list):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        print("call upload_query_pair")
        meta_pair_list = self.meta_stringify(meta_pair_list)
        local_url = p_join(self.url, "save_pair")
        params_dict = {
            "meta_pair_list": meta_pair_list,
            "distance_list": distance_list
        }
        result = requests.post(url=local_url, json=params_dict, timeout=http_timeout)

        return result.json()
    

    def inference_uncertainty(self, meta_pair_list):
        """
        {Description}
    
        Args:
            meta_new_list:
            meta_ref_list:
        Returns:
            result_list:
            return2:
        """
        print("call inference_uncertainty")
        meta_pair_list = self.meta_stringify(meta_pair_list)

        local_url = p_join(self.url, "infer_pair")
        params_dict = {
            "meta_pair_list": meta_pair_list
        }
        result = requests.post(url=local_url, json=params_dict, timeout=http_timeout)
        result_list = result.json()

        return result_list

# %%
