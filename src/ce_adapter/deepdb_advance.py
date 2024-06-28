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
from base import BaseCaller
from ce_adapter.remote_caller import DeepDBAdvanceRemote, remote_config

# %%

def get_DeepDB_advance_instance(workload = "job", option = "rdc"):
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    return DeepDBAdvance(workload=workload, option=option)

# %%

class DeepDBAdvance(BaseCaller):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, workload, option):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        self.workload = workload
        self.option = option
        self.remote = DeepDBAdvanceRemote(ip_address = remote_config["THU_spark08"]["ip_address"], \
            port = remote_config["THU_spark08"]["port"], option=option)
        

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
        print("deepdb_advance: initialization.")
        self.remote.launch_service(workload=self.workload)


    def get_estimation_in_batch(self, query_list, label_list):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        return self.remote.get_cardinalities(\
            sql_list = query_list, label_list = label_list)
    

    def get_estimation(self, query_text, label):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        return self.remote.get_cardinality(query_text, label)
