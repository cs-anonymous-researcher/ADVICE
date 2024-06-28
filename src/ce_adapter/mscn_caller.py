#!/usr/bin/env python
# -*- coding: utf-8 -*-


# %%

import numpy as np
import pandas as pd
from os.path import join as p_join
from matplotlib import pyplot as plt
import json, os, time, re, math
import pickle

# %%
import ce_adapter.remote_caller as remote_caller
from base import BaseCaller
from ce_adapter.fcn_caller import workload_option

# workload_option = {
#     "job": {
#         "ip_address": "101.6.96.160",
#         "port": "30005",
#         "workload": "job"
#     },
#     "stats": {
#         "ip_address": "101.6.96.160",
#         "port": "30005",
#         "workload": "stats",
#     },
#     "dsb": {
#         "ip_address": "101.6.96.160",
#         "port": "30005",
#         "workload": "dsb"
#     }
# }


def get_MSCN_caller_instance(workload = "job"):
    """
    {Description}
    
    Args:
        workload:
    Returns:
        res1:
        res2:
    """
    return MSCNCaller(**workload_option[workload])


class MSCNCaller(BaseCaller):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, ip_address = "101.6.96.160", port = "30005", workload = "job"):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        self.workload = workload
        self.remote = remote_caller.MSCNRemote(ip_address, port)

        self.initialization()


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
        return self.remote.launch_service(workload=self.workload)

    def get_estimation_in_batch(self, query_list, label_list):
        """
        {Description}

        Args:
            query_list:
            label_list:
        Returns:
            card_list:
        """
        return self.remote.get_cardinalities(\
            sql_list = query_list, label_list = label_list)


    def get_estimation(self, sql_text, label):
        """
        {Description}

        Args:
            sql_text:
            label:
        Returns:
            card:
        """
        return self.remote.get_cardinality(sql_text, label)
        