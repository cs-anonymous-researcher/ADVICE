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
import psycopg2 as pg
from collections import defaultdict

# %%
from baseline.utility import parallel

from baseline.random import random_search
from asynchronous import construct_input, task_management, state_inspection
from utility.utils import trace
from utility import utils
from plan import advance_search, node_extension
# %%

class RandomParallelSearcher(parallel.ParallelSearcher, random_search.RandomPlanSearcher):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, schema_total, workload, table_num_dist, \
        time_limit: int = 60000, ce_type: str = "internal", total_task_num = 5):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        # super(RandomParallelSearcher, self).__init__(schema_total, workload,
        #         table_num_dist, time_limit, ce_type)
        random_search.RandomPlanSearcher.__init__(self, schema_total, workload,
                table_num_dist, time_limit, ce_type)
        parallel.ParallelSearcher.__init__(self, workload, total_task_num=5)
        
    
    def generate_single_query(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            query_text: 
            query_meta:  
        """
        return super().single_query_generation()
    
    def launch_search_process(self, total_time, with_start_time=False):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        print("call RandomParallelSearcher.launch_search_process")
        return super().launch_search_process(total_time, with_start_time)

    
# %%
