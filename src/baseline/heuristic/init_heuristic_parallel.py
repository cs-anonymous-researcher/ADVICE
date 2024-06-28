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
from baseline.utility import parallel
from baseline.heuristic import init_heuristic
from collections import deque

# %%


class InitGreedyParallelSearcher(parallel.ParallelSearcher, init_heuristic.InitGreedyPlanSearcher):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self,schema_total, workload, time_limit = 60000, ce_type:str = "internal", \
        q_error_threshold = 2.0, candidate_num = 50, table_num_dist = {3: 1.0}):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        init_heuristic.InitGreedyPlanSearcher.__init__(self, schema_total, workload, time_limit, ce_type)
        parallel.ParallelSearcher.__init__(self, workload, total_task_num=5)

        self.q_error_threshold = q_error_threshold
        self.candidate_num = candidate_num
        self.table_num_dist = table_num_dist
        self.candidate_queue = deque()
    

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
        workload_gen_num, candidate_num = 100, 5
        schema_init_num, schema_final_num = 3, 5
        out_num = 5
        if len(self.candidate_queue) == 0:
            query_list, meta_list = self.generate_candidates(workload_gen_num, \
                candidate_num, schema_init_num, schema_final_num, out_num)

            for query, meta in zip(query_list, meta_list):
                self.candidate_queue.append((query, meta))

        item = self.candidate_queue.popleft()
        query_text, query_meta = item
        return query_text, query_meta            


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
        print("call InitGreedyParallelSearcher.launch_search_process")
        return super().launch_search_process(total_time, with_start_time)
    
    
