#!/usr/bin/env python
# -*- coding: utf-8 -*-

# %%

from baseline.feedback_based import feedback_based_search
from baseline.utility import parallel
from collections import deque

# %%
from plan import node_extension

# %%

class FBBasedParallelSearcher(parallel.ParallelSearcher, feedback_based_search.FBBasedPlanSearcher):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, schema_total, workload, mode: list, time_limit = 60000, ce_type = "internal"):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        feedback_based_search.FBBasedPlanSearcher.__init__(\
            self, schema_total, workload, mode, time_limit, ce_type)
        parallel.ParallelSearcher.__init__(self, self.workload, total_task_num=5)

        self.candidate_queue = deque()
        self.result_list, self.result_target = [], 0


    def generate_single_query(self,):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        if len(self.candidate_queue) == 0: 
            if self.result_target > len(self.result_list):
                # 已经生成的任务未结束
                print("FBBasedParallelSearcher.generate_single_query: tasks haven't finished.")
                return None, None
            else:
                # 完成一个episode，重新生成任务
                if len(self.result_list) > 0:
                    print("FBBasedParallelSearcher.generate_single_query: adjust probablity table.")
                    meta_list, p_error_list = zip(*self.result_list)
                    self.update_prob_tables(meta_list, p_error_list)
                else:
                    print("FBBasedParallelSearcher.generate_single_query: no result_list item.")

                query_list, meta_list = self.workload_generation(query_num=10)

                for query_text, query_meta in zip(query_list, meta_list):
                    self.candidate_queue.append((query_text, query_meta))
                self.result_list = []
                self.result_target = len(self.candidate_queue)

        item = self.candidate_queue.popleft()
        # query_text, query_meta = item
        # return query_text, query_meta
        return item

    def add_complete_instance(self, node_sig, cost_true, cost_estimation):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        self.complete_order.append(node_sig)
        self.time_list.append(self.task_info_dict[node_sig]['start_time'])
        
        curr_ext: node_extension.ExtensionInstance = self.extension_ref_dict[node_sig]
        query = curr_ext.query_text
        meta = curr_ext.query_meta
        p_error = cost_estimation / cost_true
        result = p_error, cost_estimation, cost_true
        card_dict = {
            "true": {
                "subquery": curr_ext.subquery_true,
                "single_table": curr_ext.single_table_true
            },
            "estimation": {
                "subquery": curr_ext.subquery_estimation,
                "single_table": curr_ext.single_table_estimation
            }
        }

        self.result_list.append((meta, p_error))
        return query, meta, result, card_dict
    

    def add_incomplete_instance(self, node_sig, cost_true, cost_estimation):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        curr_ext: node_extension.ExtensionInstance = self.extension_ref_dict[node_sig]
        meta = curr_ext.query_meta
        self.result_list.append((meta, 1.0))

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
        print("call FBBasedParallelSearcher.launch_search_process")
        return super().launch_search_process(total_time, with_start_time)

# %%
