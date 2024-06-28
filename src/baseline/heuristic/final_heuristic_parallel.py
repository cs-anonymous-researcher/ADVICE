#!/usr/bin/env python
# -*- coding: utf-8 -*-


# %%
from baseline.utility import parallel
from baseline.heuristic import final_heuristic
from collections import deque
# %%

class FinalGreedyParallelSearcher(parallel.ParallelSearcher, final_heuristic.FinalGreedyPlanSearcher):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, schema_total, workload, time_limit = 60000, ce_type:str = "internal", \
        q_error_threshold = 2.0, candidate_num = 50, table_num_dist = {5: 1.0}):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        final_heuristic.FinalGreedyPlanSearcher.__init__(self, schema_total, workload, time_limit, ce_type)
        parallel.ParallelSearcher.__init__(self, self.workload, total_task_num=5)

        self.q_error_threshold = q_error_threshold
        self.candidate_num = candidate_num
        self.table_num_dist = table_num_dist
        self.candidate_queue = deque()

    def generate_candidates(self, ):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        query_list, meta_list, label_list = self.search_initializer.workload_generation(\
            table_num_dist=self.table_num_dist, total_num=self.candidate_num, timeout=30000)
        
        estimation_list = self.workload_evaluation(query_list, meta_list)

        result_list = []
        for query, meta, label, estimation in \
            zip(query_list, meta_list, label_list, estimation_list):
            q_error = max(label / (estimation + 2.0), estimation / (label + 2.0))
            if q_error > self.q_error_threshold:
                result_list.append((query, meta, q_error))

        result_list.sort(key=lambda a: a[2], reverse=True)
        self.candidate_queue.extend(result_list)
        return len(result_list)
    
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
        max_try_times = 5
        while len(self.candidate_queue) == 0:
            result_num = self.generate_candidates()
            print(f"generate_single_query: result_num = {result_num}")
            max_try_times -= 1
            if max_try_times <= 0:
                raise ValueError("generate_single_query: max_try_times = 0")
            
        item = self.candidate_queue.popleft()
        
        query_text, query_meta, q_error = item
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
        print("call FinalGreedyParallelSearcher.launch_search_process")
        return super().launch_search_process(total_time, with_start_time)
    
    

# %%
