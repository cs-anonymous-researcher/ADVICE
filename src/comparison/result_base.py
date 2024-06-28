#!/usr/bin/env python
# -*- coding: utf-8 -*-


# %%

from utility import utils, workload_parser
from result_analysis import res_verification, execution_evaluation
from os.path import join as p_join
import os
import numpy as np
from result_analysis import case_analysis

# %%

def process_time_info(result_obj):
    """
    处理时间的信息
    
    Args:
        result_obj:
        arg2:
    Returns:
        result_out:
        res2:
    """
    assert len(result_obj) in (5,), f"process_time_info: len(result_obj) = "
    if len(result_obj) == 5:
        query_list, meta_list, result_list, \
            card_dict_list, time_info = result_obj
    
    time_list, time_start, time_end = time_info
    time_normalized = [t - time_start for t in time_list]
    time_delta = time_end - time_start
    result_out = query_list, meta_list, result_list, \
        card_dict_list, time_normalized
    
    return result_out, time_delta


def update_time_info(time_info, valid_index, expected_num):
    """
    更新时间信息

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    time_list, time_start, time_end = time_info
    
    assert expected_num == len(time_list)
    time_selected = utils.list_index(time_list, valid_index)
    return time_selected, time_start, time_end


def truncate_time_info(time_info, adjust_factor = 2.5):
    """
    {Description}

    Args:
        time_info:
        adjust_factor:
    Returns:
        valid_index:
        time_info_new:
    """
    time_list, time_start, time_end = time_info
    valid_index = []
    time_normalized = [t - time_start for t in time_list]
    time_normalized = [item * adjust_factor for item in time_normalized]
    for idx, item in enumerate(time_normalized):
        if time_start + item < time_end:
            valid_index.append(idx)
    
    time_list_new = [time_normalized[idx] + time_start for idx in valid_index]
    time_info_new = time_list_new, time_start, time_end
    print(f"truncate_time_info: len(valid_index) = {len(valid_index)}. len(time_list) = {len(time_list)}.")
    return valid_index, time_info_new


# %%

class ResultBase(object):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, workload, intermediate_dir, result_dir,
        config_dir = "/home/lianyuan/Research/CE_Evaluator/evaluation/config"):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        print("call ResultBase.__init__")
        # 结果验证程序
        self.metrics_path = p_join(config_dir, workload, "metrics.json")
        self.workload = workload
        self.result_verifier = res_verification.ResultVerifier(workload)
        self.intermediate_dir = intermediate_dir
        self.result_dir = result_dir
        self.result_meta_dict = self.load_meta()


    def get_instance_meta(self, id):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        return self.result_meta_dict[id]
    
    def load_meta(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        # 加载结果的元信息
        meta_path = p_join(self.intermediate_dir, \
            self.workload, "experiment_obj", "result_meta.json")
        meta_dict = utils.load_json(data_path=meta_path)

        return meta_dict



    def load_object(self, obj_path, with_path = False):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        out_path = obj_path
        result_obj = utils.load_pickle(data_path=obj_path)

        if result_obj is None:
            # 尝试新的路径
            obj_dir = os.path.dirname(obj_path)
            obj_name = os.path.basename(obj_path)
            new_path = p_join(obj_dir, "history_pickle", obj_name)
            out_path = new_path
            result_obj = utils.load_pickle(data_path=new_path)
            assert result_obj is not None, f"construct_instance: new_path = {new_path}"

        if with_path:
            return result_obj, out_path
        else:
            return result_obj
    

    def verify_plan_cost(self, query_text, query_meta, card_dict, result):
        """
        {Description}
    
        Args:
            query_text:
            query_meta
            card_dict:
            result:
        Returns:
            flag_true:
            flag_est:
        """
        subquery_true, single_table_true, _, _ = utils.extract_card_info(card_dict)
        true_physical, estimation_physical = self.result_verifier.\
            construct_physical_plans(query_text, query_meta, card_dict)
        
        cost_true_new = true_physical.get_plan_cost(subquery_true, single_table_true)
        cost_est_new = estimation_physical.get_plan_cost(subquery_true, single_table_true)
        p_error, cost_est_old, cost_true_old = result

        flag_true = np.abs(cost_true_new - cost_true_old) < 10
        flag_est = np.abs(cost_est_new - cost_est_old) < 10

        if flag_true == False or flag_est == False:
            print(f"verify_plan_cost: cost_true_new = {cost_true_new: .2f}. cost_true_old = {cost_true_old: .2f}. "\
                f"cost_est_new = {cost_est_new: .2f}. cost_est_old = {cost_est_old: .2f}.")

        return flag_true, flag_est

    def filter_cost_error_cases(self, instance_list: list):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        result_list = []
        for idx, data_tuple in enumerate(instance_list):
            if len(data_tuple) == 4:
                query, meta, result, card_dict = data_tuple
            elif len(data_tuple) == 5:
                query, meta, result, card_dict, time_info = data_tuple
            else:
                raise ValueError(f"filter_invalid_cases: len(data_tuple) = {len(data_tuple)}.")
            
            flag_true, flag_est = self.verify_plan_cost(query, meta, card_dict, result)
            result_list.append((flag_true, flag_est))

        return result_list
    
    def verify_card_top(self, card_dict: dict):
        """
        验证sub-expression中的基数是否包含0
    
        Args:
            card_dict:
            arg2:
        Returns:
            return1:
            return2:
        """
        subquery_true, single_table_true, _, _ = utils.extract_card_info(card_dict)
        subquery_values = list(subquery_true.values())

        if 0 in subquery_values:
            # print(f"verify_card_top: subquery_values = {subquery_values}.")
            return False
        else:
            return True
        

    def verify_card_estimation(self, query_text, card_dict, ce_type):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            flag:
        """
        subquery_estimation, single_table_estimation = \
            card_dict['estimation']['subquery'], card_dict['estimation']['single_table']
        
        self.result_verifier.reset_ce_handler(ce_type)
        query_parser = workload_parser.SQLParser(query_text, self.workload)
        # self.result_verifier.set_instance()
        # 验证技术估计
        self.result_verifier.query_ctrl.set_query_instance(query_text, query_parser.generate_meta())
        estimation_error_list, _ = self.result_verifier.verify_cardinalities(\
            query_parser, {}, {}, subquery_estimation, single_table_estimation, False, True)
        
        return len(estimation_error_list) == 0

    def verify_card_complete(self, query_meta, card_dict):
        """
        {Description}
        
        Args:
            query_meta:
            card_dict:
        Returns:
            flag:
        """
        subquery_true, single_table_true, subquery_estimation,\
            single_table_estimation = utils.extract_card_info(card_dict)
        missing_list = self.result_verifier.verify_card_complete(query_meta, \
            subquery_true, single_table_true, subquery_estimation, single_table_estimation)
        
        return len(missing_list) == 0

    def filter_invalid_cases(self, instance_list: list, card_complete = True, \
        card_estimation = False, card_top = True, ce_type = "internal", return_index = False):
        """
        {Description}
    
        Args:
            arg1:
            return_index: 是否返回索引
        Returns:
            return1:
            return2:
        """
        instance_out = []
        valid_index = []
        for idx, data_tuple in enumerate(instance_list):
            if len(data_tuple) == 4:
                query, meta, result, card_dict = data_tuple
            elif len(data_tuple) == 5:
                query, meta, result, card_dict, time_info = data_tuple
            else:
                raise ValueError(f"filter_invalid_cases: len(data_tuple) = {len(data_tuple)}.")
            
            # if idx == 0:
            #     print(f"filter_invalid_cases: query = {query}")
            #     print(f"filter_invalid_cases: meta = {meta}")
            #     print(f"filter_invalid_cases: result = {result}")
            #     print(f"filter_invalid_cases: card_dict = {card_dict}")

            flag = True
            if card_complete == True and flag == True:
                flag = flag and self.verify_card_complete(meta, card_dict)
            if card_estimation == True and flag == True:
                flag = flag and self.verify_card_estimation(query, card_dict, ce_type)
            if card_top == True and flag == True:
                flag = flag and self.verify_card_top(card_dict)

            if flag:
                instance_out.append(data_tuple)
                valid_index.append(idx)
            else:
                # print(f"filter_invalid_cases: result = {result}.")
                pass

        if return_index:
            return instance_out, valid_index
        else:
            return instance_out

    def filter_wrap_func(self, output_tuple, card_complete = True, \
            card_estimation = False, card_top = False, ce_type = "internal", return_index = False):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        instance_list = list(zip(*output_tuple))
        print(f"filter_wrap_func: len(output_tuple) = {len(output_tuple)}. len(instance_list) = {len(instance_list)}.")
        func_res = self.filter_invalid_cases(instance_list, \
            card_complete, card_estimation, card_top, ce_type, return_index)
        
        if return_index == True:
            instance_out, valid_index = func_res
        else:
            instance_out = func_res

        output_filter = list(zip(*instance_out))
        print(f"filter_wrap_func: len(output_filter) = {len(output_filter)}. len(instance_out) = {len(instance_out)}")

        if return_index == True:
            return output_filter, valid_index
        else:
            return output_filter

    def update_instance(self, config: dict, verify_complete = False, verify_estimation = False):
        """
        更正结果实例，由于
    
        Args:
            config:
            verify_complete:
            verify_estimation:
        Returns:
            return1:
            return2:
        """
        result_list = config["result_list"]
        
        for result_id in result_list:
            instance_meta = self.result_meta_dict[result_id]
            obj_path = instance_meta['obj_path']

            result_obj, result_path = self.load_object(obj_path, with_path=True)

            if len(result_obj) == 5:
                # 有time信息的情况，直接删除
                result_obj, time_info = result_obj[:4], result_obj[4]
                
            estimator_name = instance_meta['estimation_method']
            result_obj = self.filter_wrap_func(result_obj, \
                verify_complete, verify_estimation, estimator_name)
            
            # 
            query_list, meta_list, result_list, card_dict_list = result_obj
            assert len(query_list) == len(meta_list) == len(card_dict_list), \
                f"update_instance: len(query_list) = {len(query_list)}. len(meta_list) = {len(meta_list)}. len(card_dict_list) = {len(card_dict_list)}."
            
            result_new = []
            for query_text, query_meta, card_dict in zip(query_list, meta_list, card_dict_list):
                analyzer = case_analysis.CaseAnalyzer(query_text, \
                    query_meta, (1.0, 10.0, 10.0), card_dict, self.workload)
                result_new.append((analyzer.p_error, analyzer.estimation_cost, analyzer.true_cost))

            if len(result_obj) == 5:
                new_obj = (query_list, meta_list, result_new, card_dict_list, time_info)
            else:
                new_obj = (query_list, meta_list, result_new, card_dict_list)

            utils.dump_pickle(new_obj, result_path)


# %%
