#!/usr/bin/env python
# -*- coding: utf-8 -*-


# %%
import numpy as np
from utilities import utils
from collections import defaultdict
from sklearn.neighbors import KernelDensity


# %%


equal_diff_strategy = {
    "option": "equal_diff",
    "start": 0.5,
    "end": 2,
    "number": 4
}

equal_ratio_strategy = {
    "option": "equal_ratio",
    "start": 0.5,
    "end": 2,
    "number": 4
}

graph_strategy_dict = {
    "kde": {
        "mode": "kde"
    },
    "max": {
        "mode": "max"
    }
}

# %%

class BaseEstimator(object):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, workload):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        self.workload = workload
        self.query_text = ""
        self.query_meta = (), ()
        self.subquery_true, self.single_table_true = {}, {}
        self.subquery_estimation, self.single_table_estimation = {}, {}

    def set_instance(self, query_text, query_meta):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        self.query_text = query_text
        self.query_meta = query_meta

    def set_existing_card_dict(self, subquery_true, single_table_true, \
                subquery_estimation, single_table_estimation):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        self.subquery_true, self.single_table_true = subquery_true, single_table_true
        self.subquery_estimation, self.single_table_estimation = \
            subquery_estimation, single_table_estimation

    def make_value_sampling(self, subquery_missing, single_table_missing):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        raise NotImplementedError("BaseEstimator: make_value_sampling has not been implemented!")

    def result_validation(self, subquery_missing: list, single_table_missing: list, \
                          subquery_res: dict, single_table_res: dict):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        assert set(subquery_missing) == set(subquery_res.keys()), \
            f"result_validation: subquery_missing = {subquery_missing}. subquery_res = {subquery_res.keys()}."
        assert set(single_table_missing) == set(single_table_res.keys()), \
            f"result_validation: single_table_missing = {single_table_missing}. single_table_res = {single_table_res.keys()}."

        return True
    
# %%

class BuiltinEstimator(BaseEstimator):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, workload, strategy):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        super().__init__(workload=workload)
        self.strategy = strategy

    def make_value_sampling(self, subquery_missing, single_table_missing):
        """
        {Description}

        Args:
            subquery_missing: key_list
            single_table_missing: key_list
        Returns:
            subquery_candidates: card_dict
            single_table_candidates: card_dict
        """
        return self.value_sampling(subquery_missing, single_table_missing, self.strategy)

    # @utils.timing_decorator
    def value_sampling(self, subquery_key_list, single_table_key_list, strategy: dict):
        """
        真实值的采样，目前是每个独立的，之后考虑加上关联性

        Args:
            subquery_key:
            single_table_key:
            strategy: 采样的策略
        Returns:
            subquery_candidates:
            single_table_candidates:
        """
        subquery_candidates = defaultdict(list)
        single_table_candidates = defaultdict(list)

        for k in single_table_key_list:
            single_table_candidates[k] = self.infer_single_table(\
                single_table_key=k, strategy=strategy)
            
        for k in subquery_key_list:
            subquery_candidates[k] = self.infer_subquery(\
                subquery_key=k, strategy=strategy)

        return subquery_candidates, single_table_candidates
    

    # @utils.timing_decorator
    def equal_diff_generation(self, start, end, num):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        value_list = np.linspace(start=start, stop=end, endpoint=True, num=num)
        return [int(np.ceil(val)) for val in value_list]

    # @utils.timing_decorator
    def equal_ratio_generation(self, start, end, ref_point, num):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        value_list = []
        if num % 2 == 0:
            one_side_num = num // 2
        else:
            one_side_num = (num - 1) // 2

        left_factor, right_factor = (end / ref_point), (ref_point / start)
        value_left, value_right = [], []

        left_val, right_val = ref_point, ref_point
        for _ in range(one_side_num):
            left_val /= left_factor
            right_val *= right_factor

            value_left.append(left_val)
            value_right.append(right_val)
        
        value_left.reverse()
        if num % 2 == 0:
            value_list = value_left + value_right
        else:
            value_list = value_left + [ref_point, ] + value_right

        return [int(np.ceil(val)) for val in value_list]


    # @utils.timing_decorator
    def infer_subquery(self, subquery_key, strategy:dict):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        estimation_value = self.subquery_estimation[subquery_key]
        return self.infer_one_value(estimation_value, strategy)


    def infer_single_table(self, single_table_key, strategy:dict):
        """
        {Description}
        
        Args:
            single_table_key:
            estimation_value:
        Returns:
            res1:
            res2:
        """
        estimation_value = self.single_table_estimation[single_table_key]
        return self.infer_one_value(estimation_value, strategy)
    
    def infer_one_value(self, in_val, strategy:dict):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        estimation_value = in_val
        value_list = []

        if strategy['option'] == "dummy":
            value_list = [estimation_value,]
        elif strategy['option'] == "equal_diff":
            # print("infer_single_table: strategy = {}.".format(strategy))
            start_val, end_val = estimation_value * strategy['start'], estimation_value * strategy['end']
            value_list = self.equal_diff_generation(start=start_val, end=end_val, num=strategy['number'])
        elif strategy['option'] == "equal_ratio":
            start_val, end_val = estimation_value * strategy['start'], estimation_value * strategy['end']
            ref_point = estimation_value
            value_list = self.equal_ratio_generation(start=start_val, end=end_val, ref_point=ref_point, num=strategy['number'])
        else:
            raise ValueError("infer_single_table: Unsupported strategy option ({}).".\
                             format(strategy['option']))
        
        return value_list

    def get_error_list(self, key_list):
        """
        {Description}
        
        Args:
            key_list:
            arg2:
        Returns:
            error_list:
            res2:
        """
        error_list = []
        for k in key_list:
            if isinstance(k, str):
                error_list.append(self.single_table_error[k])
            elif isinstance(k, tuple):
                error_list.append(self.subquery_error[k])
        return error_list
    

# %%

class GraphCorrBasedEstimator(BaseEstimator):
    """
    基于图关联性的基数估计器

    Members:
        field1:
        field2:
    """

    def __init__(self, workload, strategy):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        super().__init__(workload=workload)
        self.strategy = strategy
        # print(f"GraphCorrBasedEstimator: strategy = {strategy}")
        self.subquery_error, self.single_table_error = {}, {}

    def construct_error_dict(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        subquery_true, single_table_true = \
            self.subquery_true, self.single_table_true
        subquery_estimation, single_table_estimation = \
            self.subquery_estimation, self.single_table_estimation
        
        def convert_func(true_card, est_card):
            return (1.0 + true_card) / (1.0 + est_card)
        
        subquery_error = {k: convert_func(subquery_true[k], \
                        subquery_estimation[k]) for k in subquery_true}
        single_table_error = {k: convert_func(single_table_true[k], \
                        single_table_estimation[k]) for k in single_table_true}
        
        # 检查subquery_error的合法性
        def eval_dict(in_dict: dict, dict_name: str = ""):
            invalid_num = 0
            for k, v in in_dict.items():
                if v < 0:
                    print("construct_error_dict.eval_dict: name = {}. k = {}. v = {}.".\
                          format(dict_name, k, v))
                    invalid_num += 1
                    in_dict[k] = 1
            return invalid_num
        
        invalid_1 = eval_dict(subquery_true, "subquery_true")
        invalid_2 = eval_dict(single_table_true, "single_table_true")
        invalid_3 = eval_dict(subquery_estimation, "subquery_estimation")
        invalid_4 = eval_dict(single_table_estimation, "single_table_estimation")
        
        if sum([invalid_1, invalid_2, invalid_3, invalid_4]) > 0:
            # raise ValueError(f"construct_error_dict: invalid_1 = {invalid_1}. invalid_2 = {invalid_2}."\
            #                  f"invalid_3 = {invalid_3}. invalid_4 = {invalid_4}.")
            raise RuntimeWarning(f"construct_error_dict: invalid_1 = {invalid_1}. invalid_2 = {invalid_2}."\
                             f"invalid_3 = {invalid_3}. invalid_4 = {invalid_4}.")
        
        self.subquery_error, self.single_table_error = \
            subquery_error, single_table_error
        return subquery_error, single_table_error
    
    def get_error_list(self, in_key_list):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        error_list = []
        for in_key in in_key_list:
            if isinstance(in_key, str):
                error_list.append(self.single_table_error[in_key])
            else:
                error_list.append(self.subquery_error[in_key])
        
        # 去除error_list包含0的问题
        # 
        tolerance = 1e-8
        error_list = np.where(np.isclose(error_list, 0.0, \
            atol=tolerance), 1.0, error_list)
        
        return error_list

    def restore_value_list(self, error_list, est_value):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        def restore_func(error, est_card):
            return error * est_card
        
        return [restore_func(error, est_value) for error in error_list]

    def make_value_sampling(self, subquery_missing: list, single_table_missing: list):
        """
        {Description}

        Args:
            subquery_missing: key_list
            single_table_missing: key_list
        Returns:
            subquery_candidates: card_dict
            single_table_candidates: card_dict
        """
        self.construct_error_dict()
        subquery_candidates, single_table_candidates = {}, {}

        # print(f"make_value_sampling: subquery_missing = {subquery_missing}. single_table_missing = {single_table_missing}.")

        for tuple_key in subquery_missing:
            value_list = self.infer_subquery(in_key=tuple_key, strategy=self.strategy)
            subquery_candidates[tuple_key] = value_list

        for alias_key in single_table_missing:
            value_list = self.infer_single_table(in_key=alias_key, strategy=self.strategy)
            single_table_candidates[alias_key] = value_list

        return subquery_candidates, single_table_candidates
    

    def kernal_density_estimation(self, val_list, sample_num = 5):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        try:
            ln_val_list = np.log(val_list).reshape(-1, 1)
            # bandwidth参数考虑后续调优
            kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(ln_val_list)
        except ValueError as e:
            print(f"kernal_density_estimation: val_list = {val_list}.")
            raise e
        except RuntimeWarning as w:
            print("w")
            raise ValueError(f"kernal_density_estimation: val_list = {val_list}.")

        ln_sampled = kde.sample(sample_num)
        # print(f"kernal_density_estimation: ln_sampled.shape = {ln_sampled.shape}.")
        ln_sampled_flatten = ln_sampled.flatten()
        out_list = list(np.exp(ln_sampled_flatten))
        return out_list


    def infer_single_table(self, in_key: str, strategy: dict):
        """
        推断单表的基数

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        est_val = self.single_table_estimation[in_key]
        if strategy['mode'] == "direct":
            return [est_val,]
        elif strategy['mode'] == "kde":
            error_list = self.kernal_density_estimation(val_list=[1.0,])
            return self.restore_value_list(error_list, est_val)
        elif strategy['mode'] == "max":
            return [est_val,]

    def infer_subquery(self, in_key: tuple, strategy: dict):
        """
        推断子查询的基数
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        est_val = self.subquery_estimation[in_key]
        key_list = self.get_correlated_nodes(in_key=in_key)
        
        if strategy['mode'] == "direct":
            # 直接估计
            error_list = self.get_error_list(key_list)
        elif strategy['mode'] == "kde":
            # 基于KDE估计
            error_ref = self.get_error_list(key_list)
            # print(f"infer_subquery: kde mode. key_list = {key_list}. error_ref = {error_ref}.")
            if len(error_ref) == 0:
                error_ref = [1.0]
            error_list = self.kernal_density_estimation(error_ref)
        elif strategy['mode'] == "max":
            error_list = self.get_error_list(key_list)
            if len(error_list) == 0:
                error_list = [1.0, ]
            else:
                error_list = [max(error_list), ]
        else:
            raise ValueError("infer_subquery: mode = {}. valid_list = []")
        
        true_val_list = self.restore_value_list(error_list, est_val)
        return true_val_list


    def get_correlated_nodes(self, in_key: tuple, curr_delta = 1):
        """
        获得相关联的节点
        
        Args:
            in_key:
            arg2:
        Returns:
            node_key_list:
            res2:
        """
        # 查找相关的节点
        node_key_list = []
        for k in self.subquery_true.keys():
            # 判断tuple之间的关系
            if utils.tuple_in(in_key, k) == True and len(in_key) - len(k) <= curr_delta:
                node_key_list.append(k)

        for k in self.single_table_true.keys():
            # 判断tuple之间的关系
            if utils.tuple_in(in_key, (k,)) == True and len(in_key) - 1 <= curr_delta:
                node_key_list.append(k)

        return node_key_list
    

# %%
