#!/usr/bin/env python
# -*- coding: utf-8 -*-


# %%
import numpy as np
from comparison import external_memory

# %%

def max_p_error(val_dict: dict):
    """
    {Description}

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    res_dict = {}
    for k, val_list in val_dict.items():
        if isinstance(val_list[0], (list, tuple)):
            try:
                res_dict[k] = [np.max(local_list) if len(local_list) > 0 \
                    else 0.0 for local_list in val_list]
            except Exception as e:
                print(f"max_p_error: meet ValueError. val_list = {val_list}.")
                raise e
        else:
            res_dict[k] = np.max(val_list)
    return res_dict


def quantile_p_error(ratio: float, val_dict: dict):
    """
    {Description}

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    res_dict = {}
    for k, val_list in val_dict.items():
        if isinstance(val_list[0], (list, tuple)):
            try:
                res_dict[k] = [np.quantile(local_list, ratio) if len(local_list) > 0 \
                    else 0.0 for local_list in val_list]
            except Exception as e:
                print(f"max_p_error: meet ValueError. val_list = {val_list}.")
                raise e
        else:
            res_dict[k] = np.quantile(val_list, ratio)
    return res_dict


def ninetieth_p_error(val_dict: dict):
    """
    {Description}

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    res_dict = {}
    for k, val_list in val_dict.items():
        if isinstance(val_list[0], (list, tuple)):
            try:
                res_dict[k] = [np.quantile(local_list, 0.9) if len(local_list) > 0 \
                    else 0.0 for local_list in val_list]
            except Exception as e:
                print(f"max_p_error: meet ValueError. val_list = {val_list}.")
                raise e
        else:
            res_dict[k] = np.quantile(val_list, 0.9)
    return res_dict


def topk_p_error(kth, val_dict: dict):
    """
    {Description}

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    res_dict = {}
    # local_func = lambda in_list: sorted(in_list, reverse=True)[-1] if \
    #     len(in_list) <= kth else sorted(in_list, reverse=True)[kth]
    local_func = lambda in_list: 0.0 if len(in_list) == 0 else \
        (sorted(in_list, reverse=True)[-1] if len(in_list) <= kth else \
            sorted(in_list, reverse=True)[kth])

    for k, val_list in val_dict.items():
        if isinstance(val_list[0], (tuple, list)):
            res_dict[k] = [local_func(i) for i in val_list]
        else:
            res_dict[k] = local_func(val_list)

    return res_dict

def median_p_error(val_dict: dict):
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    res_dict = {}
    for k, val_list in val_dict.items():
        if isinstance(val_list[0], (list, tuple)):
            try:
                res_dict[k] = [np.median(local_list, axis=-1) if len(local_list) > 0 \
                    else 0.0 for local_list in val_list]
            except Exception as e:
                print(f"median_p_error: meet ValueError. val_list = {val_list}.")
                raise e
        else:
            res_dict[k] = np.median(val_list, axis=-1)
    return res_dict

def cumulative_workload_p_error(threshold, val_dict: dict):
    """
    {Description}

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    cost_pair_dict = val_dict
    res_dict = {}
    
    def local_func(pair_local):
        if len(pair_local) == 0:
            return 0.0
        p_error_list = [i[0] / (i[1] + 1.1) for i in pair_local]
        cumulative_actual_cost, cumulative_expected_cost = 0, 0
        sorted_idx = np.argsort(p_error_list)[::-1]

        for idx in sorted_idx:
            cumulative_actual_cost += pair_local[idx][0]
            cumulative_expected_cost += pair_local[idx][1]
            if cumulative_expected_cost >= threshold:
                break
        
        return cumulative_actual_cost / cumulative_expected_cost
    
    for k, pair_list in cost_pair_dict.items():
        if isinstance(pair_list[0][0], tuple):
            res_dict[k] = [local_func(pair_local) for pair_local in pair_list]
        else:
            res_dict[k] = local_func(pair_list)

    return res_dict

def number_of_notable_instances(error_threshold, val_dict: dict):
    """
    {Description}

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    if error_threshold <= 1.0:
        # 2024-03-26: 表示为分位数，从外部存储中计算得出
        error_quantile = error_threshold
        error_threshold = np.quantile(\
            external_memory.current_error_list, error_quantile)
        print(f"number_of_notable_instances: config = {external_memory.current_config}. "\
              f"quantile = {error_quantile:.2f}. threshold = {error_threshold:.2f}.")
    
    res_dict = {}
    for k, val_list in val_dict.items():
        try:
            if isinstance(val_list[0], (list, tuple)) == False:
                res_dict[k] = int(np.sum(val_list > error_threshold))
            else:
                res_dict[k] = [int(np.sum(local_list > error_threshold)) if len(local_list) > 0 \
                    else 0.0 for local_list in val_list]
        except Exception as e:
            print(f"number_of_notable_instances: meet ValueError. k = {k}. v = {val_list}. ")
            raise e

    return res_dict


# %%
