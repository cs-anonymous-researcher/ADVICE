#!/usr/bin/env python
# -*- coding: utf-8 -*-

# %%
from copy import deepcopy
from utilities import utils
import numpy as np
from collections import defaultdict


# %%

def card_dict_mask_along_table(workload: str, in_card_dict: dict, mask_num_info, 
        target_table = None, mask_single_table = False):
    """
    {Description}

    Args:
        in_card_dict:
        mask_num_info: 
        target_table: 
    Returns:
        out_card_dict:
        subquery_missing:
        single_table_missing:
        target_table: 
    """
    alias_inverse = utils.workload_alias_option[workload]
    alias_mapping = {v: k for k, v in alias_inverse.items()}

    subquery_true, single_table_true, subquery_est, single_table_est = \
        utils.extract_card_info(in_card_dict, dict_copy = True)
    
    assert len(subquery_true) == len(subquery_est), \
        f"card_dict_mask_along_table: len(subquery_true) = {len(subquery_true)}. len(subquery_est) = {len(subquery_est)}."
    assert len(single_table_true) == len(single_table_est), \
        f"card_dict_mask_along_table: len(single_table_true) = {len(single_table_true)}. len(subquery_est) = {len(single_table_est)}."
    
    if target_table is not None:
        target_alias = alias_inverse[target_table]
    else:
        try:
            target_alias = np.random.choice(list(single_table_est.keys()))
        except ValueError as e:
            print(f"card_dict_mask_along_table: single_table_est.keys = {list(single_table_est.keys())}")
            raise e
        
    target_table = alias_mapping[target_alias]
    candidate_dict = defaultdict(list)
    # target_su

    for tuple_key in subquery_true.keys():
        if target_alias in tuple_key:
            candidate_dict[len(tuple_key)].append(tuple_key)

    mask_num_total = sum([len(v) for v in candidate_dict.values()])

    if mask_num_info == "all":
        # 屏蔽alias所有的subquery
        left_num = mask_num_total
    elif isinstance(mask_num_info, int) and mask_num_info >= 1:
        # 按数目屏蔽subquery
        left_num = min(mask_num_info, mask_num_total)
    elif isinstance(mask_num_info, float) and 0.0 <= mask_num_info <= 1.0:
        # 按比例屏蔽subquery
        left_num = max(1, int(mask_num_info * mask_num_total))
    else:
        raise ValueError(f"card_dict_mask_along_table: mask_num_info = {mask_num_info}.")
    
    subquery_missing, single_table_missing = [], []

    for alias_num in sorted(candidate_dict.keys(), reverse=True):
        alias_tuple_list = candidate_dict[alias_num]
        if left_num <= len(candidate_dict[alias_num]):
            selected_idx_list = np.random.choice(range(len(alias_tuple_list)), left_num)
            selected_local = utils.list_index(alias_tuple_list, selected_idx_list)
            subquery_missing.extend(selected_local)
            break
        else:
            selected_local = alias_tuple_list
            subquery_missing.extend(selected_local)

    if mask_single_table == True:
        single_table_missing.append(target_alias)

    for k in subquery_missing:
        subquery_true[k] = None

    for k in single_table_missing:
        single_table_true[k] = None

    out_card_dict = utils.pack_card_info(subquery_true, 
        single_table_true, subquery_est, single_table_est)

    return out_card_dict, subquery_missing, single_table_missing, target_table


def card_dict_mask(in_card_dict, mask_num):
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    out_card_dict = {}
    out_card_dict['estimation'] = deepcopy(in_card_dict['estimation'])
    out_card_dict['true'] = {
        "subquery": {},
        "single_table": deepcopy(in_card_dict['true']['single_table'])
    }
    mask_key_set = set(sorted(in_card_dict['true']['subquery'].keys(), \
        key=lambda a:len(a), reverse=True)[:mask_num])
    # print(f"card_dict_mask: mask_num = {mask_num}. mask_key_set = {mask_key_set}.")

    for k, v in in_card_dict['true']['subquery'].items():
        if k in mask_key_set:
            out_card_dict['true']['subquery'][k] = None
        else:
            out_card_dict['true']['subquery'][k] = v

    return out_card_dict




def card_dict_list_mask(in_card_dict_list, mask_num):
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    out_card_dict_list = [card_dict_mask(card_dict, \
        mask_num) for card_dict in in_card_dict_list]
    return out_card_dict_list

# %%
