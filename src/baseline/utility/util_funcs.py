#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from os.path import join as p_join
from matplotlib import pyplot as plt
import json, os, time, re, math
import pickle
from copy import deepcopy

def dict2list(in_dict: dict):
    """
    {Description}
    
    Args:
        in_dict:
    Returns:
        key_list:
        value_list:
    """
    print("utils.dict2list: in_dict = {}.".format(in_dict.items()))

    pair_list = [(k, v) for k, v in in_dict.items()]
    key_list, value_list = zip(*pair_list)
    return key_list, value_list


def list2dict(key_list:list, value_list:list):
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

    for k, v in zip(key_list, value_list):
        res_dict[k] = v

    return res_dict


def prob_list_resize(in_prob_list):
    """
    {Description}
    
    Args:
        in_prob_list:
        arg2:
    Returns:
        out_prob_list:
        res2:
    """
    expected_sum = 1.0
    actual_sum = np.sum(in_prob_list)
    resize_factor = expected_sum / actual_sum

    out_prob_list = [i * resize_factor for i in in_prob_list]
    return out_prob_list
    
def prob_dict_resize(in_prob_dict):
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    out_prob_dict = {}
    key_list, value_list = dict2list(in_dict=in_prob_dict)
    value_list = prob_list_resize(in_prob_list=value_list)
    for k, v in zip(key_list, value_list):
        out_prob_dict[k] = v

    return out_prob_dict