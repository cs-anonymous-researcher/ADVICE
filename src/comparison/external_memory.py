#!/usr/bin/env python
# -*- coding: utf-8 -*-

# %%
from collections import defaultdict
# %%

result_error_dict = defaultdict(list)
current_error_list = []
current_config = "", ""

def set_current_list(workload, ce_method):
    """
    {Description}

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    global current_config
    global current_error_list
    current_config = workload, ce_method
    current_error_list = result_error_dict[(workload, ce_method)]


def func_name(self,):
    """
    {Description}

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    pass
# %%

