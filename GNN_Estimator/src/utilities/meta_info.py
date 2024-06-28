#!/usr/bin/env python
# -*- coding: utf-8 -*-


# %%

def generate_subquery_meta(query_meta, alias_list, alias_mapping):
    """
    {Description}

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    if isinstance(alias_list, str):
        alias_list = [alias_list,]
        
    schema_sublist = []
    filter_sublist = []
    # print(f"generate_subquery_meta: alias_mapping = {alias_mapping}.")
    
    for s in query_meta[0]:
        if alias_mapping[s] in alias_list:
            schema_sublist.append(s)

    for item in query_meta[1]:
        if item[0] in alias_list:
            filter_sublist.append(item)

    out_meta = schema_sublist, filter_sublist
    return out_meta


# %%
