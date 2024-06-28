#!/usr/bin/env python
# -*- coding: utf-8 -*-


# %%

import numpy as np
import pandas as pd
from os.path import join as p_join
from matplotlib import pyplot as plt
import json, os, time, re, math
import pickle
# %%

from query import query_exploration
from utility import generator
from data_interaction import data_management, postgres_connector

# %%
def parse_join_order(leading_str):
    """
    {Description}
    
    Args:
        leading_str:
    Returns:
        is_bushy:
        level_stack:
    """
    leading_str = leading_str[8:-1]
    level_dict = {}

    def find_next_bracket(curr_str: str):
        next_left_bracket = curr_str.find('(') if \
            curr_str.find('(') != -1 else 1000
        next_right_bracket = curr_str.find(')') if \
            curr_str.find(')') != -1 else 1000

        if next_left_bracket < next_right_bracket:
            return "(", next_left_bracket
        elif next_left_bracket > next_right_bracket:
            return ")", next_right_bracket

    def parse_content(in_str):
        in_str = in_str.strip()
        if " " in in_str:
            return tuple(in_str.split(" "))
        else:
            return in_str

    curr_str = leading_str
    op_stack, value_stack, curr_level = [], [], 0

    while True:
        next_pos = 0
        if curr_str[0] == "(":
            next_pos = 1
            curr_level += 1
        elif curr_str[0] == ")":
            next_pos = 1
            val = value_stack.pop()
            if val == None:
                continue
            if isinstance(val, (tuple, list)):
                for v in val:
                    level_dict[v] = curr_level
            elif isinstance(val, str):
                level_dict[val] = curr_level
            else:
                raise TypeError("Unsupport val type: {}".format(val))
            curr_level -= 1
        else:
            op, next_pos = find_next_bracket(curr_str)
            # print(op, next_pos)
            value_stack.append(parse_content(curr_str[:next_pos]))
        curr_str = curr_str[next_pos:]
        if len(curr_str) == 0:
            break
    return level_dict



def level2order(level_dict):
    """
    解析level_dict，转化成连接顺序
    
    Args:
        level_dict:
    Returns:
        is_bushy: 是否是灌木丛形状的
        join_order: 连接顺序
    """
    level_list = []
    max_level = 0

    for k, v in level_dict.items():
        level_list.append((k, v))

    level_list = sorted(level_list, key = lambda a: a[1], reverse = True)
    join_order = []
    max_level = level_list[0][1]
    # for 
    print("max_level = {}. level_list = {}.".format(max_level, level_list))
    if max_level == len(level_list) - 1:
        is_bushy = False
    else:
        is_bushy = True
    return is_bushy, list(zip(*level_list))[0]

# %%

class ZigZagJoinOrder(object):
    """
    {Description}

    Members:
        start_pair:
        follow_up_seq:
    """

    def __init__(self, join_order):
        """
        {Description}

        Args:
            join_order:
        """
        self.start_pair = set(join_order[:2])
        self.follow_up_seq = join_order[2:]

    def __eq__(self, other):
        """
        {Description}

        Args:
            other:
        Returns:
            flag:
        """
        return other.start_pair == self.start_pair and \
            self.follow_up_seq == other.follow_up_seq
    
    def __sub__(self, other):
        """
        {Description}
        
        Args:
            other:
        Returns:
            left_seq:
        """
        return []

    def __str__(self,):
        """
        {Description}

        Args:
            None
        Returns:
            join_order_str:
        """
        return "{}, {}".format(\
            self.start_pair, self.follow_up_seq)

# %%
import graphviz
from collections import defaultdict
from copy import deepcopy

class TreeNode(object):
    """
    {Description}

    Members:
        parent:
        children:
    """

    def __init__(self, parent, current_path):
        """
        {Description}

        Args:
            parent:
            current_path:
        """
        self.children = {}
        self.parent = parent
        self.current_path = current_path

    def construct_subtree(self, join_order_batch):
        """
        {Description}

        Args:
            join_order_batch:
        Returns:
            return1:
            return2:
        """
        prefix_dict = defaultdict(list)
        for join_order in join_order_batch:
            prefix_dict[join_order[0]].append(join_order[1:])

        for k, v in prefix_dict.items():
            new_path = deepcopy(self.current_path)
            new_path.append(k)

            new_node = TreeNode(parent = self, current_path = new_path)
            new_node.construct_subtree(v)
            self.children[k] = new_node

        return self

    def build_prob_table(self, ):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        prob_table = {}
        return prob_table


class PriorTree(object):
    """
    先验树，估计节点探索的成功率

    Members:
        field1:
        field2:
    """

    def __init__(self,):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        self.root = TreeNode(parent = None, current_path = ())
        self.join_order_list = []

    def load_data_batch(self, external_list):
        """
        {Description}
        
        Args:
            external_list:
        Returns:
            join_order_list:
        """
        self.join_order_list.extend(external_list)
        return self.join_order_list

    def load_data(self, join_order):
        """
        {Description}

        Args:
            join_order:
        Returns:
            join_order_list:
        """
        self.join_order_list.append(join_order)
        return self.join_order_list

    def display(self,):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        # graph = graphviz.Digraph()
        # graph.node(name)
        # graph.edge(tail_name, head_name)
        pass