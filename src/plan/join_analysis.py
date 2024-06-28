#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from os.path import join as p_join
from matplotlib import pyplot as plt
import json, os, time, re, math
import pickle
from copy import deepcopy

import graphviz

class JoinOrderAnalyzer(object):
    """
    连接顺序的分析器

    Members:
        field1:
        field2:
    """

    def __init__(self, join_order_str: str):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        # print("JoinOrderAnalyzer.__init__: join_order_str = {}.".format(join_order_str))

        if join_order_str.startswith("Leading"):
            join_order_str = join_order_str[len("Leading"):]

        self.table_num = 0
        self.join_order_str = join_order_str
        self.join_tree = self.parse_join_order_str(join_order_str)
        self.eval_levels()

    def top_table_eval(self, target_alias):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        # print(f"top_table_eval: join_tree = {self.join_tree}.")
        # print(f"top_table_eval: tree_left = {self.join_tree['left']}. tree_right = {self.join_tree['right']}.")

        if self.join_tree['left'] == target_alias or \
            self.join_tree['right'] == target_alias:
            return True
        else:
            return False

    def get_leading_tables(self, table_num = None):
        """
        {Description}
    
        Args:
            table_num: 表的数目
            arg2:
        Returns:
            table_subset: 表的子集，并且是排好序的结果
            return2:
        """
        # 
        if table_num is None:
            table_num = self.table_num

        table_subset = []
        for k, v in self.level_dict.items():
            if v < table_num:
                table_subset.append(k)

        return sorted(table_subset)
    
    def get_leading_order(self, table_num = None):
        """
        获得leading table的join顺序，不同于get_leading_tables函数，prefix的顺序得以保留
    
        Args:
            table_num: 表的数目
            arg2:
        Returns:
            table_subset: 表的子集，并且是排好序的结果
            return2:
        """
        # 
        if table_num is None:
            table_num = self.table_num

        table_subset = []
        for k, v in self.level_dict.items():
            if v < table_num:
                table_subset.append((k, v))

        # return sorted(table_subset)
        table_subset.sort(key=lambda a: a[1])
        return [item[0] for item in table_subset]

    def visualization(self, output_path = ""):
        """
        可视化join_order
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        global_id = 0
        dot = graphviz.Digraph("join tree")
        dot.attr("node", shape="box")
        def visit(node):
            nonlocal global_id
            global_id += 1
            current_id = str(global_id)

            if isinstance(node, dict):
                dot.node(str(global_id), "⋈")   # join的运算符
                left_child_id = visit(node=node['left'])
                right_child_id = visit(node=node['right'])
                dot.edge(current_id, left_child_id)
                dot.edge(current_id, right_child_id)

            elif isinstance(node, str):
                dot.node(str(global_id), str(node))

            return current_id
        
        visit(self.join_tree)
        return dot

    def parse_join_order_str(self, in_str):
        """
        解析join order的字符串

        Args:
            in_str:
            arg2:
        Returns:
            join_tree:
            return2:
        """
        join_tree = {}
        elem_stack = []

        def next_token(curr_str: str):
            """
            {Description}
            
            Args:
                curr_str:
                arg2:
            Returns:
                curr_token: 当前的token
                left_str:
            """

            if curr_str[0] == "(":
                # print("curr_str = {}. end_pos = {}.".format(curr_str, 1))
                return "(", curr_str[1:]
            elif curr_str[0] == ")":
                # print("curr_str = {}. end_pos = {}.".format(curr_str, 1))
                return ")", curr_str[1:]
            else:
                # 找到下一个"(",")"," "
                pos1 = curr_str.find("(")
                pos2 = curr_str.find(")")
                pos3 = curr_str.find(" ")

                pos1 = pos1 if pos1 != -1 else 1e8
                pos2 = pos2 if pos2 != -1 else 1e8
                pos3 = pos3 if pos3 != -1 else 1e8
                end_pos = int(min([pos1, pos2, pos3]))
                # print("curr_str = {}. end_pos = {}.".format(curr_str, end_pos))
                if end_pos == pos3:
                    # 末端
                    return curr_str[:end_pos], curr_str[end_pos + 1:]
                else:
                    return curr_str[:end_pos], curr_str[end_pos:]
        
        curr_str = in_str
        while True:
            token, curr_str = next_token(curr_str=curr_str)
            if len(token) == 0:
                continue

            if len(curr_str) == 0:
                break
            if token == "(":
                pass
            elif token == ")":
                local_tree = {}
                right_child = elem_stack.pop()
                left_child = elem_stack.pop()
                local_tree = {
                    "left": left_child,
                    "right": right_child
                }
                elem_stack.append(local_tree)
            else:
                self.table_num += 1
                elem_stack.append(token)

        join_tree = elem_stack[0]
        return join_tree


    def is_bushy(self,):
        """
        join_order是否为灌木丛的类型

        Args:
            arg1:
            arg2:
        Returns:
            flag:
            return2:
        """
        flag = False
        def visit(node):
            # 访问函数
            nonlocal flag
            if isinstance(node, str):
                return
            elif isinstance(node, dict):
                if isinstance(node['left'], dict) and isinstance(node['right'], dict):
                    flag = True
                visit(node["left"])
                visit(node["right"])
            else:
                raise TypeError("Unsupported node type: {}.".format(type(node)))

        visit(self.join_tree)
        return flag
    

    def eval_levels(self):
        """
        判断table_subset所属的level
        
        Args:
            table_subset:
            arg2:
        Returns:
            level_dict:
            res2:
        """
        level_dict = {}
        def visit(node, level):
            if isinstance(node, str):
                level_dict[node] = level
                return
            elif isinstance(node, dict):
                visit(node["left"], level=level - 1)
                visit(node["right"], level=level - 1)
            else:
                raise TypeError("Unsupported node type: {}.".format(type(node)))
                    
        visit(self.join_tree, self.table_num)
        self.level_dict = level_dict
        return level_dict

    def table_subset_score(self, alias_subset):
        """
        table子集的分数，考虑最高level的表的位置
        
        Args:
            alias_subset: 别名组成的子集
            arg2:
        Returns:
            level_list:
            res2:
        """
        level_list = []
        for t in alias_subset:
            level_list.append(self.level_dict[t])

        return np.max(level_list)


    def build_intermediate_tree(self, join_tree = None):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        if join_tree is None:
            join_tree = self.join_tree

        inter_tree = deepcopy(join_tree)

        def visit(node):
            if isinstance(node, str):
                return [node,]
            elif isinstance(node, dict):
                left_list = visit(node['left'])
                right_list = visit(node['right'])
                merged_list = left_list + right_list

                # 完成表达式更新
                node['repr'] = tuple(sorted(merged_list))   
                return merged_list
            else:
                raise ValueError("build_intermediate_tree: Unsupported node type!")

        visit(inter_tree)
        return inter_tree
    
    def get_all_edges(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        edge_list = []
        inter_tree = self.build_intermediate_tree()

        # print(f"get_all_edges: inter_tree = {inter_tree}")

        def visit(node):
            if isinstance(node, str):
                return str(node)
            elif isinstance(node, dict):
                curr_repr = node['repr']
                left_repr = visit(node['left'])
                right_repr = visit(node['right'])

                edge_list.append((curr_repr, left_repr))
                edge_list.append((curr_repr, right_repr))
                return curr_repr
            
        visit(inter_tree)
        # print(f"get_all_edges: edge_list = {edge_list}")

        return edge_list
# %%
