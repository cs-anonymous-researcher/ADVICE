#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from os.path import join as p_join
from matplotlib import pyplot as plt
import json, os, time, re, math
import pickle
from copy import copy, deepcopy
from collections import defaultdict, namedtuple
import hashlib
from utility import workload_spec
from query.query_construction import abbr_option

class ColumnExpr(namedtuple("ColumnExpr", ["alias", "column"])):
    '''
    
    '''
    __slots__ = ()
    def __str__(self):
        return "{}.{}".format(self.alias, self.column)

    


class ValueExpr(namedtuple("ValueExpr", ["value", "type"])):
    __slots__ = ()
    def __str__(self):
        if self.type == "str":
            return "\'{}\'".format(self.value)
        else:
            return str(self.value)

def deterministic_hash(string):
    """
    确定性的hash函数

    Args:
        string:
    Returns:
        hash_number:
    """
    return int(hashlib.sha1(str(string).encode("utf-8")).hexdigest(), 16)

def create_default_set():
    return defaultdict(set)

class SQLParser(object):
    """
    手动解析SQL文本，获得信息，这里不考虑复杂的语句?

    Members:
        field1:
        field2:
    """


    # def __init__(self, sql_text, workload = None):
    def __init__(self, sql_text, workload):     # 强制将workload作为入参
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        # print("new SQLParser initilization!")
        # print("SQLParser: sql_text = {}.".format(sql_text))
        self.set_sql_text(sql_text = sql_text)
        self.workload = workload
        if workload is not None:
            self.load_external_alias_info(alias_external = abbr_option[workload])
            # self.update_global_foreign_mapping(workload=workload)   # 更新全局信息

            if workload == "stats":
                workload_foreign_mapping = workload_spec.stats_foreign_mapping
            elif workload in ["job", "job-light"]:
                workload_foreign_mapping = workload_spec.job_light_foreign_mapping
            elif workload == "release":
                workload_foreign_mapping = workload_spec.release_foreign_mapping
            elif workload == "dsb":
                workload_foreign_mapping = workload_spec.dsb_foreign_mapping
            self.load_workload_info(workload_foreign_mapping)
            # self.alias_mapping = abbr_option[workload]

    def set_sql_text(self, sql_text):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        self.alias_mapping = {}
        self.inverse_alias = {}

        self.schema_list = []

        self.join_mapping = defaultdict(set)        # Join映射
        self.join_columns = defaultdict(set)        # 每一个表参与join的列信息

        # self.columns_mapping = defaultdict(lambda: defaultdict(set))    # schema行信息
        self.columns_mapping = defaultdict(create_default_set)
        self.columns_type = defaultdict(dict)      # 嵌套dictionary，用来指明每一个column的类型

        self.sql_text = self.preprocess(sql_text)
        joined_table_part, condition_part = self.query_split(sql_text=self.sql_text)
        self.parse_joined_table(joined_table_part)
        self.parse_condition_table(condition_part)
        self.enrich_join_condition()                # 丰富join的条件

    def generate_meta(self,):
        """
        生成当前查询的meta信息
    
        Args:
            None
        Returns:
            meta_info
        """
        schema_list = self.schema_list
        filter_list = []
        
        # for k, v in self.alias_mapping.items():
        #     schema_list.append(v)

        for curr_alias, col_info in self.columns_mapping.items():
            for col, values in col_info.items():
                lower_bound, upper_bound = None, None
                for v in values:
                    kw, value_expr = v
                    actual_val = value_expr.value
                    if kw == "=":
                        lower_bound = actual_val
                        upper_bound = actual_val
                        break
                    elif kw == "<=":
                        upper_bound = actual_val
                    elif kw == ">=":
                        lower_bound = actual_val
                    elif kw == "<":
                        upper_bound = actual_val - 1
                    elif kw == ">":
                        upper_bound = actual_val + 1
                    else:
                        raise ValueError("generate_meta: Unsupported kw = {}".format(kw))

                # 表缩写，列名，下界，上界(这里str暂时全部转化成小写)
                curr_info = (curr_alias.lower(), col.lower(), lower_bound, upper_bound) 
                filter_list.append(curr_info)

        return schema_list, filter_list

    def generate_subquery_meta(self, alias_list):
        """
        生成子查询相关的meta信息
        
        Args:
            alias_list:
        Returns:
            subquery_meta:
        """
        schema_list = []
        filter_list = []
        
        for k, v in self.alias_mapping.items():
            if k in alias_list:             # 需要满足在alias_list中
                schema_list.append(v)

        for curr_alias, col_info in self.columns_mapping.items():
            if curr_alias in alias_list:    # 需要满足在alias_list中
                for col, values in col_info.items():
                    lower_bound, upper_bound = None, None
                    for v in values:
                        kw, value_expr = v
                        actual_val = value_expr.value
                        if kw == "=":
                            lower_bound = actual_val
                            upper_bound = actual_val
                            break
                        elif kw == "<=":
                            upper_bound = actual_val
                        elif kw == ">=":
                            lower_bound = actual_val
                        elif kw == "<":
                            upper_bound = actual_val - 1
                        elif kw == ">":
                            upper_bound = actual_val + 1
                        else:
                            raise ValueError("generate_meta: Unsupported kw = {}".format(kw))

                    # 表缩写，列名，下界，上界(这里str暂时全部转化成小写)
                    curr_info = (curr_alias.lower(), col.lower(), lower_bound, upper_bound) 
                    filter_list.append(curr_info)

        return schema_list, filter_list


    def print_elements(self,):
        """
        打印相关的元素
        
        Args:
            None
        Returns:
            None
        """

        print("alias: {}".format(self.alias_mapping))
        print("join info:")
        for k, v in self.join_mapping.items():
            print("table_pair: {} . conditions: {}".format(k, v))

        print("columns info: ")
        for k, v in self.columns_mapping.items():
            print("table: {}. conditions: {}".format(k, v))


    def preprocess(self, sql_text):
        """
        对SQL进行预处理，关键词大写
        
        TODO: 考虑把表达式中多余的空格给去掉
        
        Args:
            sql_text
        Returns:
            new_sql
        """
        new_sql = str(sql_text)
        # 新增: 将as关键字删除
        new_sql = new_sql.replace(" as ", " ")
        # 删除表达式中多余的空格
        new_sql = new_sql.replace(" <", "<")
        new_sql = new_sql.replace(" >", ">")
        new_sql = new_sql.replace(" =", "=")
        new_sql = new_sql.replace("< ", "<")
        new_sql = new_sql.replace("> ", ">")
        new_sql = new_sql.replace("= ", "=")

        new_sql = new_sql.replace(" from ", " FROM ")
        new_sql = new_sql.replace(" where ", " WHERE ")
        self.sql_text = new_sql
        return new_sql

    def query_calibration(self, sql_text:str):
        '''
        query各部分定位
        '''
        FROM_position, WHERE_positon, colon_position = 0, 0, 0
        FROM_position = sql_text.find("FROM")
        WHERE_positon = sql_text.find("WHERE")
        colon_position = sql_text.find(";")
        return FROM_position, WHERE_positon, colon_position


    def query_split(self, sql_text:str):
        '''
        拆分query
        '''
        FROM_position, WHERE_positon, colon_position = self.query_calibration(sql_text)
        joined_table_part = sql_text[FROM_position + 5 : WHERE_positon]
        condition_part = sql_text[WHERE_positon + 6 : colon_position]
        return joined_table_part, condition_part

    def enrich_join_condition(self,):
        """
        利用传递性，将连接的条件全部补全
        
        Args:
            None
        Returns:
            None
        """
        origin_mapping = deepcopy(self.join_mapping)
        equivalent_conditions = {}      # 相互等价的条件

        def eval(expr):
            if expr not in equivalent_conditions.keys():
                equivalent_conditions[expr] = expr

        def get_parent(expr):
            if equivalent_conditions[expr] == expr:
                return equivalent_conditions[expr]
            else:
                par_expr = get_parent(equivalent_conditions[expr])
                equivalent_conditions[expr] = par_expr
                return par_expr

        for k, v in origin_mapping.items():
            for expr1, expr2 in v:
                # 分别两个表达式
                eval(expr1), eval(expr2)
                par1, par2 = get_parent(expr1), get_parent(expr2)
                equivalent_conditions[par1] = par2

        # 遍历equivalent_conditions，确定最终的parent
        for k in list(equivalent_conditions.keys()):
            equivalent_conditions[k] = get_parent(k)

        # print("equivalent_conditions = {}.".format(equivalent_conditions))
        equivalent_groups = defaultdict(list)

        for expr, par_expr in equivalent_conditions.items():
            equivalent_groups[par_expr].append(expr)

        result_mapping = defaultdict(set)

        for group in equivalent_groups.values():
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    ii, jj = group[i].alias, group[j].alias
                    result_mapping[(ii, jj)].add((group[i], group[j]))  # 对称的添加两个条件
                    result_mapping[(jj, ii)].add((group[j], group[i]))

        # print("len(result_mapping) = {}. len(self.join_mapping) = {}.".\
        #     format(len(result_mapping), len(self.join_mapping)))
        self.join_mapping = result_mapping
        return origin_mapping, result_mapping


    def parse_joined_table(self, joined_table_text:str):
        '''
        解析语句的Join关系
        '''
        item_list = list(map(lambda a:a.strip(), joined_table_text.split(",")))
        # print(item_list)
        for i in item_list:
            if len(i.split(" ")) == 2:
                schema, alias = i.split(" ")
            else:
                schema = i.strip()
                alias = schema      # alias和schema保持一致

            # print("schema = {}. alias = {}".format(schema, alias))

            self.alias_mapping[alias] = schema   # 处理相应的映射关系
            self.inverse_alias[schema] = alias
            self.schema_list.append(schema)


    def parse_expr(self, expr_text:str):
        '''
        解析condition中的一个表达式
        '''
        
        if "." in expr_text:
            alias, column = expr_text.split(".")
            return ColumnExpr(alias.lower(), column.lower())    # 转换成小写
        else:
            if expr_text.lstrip('-').isdigit() == True:
                # 考虑了负号的情况
                return ValueExpr(int(expr_text), "int")
            else:
                return ValueExpr(expr_text.strip("\'").strip("\""), "str")


    def parse_condition_table(self, condition_text:str):
        '''
        解析语句的相关条件
        '''
        item_list = list(map(lambda a:a.strip(), condition_text.split("AND")))
        for i in item_list:
            kw = "="
            if "<=" in i:
                kw = "<="
                # print("<= has detected")
            elif ">=" in i:
                kw = ">="
                # print(">= has detected")
            elif "<" in i:
                kw = "<"
            elif ">" in i:
                kw = ">"
            elif "!=" in i:
                kw = "!="

            
            try:
                left_part, right_part = i.split(kw)
            except ValueError as e:
                print("parse_condition_table: i = {}. kw = {}.".format(i, kw))
                raise e
            
            left_part, right_part = left_part.strip(), right_part.strip()
            left_expr, right_expr = self.parse_expr(left_part), self.parse_expr(right_part)
            if isinstance(left_expr, ColumnExpr) and isinstance(right_expr, ColumnExpr):
                # print(f"parse_condition_table: left_expr = {left_expr}. right_expr = {right_expr}.")
                # 这是一个inner join的condition
                self.join_mapping[(left_expr.alias, right_expr.alias)].add((left_expr, right_expr))
                self.join_mapping[(right_expr.alias, left_expr.alias)].add((right_expr, left_expr)) # 这里的顺序应该反过来

                # 更新join_columns的信息，直接处理成小写
                self.join_columns[left_expr.alias.lower()].add(left_expr.column.lower())
                self.join_columns[right_expr.alias.lower()].add(right_expr.column.lower())

            elif isinstance(left_expr, ColumnExpr) and isinstance(right_expr, ValueExpr):
                # Column上的一个Filter
                # self.columns_mapping[left_expr.alias].add((left_expr.column, kw))
                self.columns_mapping[left_expr.alias][left_expr.column].add((kw, right_expr))
                self.columns_type[left_expr.alias][left_expr.column] = right_expr.type  # 设置column类型
            else:
                raise ValueError("Unexpected Expression Form")
       

    def construct_complete_queries(self, limit_join = False):
        """
        构造完整的查询
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        return self.construct_sub_queries(self.alias_mapping.keys())

    def redundancy_elimination(self, in_sql):
        """
        消除查询中的TRUE条件，以下是TRUE可能出现的模板以及对应消除的结果
        "AND TRUE AND" ==> "AND"
        "AND TRUE;" ==> ";"
        "WHERE TRUE AND" ==> "WHERE "
        "WHERE TRUE;" ==> ";"
        "" 

        Args:
            in_sql:
        Returns:
            out_sql:
        """
        out_sql = in_sql
        out_sql = out_sql.replace("AND TRUE AND", "AND")
        out_sql = out_sql.replace(" AND TRUE;", ";")
        out_sql = out_sql.replace("WHERE TRUE AND", "WHERE ")
        out_sql = out_sql.replace("WHERE TRUE;", ";")

        # print("redundancy_elimination: in_sql = {}. out_sql = {}.".\
        #     format(in_sql, out_sql))
        return out_sql


    def construct_sub_queries(self, alias_list, limit_join = False):
        """
        根据别名生成对应基数的查询
        20230425: 需要重写此方法
        
        Args:
            alias_set:
        Returns:
            query_text: 
            limit_join: 
        """

        def make_schema_str(schema_set, alias2table):
            '''
            设置JOIN中SCHMEA字符串，包含别名的信息
            '''
            return ",".join(map(lambda a:"{} {}".format(alias2table[a], a)\
                if a in alias2table else "{}".format(a), schema_set))
            # return ",".join(map(lambda a:"{} AS {}".format(alias2table[a], a)\
            #     if a in alias2table else "{}".format(a), schema_set))

        def make_condition_str(condition_set):
            '''
            
            '''
            if len(condition_set) == 0:
                return "TRUE"
            condition_list = []
            # condition_content = " AND ".join(map(lambda a: "{} {} {}".\
            #     format(a[0], a[1], a[2]), sorted(list(condition_set))))

            for alias, col, sub_cond_set in condition_set:
                for cond in sub_cond_set:
                    condition_list.append("{}.{} {} {}".\
                        format(alias, col, cond[0], cond[1]))
            return " AND ".join(condition_list)

        def make_join_str(join_set):
            '''
            
            '''
            if len(join_set) == 0:
                return "TRUE"
            join_content = " AND ".join(map(lambda a: "{} = {}".\
                format(a[0], a[1]), sorted(list(join_set))))
            return join_content

        condition_set = list()   #
        join_set = list()        # 

        for i in range(len(alias_list)):
            ii = alias_list[i]
            if ii in self.columns_mapping.keys():
                for col, values in self.columns_mapping[ii].items():
                    condition_set.append((ii, col, values))           # 

            for j in range(i+1, len(alias_list)):
                jj = alias_list[j]
                if (ii, jj) in self.join_mapping.keys():
                    if limit_join == True:
                        if (ii != "t" and jj != "t"): # job-light下hardcode取巧的做法，之后需要修改
                            continue
                    for join_cond in self.join_mapping[(ii, jj)]:
                        join_set.append(join_cond)      #

        schema_str = make_schema_str(alias_list, self.alias_mapping)        # schema字符串
        join_str = make_join_str(join_set)                                  # join字符串
        filter_str = make_condition_str(condition_set)                      # condition字符串

        # 构造整个查询的hash结构
        sql_prefix = "SELECT COUNT(*) FROM" # 前缀
        sql_template = "{} {} WHERE {} AND {};"

        return self.redundancy_elimination(sql_template.format(sql_prefix, schema_str, join_str, filter_str))


    def load_external_alias_info(self, alias_external):
        """
        加载外部的别名映射
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        self.inverse_alias = alias_external
        for k, v in alias_external.items():
            self.alias_mapping[v] = k
        return alias_external

    def load_workload_info(self, workload_foreign_mapping):
        """
        加载workload的信息，构造等价类的字典
        
        Args:
            workload_foreign_mapping: 负载的主外键映射
        Returns:
            equivalent_group_dict: 等价类，key是主键，value是外键构成的字典，
            主键和外键在这里都是(schema, column)这样元组的表示
        """
        if self.workload == "release":
            # release的情况下直接特判
            self.equivalent_group_dict = {
                ('release', 'id'): {('release_country', 'release'), ('medium', 'release'), ('release_meta', 'id'), 
                                    ('release_label', 'release'), ('release_tag', 'release')}, 
                ('release_group', 'id'): {('release', 'release_group')}, 
                ('artist_credit', 'id'): {('release', 'artist_credit')}
            }
            equivalent_group_dict = self.equivalent_group_dict
        else:
            equivalent_group_dict = defaultdict(set)

            for fk_tbl, (fk_col, pk_tbl, pk_col) in workload_foreign_mapping.items():
                equivalent_group_dict[(pk_tbl, pk_col)].add((fk_tbl, fk_col))

            self.equivalent_group_dict = equivalent_group_dict
        return equivalent_group_dict


    def get_primary_key(self, schema_name, column_name):
        """
        获得对应的primary key
        
        Args:
            schema_name:
            column_name:
        Returns:
            alias_primary:
            column_primary:
        """
        # schema_name, column_name全部转化成小写
        schema_name, column_name = schema_name.lower(), column_name.lower()

        # print("self.equivalent_group_dict = {}".format(self.equivalent_group_dict))
        # print("schema_name = {}. column_name = {}.".format(schema_name, column_name))
        if (schema_name, column_name) in self.equivalent_group_dict.keys():
            # 如果是主键，直接return None
            return None

        for k, v in self.equivalent_group_dict.items():
            schema_primary, column_primary = k
            if (schema_name, column_name) in v:
                return self.inverse_alias[schema_primary], column_primary

        raise ValueError("get_primary_key: not find. fk = {}. equivalent_group_dict = {}. join_mapping = {}. query_text = {}".\
                         format((schema_name, column_name), self.equivalent_group_dict, self.join_mapping, self.sql_text))

    def make_pk_fk_join_str(self, sub_equivalent_dict):
        """
        {Description}
        
        Args:
            sub_equivalent_dict:
        Returns:
            join_str:
        """
        join_element_list = []
        for pk, fk_list in sub_equivalent_dict.items():
            join_element_list.extend(self.build_equivalent_join_list(pk, fk_list))
        return " AND ".join(join_element_list)

    def build_equivalent_join_list(self, pk, fk_list):
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
        for fk in fk_list:
            result_list.append("{}.{} = {}.{}".format(pk[0], pk[1], fk[0], fk[1]))
        return result_list

    def construct_PK_FK_sub_query(self, alias_list, workload_info = None):
        """
        {Description}
        
        Args:
            alias_list: 别名列表
            workload_info: 数据负载的额外信息
        Returns:
            subquery_text:
        """
        if workload_info is not None:
            self.load_workload_info(workload_foreign_mapping = workload_info)

        sub_equivalent_dict = defaultdict(set)

        def make_schema_str(schema_set, alias2table):
            '''
            设置JOIN中SCHMEA字符串，包含别名的信息
            '''
            return ",".join(map(lambda a:"{} {}".format(alias2table[a], a)\
                if a in alias2table else "{}".format(a), schema_set))
            # return ",".join(map(lambda a:"{} AS {}".format(alias2table[a], a)\
            #     if a in alias2table else "{}".format(a), schema_set))

        def make_condition_str(condition_set):
            '''
            
            '''
            if len(condition_set) == 0:
                return "TRUE"
            condition_list = []
            # condition_content = " AND ".join(map(lambda a: "{} {} {}".\
            #     format(a[0], a[1], a[2]), sorted(list(condition_set))))

            for alias, col, sub_cond_set in condition_set:
                for cond in sub_cond_set:
                    condition_list.append("{}.{} {} {}".\
                        format(alias, col, cond[0], cond[1]))
            return " AND ".join(condition_list)

        condition_set = list()   #
        join_set = list()        # 

        # print("self.join_mapping.keys = {}.".format(self.join_mapping.keys()))

        for i in range(len(alias_list)):
            ii = alias_list[i]
            if ii in self.columns_mapping.keys():
                for col, values in self.columns_mapping[ii].items():
                    condition_set.append((ii, col, values))           # 

            for j in range(i+1, len(alias_list)):
                jj = alias_list[j]
                if (ii, jj) in self.join_mapping.keys():
                    for join_cond in self.join_mapping[(ii, jj)]:
                        # 处理单个连接条件
                        # print("ii = {}. jj = {}. join_cond = {}.".format(ii, jj, join_cond))
                        # schema_name, column_name = self.alias_mapping[ii], join_cond[0].column 
                        schema_name, column_name = self.alias_mapping[join_cond[0].alias], join_cond[0].column 
                        pk1 = self.get_primary_key(schema_name, column_name)
                        schema_name, column_name = self.alias_mapping[join_cond[1].alias], join_cond[1].column
                        pk2 = self.get_primary_key(schema_name, column_name)
                        # print("pk1 = {}. pk2 = {}.".format(pk1, pk2))
                        # 以防万一期间，所有的信息都从join_cond中获取
                        if pk1 is not None and pk2 is not None:
                            assert pk1 == pk2
                            fk1 = join_cond[0].alias, join_cond[0].column
                            fk2 = join_cond[1].alias, join_cond[1].column
                            sub_equivalent_dict[pk1].add(fk1)
                            sub_equivalent_dict[pk2].add(fk2)
                        elif pk1 is None:
                            fk2 = join_cond[1].alias, join_cond[1].column
                            sub_equivalent_dict[pk2].add(fk2)
                        elif pk2 is None:
                            fk1 = join_cond[0].alias, join_cond[0].column
                            sub_equivalent_dict[pk1].add(fk1)

        # alias_final，添加缺失的主键表
        alias_final = list(deepcopy(alias_list))
        # print("sub_equivalent_dict = {}".format(sub_equivalent_dict))
        for alias, column in sub_equivalent_dict.keys():
            if alias not in alias_final:
                alias_final.append(alias)
        # print("alias_list = {}. alias_final = {}.".format(alias_list, alias_final))

        schema_str = make_schema_str(alias_final, self.alias_mapping)       # schema字符串
        # print("sub_equivalent_dict = {}.".format(sub_equivalent_dict))
        pk_fk_join_str = self.make_pk_fk_join_str(sub_equivalent_dict)      # pk-fk形式的join字符串
        filter_str = make_condition_str(condition_set)                      # condition字符串

        # 构造整个查询的hash结构
        sql_prefix = "SELECT COUNT(*) FROM" # 前缀
        sql_template = "{} {} WHERE {} AND {};"

        final_query =  self.redundancy_elimination(sql_template.format(sql_prefix, \
            schema_str, pk_fk_join_str, filter_str))
    
        # print("construct_PK_FK_sub_query: alias_list = {}. final_query = {}.".format(alias_list, final_query))
        return final_query


    def get_single_table_query(self, alias):
        """
        根据别名获得查询
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        # query_template = "SELECT COUNT(*) from {table_name} where {conditions};"
        query_template = "SELECT COUNT(*) FROM {table_name} WHERE {conditions};"

        def make_condition_str(alias, column, cond_set):
            expr_list = []
            for op, val_expr in cond_set:
                # print(val_expr)
                if val_expr.type == "int":  # 判断表达式的类型
                    expr_list.append("{alias}.{column} {op} {value}".\
                        format(alias = alias, column = column, op = op, value = val_expr.value))
                else:
                    expr_list.append("{alias}.{column} {op} '{value}'".\
                        format(alias = alias, column = column, op = op, value = val_expr.value))

            return " AND ".join(expr_list)

        if self.alias_mapping[alias] != alias:
            table_name = "{} {}".format(self.alias_mapping[alias] , alias)
        else:
            table_name = alias

        if alias not in self.columns_mapping.keys():
            # 如果没有出现在条件里，就全表扫描
            conditions = "TRUE"
        else:
            # 否则添加条件
            col_cond_list = []
            for col_name, cond_set in self.columns_mapping[alias].items():
                # print("column_name = {}. cond_set = {}".format(col_name, cond_set))
                col_conditions = make_condition_str(alias, col_name, cond_set)
                col_cond_list.append(col_conditions)
            conditions = " AND ".join(col_cond_list)
        local_query = query_template.format(table_name = table_name, conditions = conditions)

        return self.redundancy_elimination(local_query)

        
    def get_single_table_conditions(self,):
        """
        将每一张单表对应的condition提取出来

        Args:
            None
        Returns:
            query_list:
        """
        query_template = "SELECT COUNT(*) from {table_name} where {conditions};"
        query_list = []

        def make_condition_str(alias, column, cond_set):
            expr_list = []
            for op, val_expr in cond_set:
                # print(val_expr)
                if val_expr.type == "int":  # 判断表达式的类型
                    expr_list.append("{alias}.{column} {op} {value}".\
                        format(alias = alias, column = column, op = op, value = val_expr.value))
                else:
                    expr_list.append("{alias}.{column} {op} '{value}'".\
                        format(alias = alias, column = column, op = op, value = val_expr.value))

            return " AND ".join(expr_list)

        alias_list = self.alias_mapping.keys()
        for alias in alias_list:
            if self.alias_mapping[alias] != alias:
                table_name = "{} {}".format(self.alias_mapping[alias] , alias)
            else:
                table_name = alias

            if alias not in self.columns_mapping.keys():
                # 如果没有出现在条件里，就全表扫描
                conditions = "TRUE"
            else:
                # 否则添加条件
                col_cond_list = []
                for col_name, cond_set in self.columns_mapping[alias].items():
                    # print("column_name = {}. cond_set = {}".format(col_name, cond_set))
                    col_conditions = make_condition_str(alias, col_name, cond_set)
                    col_cond_list.append(col_conditions)
                conditions = " AND ".join(col_cond_list)
            local_query = query_template.format(table_name = table_name, conditions = conditions)
            query_list.append(local_query)
        return alias_list, query_list

    def reference_check(self,):
        """
        检查这个查询是不是每一个表的外键只存在一个?
    
        Args:
            None
        Returns:
            flag: True代表只存在一个，False代表反之
        """
        flag = True     # 
        for k, v in self.join_columns.items():
            if len(v) == 2:
                if "id" not in v:   # id表示主键
                    flag = False
            elif len(v) > 2:
                flag = False

        return flag



    def update_global_foreign_mapping(self, workload = "stats"):
        """
        利用当前query的meta信息，更新全局的stats_foreign_mapping
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        tbl_col_list = []
        # 填充tbl_col_list的内容
        try:
            for k, v in self.join_mapping.items():
                item = list(v)[0]
                alias1, alias2 = item[0].alias, item[1].alias
                assert len(v) == 1  # 我们只考虑一个condition的join
                column1, column2 = item[0].column, item[1].column
                # 直接把名字全部转成小写
                tbl_col_list.append((self.alias_mapping[alias1].lower(), column1.lower()))
                tbl_col_list.append((self.alias_mapping[alias2].lower(), column2.lower()))
            # 调用外部的函数完成更新
            tbl_col_list = list(set(tbl_col_list))  # 去重
            # print("query_text = {}.".format(self.sql_text))
            # print("tbl_col_list = {}.".format(tbl_col_list))
        except KeyError as e:
            print(f"update_global_foreign_mapping: key_error = {e}. curr_query = {self.sql_text}")

        if workload != "stats":
            return workload_spec.get_spec_foreign_mapping(workload)

        if workload == "stats":
            res_foreign_mapping = workload_spec.update_stats_multi_steps(tbl_col_list)
        else:
            raise ValueError("Unrecognize workload: {}".format(workload))
        return res_foreign_mapping

from collections import namedtuple

WorkloadInfo = namedtuple("Workload", ["alias_mapping", "inverse_alias", \
    "join_mapping", "columns_mapping", "columns_type"])
    

def construct_database_info(workload_info:WorkloadInfo):
    """
    构造Database类初始化需要的信息
    
    Args:
        workload_info:
    Returns:
        table_spec: 指定的表
        column_spec: 指定的列
    """
    table_spec = set()
    column_spec = set()

    for k, v in workload_info.alias_mapping.items():
        table_spec.add(v)

    for k, v in workload_info.columns_mapping.items():
        for kk, vv in v.items():
            tbl, col = workload_info.alias_mapping[k], kk
            column_spec.add((tbl, col))

    return table_spec, column_spec

class WorkloadParser(object):
    """
    针对一批查询的解析器

    Members:
        field1:
        field2:
    """

    def __init__(self, sql_list):
        """
        {Description}

        Args:
            sql_list:
        """
        self.sql_list = sql_list
        self.alias_mapping = {}
        self.inverse_alias = {}
        self.join_mapping = defaultdict(set)       # Join映射
        self.columns_mapping = defaultdict(lambda: defaultdict(set))    # schema行信息
        self.columns_type = defaultdict(dict)      # 嵌套dictionary，用来指明每一个column的类型
        self.parse_workload(sql_list)

    def output_info(self,):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        return WorkloadInfo(self.alias_mapping, self.inverse_alias, \
            self.join_mapping, self.columns_mapping, self.columns_type)

    def add_sql_info(self, sql_parser: SQLParser):
        """
        添加单条sql的信息
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        for k, v in sql_parser.alias_mapping.items():
            self.alias_mapping[k] = v

        for k, v in sql_parser.inverse_alias.items():
            self.inverse_alias[k] = v
        
        for k, v in sql_parser.join_mapping.items():
            self.join_mapping[k].update(v)

        for k, v in sql_parser.columns_mapping.items():
            for kk, vv in v.items():
                # print(self.columns_mapping[k][kk])
                self.columns_mapping[k][kk].update(vv)

        for k, v in sql_parser.columns_type.items():
            for kk, vv in v.items():
                self.columns_type[k][kk] = vv

        return True

    def parse_workload(self, sql_list):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        for input_sql in sql_list:
            sql_parser = SQLParser(input_sql)
            # print(sql_parser.inverse_alias)
            self.add_sql_info(sql_parser)
