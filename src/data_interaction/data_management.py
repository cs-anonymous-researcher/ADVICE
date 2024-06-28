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
from collections import defaultdict
from itertools import permutations, product
from query.query_construction import abbr_option
from utility.workload_spec import stats_foreign_mapping
from utility import utils


# %%

class DataManager(object):
    """
    数据管理类

    Members:
        field1:
        field2:
    """

    def __init__(self, data_src = "/home/lianyuan/Research/CE_Evaluator/data", \
        result_src = "/home/lianyuan/Research/CE_Evaluator/result/test", wkld_name = "job", \
        use_cache = True, load_comb = True):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        self.data_src = data_src
        self.result_src = result_src
        self.wkld_name = wkld_name
        self.table_cache = {}

        # 列的信息缓存，目前支持的字段为null_rate(数据中null的比例)，
        # 之后考虑添加新的字段
        self.column_info_cache = {}

        self.use_cache = use_cache
        self.tbl_abbr = abbr_option[wkld_name]  # 别名映射
        self.abbr_inverse = {}
        for k, v in self.tbl_abbr.items():
            self.abbr_inverse[v] = k
        self.meta_load()
        self.build_equivalence_join_group()     # 构建等价类

        if load_comb == True:
            self.load_combination_info()

    def load_combination_info(self,):
        """
        加载schema组合的信息
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """

        comb_path = p_join(self.data_src, self.wkld_name, f"combination_{self.wkld_name}.pkl")
        result = utils.load_pickle(comb_path)
        schema_comb_list = []
        for k, v in result.items():
            schema_comb_list.extend(v)

        alias_comb_list = [tuple([self.tbl_abbr[i] for i in a]) for a in schema_comb_list]
        alias_comb_sorted = [tuple(sorted(a)) for a in alias_comb_list]
        all_subqueries_dict = self.construct_subquery_dict(alias_comb_list=alias_comb_sorted)

        self.all_subqueries_dict = all_subqueries_dict

    def construct_subquery_dict(self, alias_comb_list):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        all_subqueries_dict = defaultdict(list)
        for alias_comb_key in alias_comb_list:
            for alias_comb_value in alias_comb_list:
                if set(alias_comb_value).issubset(set(alias_comb_key)):
                    all_subqueries_dict[alias_comb_key].append(alias_comb_value)

        return all_subqueries_dict


    def get_meta_subqueries(self, query_meta):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        schema_list = query_meta[0]
        alias_list = [self.tbl_abbr[s] for s in schema_list]
        return self.get_subqueries(alias_list)


    def get_subqueries(self, alias_list):
        """
        根据alias_list获得所有的subquery组合
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        alias_tuple = tuple(sorted(alias_list))
        subquery_repr_list = self.all_subqueries_dict[alias_tuple]
        return subquery_repr_list


    def meta_load(self,):
        """
        加载解析元数据的信息，这里需要考虑table_spec、table_excl的问题，
        其中column_spec、column_excl暂时不去考虑。
        
        Args:
            None
        Returns:
            res1:
            res2:
        """
        meta_path = p_join(self.data_src, self.wkld_name, "header.json")
        with open(meta_path, "r") as f_in:
            self.meta_info = json.load(f_in)

        table_spec = self.meta_info['table_spec']
        self.column_spec = self.meta_info['column_spec']

        def filter_by_spec(kw):
            if len(table_spec) > 0:
                res_dict = {}
                for tbl in table_spec:
                    res_dict[tbl] = self.meta_info[kw][tbl]
            else:
                res_dict = self.meta_info[kw]
            return res_dict

        self.header_dict = filter_by_spec('all_columns')
        self.primary_keys = filter_by_spec('primary_keys')
        self.data_types = filter_by_spec('column_types')

        if len(table_spec) > 0:
            self.foreign_keys = {}
            for tbl in table_spec:
                self.foreign_keys[tbl] = []
                try:
                    for src_col, ref_tbl, ref_col in self.meta_info['foreign_keys'][tbl]:
                        if ref_tbl in table_spec:
                            self.foreign_keys[tbl].append((src_col, ref_tbl, ref_col))
                except KeyError as e:
                    print(f"meta_load: meet KeyError. meta_info = {self.meta_info['foreign_keys']}.")
                    raise e
        else:
            self.foreign_keys = self.meta_info['foreign_keys']

        return self.header_dict, self.primary_keys,\
            self.data_src, self.foreign_keys

    def workload_serialization(self, tbl_name = None):
        """
        序列化一整个workload
        
        Args:
            tbl_name:
        Returns:
            res1:
            res2:
        """

        if tbl_name is None:
            todo_list = []
            wkld_dir = p_join(self.data_src, self.wkld_name)
            for f_name in os.listdir(wkld_dir):
                if f_name.endswith(".csv"):
                    f_pickle = f_name[:-4] + ".pkl"
                    f_path = p_join(wkld_dir, f_name)
                    if os.path.isfile(f_path) == False:
                        self.table_serialization(f_name[:-4])
                else:
                    continue
        else:
            self.table_serialization(tbl_name)


    def table_serialization(self, tbl_name):
        """
        表格序列化
        
        Args:
            tbl_name:
        Returns:
            data_df:
        """
        path_kw = "pickle_obj"
        csv_path = p_join(self.data_src, self.wkld_name, "{}.csv".format(tbl_name))
        # out_path = csv_path.replace(".csv", ".pkl")
        out_path = p_join(self.data_src, self.wkld_name, path_kw, "{}.pkl".format(tbl_name))

        col_names = self.header_dict[tbl_name]
        # 增加更多的sep选项
        sep_dict = {
            "job": ",", 
            "stats": ",",
            "dsb": "|"
        }
        sep = sep_dict[self.wkld_name]
        data_df = pd.read_csv(csv_path, names = col_names, sep=sep, \
            header = None, quotechar = '"', escapechar = "\\")
        
        # 创建pickle保存的路径
        os.makedirs(p_join(self.data_src, self.wkld_name, path_kw), exist_ok=True)  
        data_df.to_pickle(out_path)
        return data_df

    def get_table_meta(self, tbl_name):
        """
        获得一张表专属的元信息
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        info_dict = {
            "reuse_features": self.meta_info['reuse_features'],
            "src_path": self.meta_info['src_path'],
            # "primary": self.primary_keys[tbl_name],
            "foreign": self.foreign_keys[tbl_name],
            "data_type": self.data_types[tbl_name],
        }

        try:
             info_dict["primary"] = self.primary_keys[tbl_name]
        except KeyError as e:
             info_dict["primary"] = ""
        return info_dict

    def load_table_meta(self, tbl_name):
        return [tbl_name,], []

    # def convert_params_meta(self, column_list):
    #     """
    #     主要是把源表名以abbr的形式替换，其他东西不改变
        
    #     Args:
    #         column_list:
    #     Returns:
    #         params_meta:
    #     """
    #     params_meta = []
    #     for table_name, column_name in column_list:
    #         params_meta.append((self.tbl_abbr[table_name], column_name))
    #     return params_meta

    def empty_cache(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        del self.table_cache        # 清理回收缓存
        self.table_cache = {}

    def load_table(self, tbl_name):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        path_kw = "pickle_obj"
        data_path = p_join(self.data_src, self.wkld_name,\
            path_kw, "{}.pkl".format(tbl_name))

        if os.path.isfile(data_path) == False:
            self.table_serialization(tbl_name)

        if self.use_cache == True:
            if tbl_name in self.table_cache.keys():
                return self.table_cache[tbl_name]
            else:
                res_df = pd.read_pickle(data_path)
                self.table_cache[tbl_name] = res_df
                return self.table_cache[tbl_name]
        else:
            return pd.read_pickle(data_path)

    def load_table_with_prefix(self, tbl_name):
        """
        添加带前缀的dataFrame
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        return self.add_prefix_on_dataframe(self.load_table(tbl_name), tbl_name)
    

    def get_null_rate(self, schema_name, column_name):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        key = (schema_name, column_name)
        if key in self.column_info_cache.keys() and \
            self.column_info_cache[key]['null_rate'] > 0:
            return self.column_info_cache[key]['null_rate']
        else:
            if key not in self.column_info_cache.keys():
                # 初始化字典
                self.column_info_cache[key] = {}

            origin_column: pd.Series = self.load_table(tbl_name=schema_name)[column_name]
            filter_column: pd.Series = origin_column.dropna(inplace=False)
            null_rate = 1.0 - (len(filter_column) / len(origin_column))
            self.column_info_cache[key]['null_rate'] = null_rate
            return null_rate

    # @utils.timing_decorator
    def get_valid_columns(self, schema_name, primary_key = False, \
            foreign_key = False, str_column = False, null_rate_threshold = 0.3):
        """
        获得合法的列
        
        Args:
            schema_name:
            primary_key:
            foreign_key:
            str_column:
            null_rate: 最大能容忍数据为NULL的比例
        Returns:
            column_list:
        """
        # 针对column_spec设置提前返回的出口

        if schema_name in self.column_spec.keys():
            return self.column_spec[schema_name]
        
        column_list = []
        meta_dict = self.get_table_meta(schema_name)
        all_columns = self.header_dict[schema_name]
        excluded_columns = []

        # 得考虑到没有主键的情况，感觉为空的情况也是兼容的
        if primary_key == False:
            excluded_columns.append(meta_dict['primary'])

        if foreign_key == False:
            for src_col, ref_tbl, ref_col in meta_dict['foreign']:
                excluded_columns.append(src_col)

        if str_column == False:
            for col, dtype in meta_dict['data_type']:
                if "character" in dtype: 
                    excluded_columns.append(col)

        for col in all_columns:
            if col not in excluded_columns:
                curr_null_rate = self.get_null_rate(schema_name=schema_name, column_name=col)
                if curr_null_rate < null_rate_threshold:
                    column_list.append(col)

        return column_list



    def construct_valid_columns_config(self, primary_key = False, \
            foreign_key = False, str_column = False, null_rate_threshold = 0.3):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        column_dict, res_dict = {}, {}

        for schema in self.header_dict.keys():
            column_list = self.get_valid_columns(schema, \
                primary_key, foreign_key, str_column, null_rate_threshold)
            column_dict[schema] = column_list

        #
        res_dict['columns'] = column_dict
        res_dict['alias'] = self.tbl_abbr
        return res_dict

    def clip_dataframe(self, in_df, schema_name):
        """
        裁剪dataframe(目前仅根据有效的列进行裁剪)
        
        Args:
            in_df:
            schema_name:
        Returns:
            out_df:
        """
        valid_columns = []
        foreign_columns = []
        string_columns = []
        meta_dict = self.get_table_meta(schema_name)
        column_list = in_df.columns

        # foreign key暂时不考虑
        # for src_col, ref_tbl, ref_col in meta_dict['foreign']:
        #     foreign_columns.append(src_col)

        for col, dtype in meta_dict['data_type']:
            if "character" in dtype: 
                string_columns.append(col)

        for col in column_list:
            if col == meta_dict['primary'] or col in string_columns:
                continue
            else:
                valid_columns.append(col)

        out_df = in_df[valid_columns]
        return out_df

    def dump_intermidate(self, obj_name, data):
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

    def save_result(self, obj_name, data):
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
    
    def get_join_columns(self, schema):
        """
        获得待join的列
        
        Args:
            schema:
        Returns:
            column_list:
        """
        column_list = []
        for src_col, ref_tbl, ref_col in self.foreign_keys[schema]:
            column_list.append(src_col)
            
        return column_list

    def build_local_equivalence_join_group(self, foreign_mapping):
        """
        构造等价的局部join group
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        
        group_list = []
        col_set = set()
        equivalent_col_pairs = []

        # print("build_local_equivalence_join_group: {}.".format(foreign_mapping))

        for src_tbl, v in foreign_mapping.items():
            src_col, ref_tbl, ref_col = v
            equivalent_col_pairs.append(((src_tbl, src_col), (ref_tbl, ref_col)))
            col_set.add((src_tbl, src_col))
            col_set.add((ref_tbl, ref_col))

        parent_dict = {}
        for t in col_set:
            parent_dict[t] = t

        def get_parent(t):
            cnt = 0
            while parent_dict[t] != t:
                t = parent_dict[t]
                cnt += 1
                if cnt > 20:
                    print("Warning! get_parent好像写错了")
                    break
            return t

        for c1, c2 in equivalent_col_pairs:
            p1 = get_parent(c1)
            p2 = get_parent(c2)
            parent_dict[c1] = p1
            parent_dict[c2] = p2

            if p1 != p2:
                parent_dict[p1] = p2    # 调整parent

        parent_set = set()
        for t in col_set:
            p = get_parent(t)
            parent_set.add(p)

        group_mapping = {}
        for idx, p in enumerate(parent_set):
            group_mapping[p] = idx
            group_list.append([p])

        for t in col_set:
            p = get_parent(t)
            if t != p:
                idx = group_mapping[p]
                group_list[idx].append(t)

        equivalent_idx = defaultdict(list)
        for idx, group in enumerate(group_list):
            for item1, item2 in permutations(group, 2):
                t1, t2 = item1[0], item2[0]
                equivalent_idx[(t1, t2)].append(idx)

        return group_list, equivalent_idx



    def build_equivalence_join_group(self, ):
        """
        构建等价的连接组，这里考虑的是边之间的等价情况
        
        Args:
            None
        Returns:
            equivalence_group:
        """

        group_list = []
        col_set = set()
        equivalent_col_pairs = []
        for src_tbl, v in self.foreign_keys.items():
            for src_col, ref_tbl, ref_col in v:
                equivalent_col_pairs.append(((src_tbl, src_col), (ref_tbl, ref_col)))
                col_set.add((src_tbl, src_col))
                col_set.add((ref_tbl, ref_col))

        parent_dict = {}
        for t in col_set:
            parent_dict[t] = t

        def get_parent(t):
            cnt = 0
            while parent_dict[t] != t:
                t = parent_dict[t]
                cnt += 1
                if cnt > 20:
                    print("Warning! get_parent好像写错了")
                    break
            return t

        for c1, c2 in equivalent_col_pairs:
            p1 = get_parent(c1)
            p2 = get_parent(c2)
            parent_dict[c1] = p1
            parent_dict[c2] = p2

            if p1 != p2:
                parent_dict[p1] = p2    # 调整parent

        parent_set = set()
        for t in col_set:
            p = get_parent(t)
            parent_set.add(p)

        group_mapping = {}
        for idx, p in enumerate(parent_set):
            group_mapping[p] = idx
            group_list.append([p])

        for t in col_set:
            p = get_parent(t)
            if t != p:
                idx = group_mapping[p]
                group_list[idx].append(t)

        self.equivalent_col_groups = group_list     # 获得所有等价的组，每一组包含了所有的列
        self.equivalent_idx = defaultdict(list)
        for idx, group in enumerate(group_list):
            for item1, item2 in permutations(group, 2):
                t1, t2 = item1[0], item2[0]
                self.equivalent_idx[(t1, t2)].append(idx)

        return group_list


    def add_prefix_on_dataframe(self, in_df, tbl_name):
        """
        {Description}
        
        Args:
            in_df:
            tbl_name:
        Returns:
            out_df:
        """
        table_abbr = self.tbl_abbr[tbl_name]
        df_columns = in_df.columns
        prefix = table_abbr + "_"
        valid = True
        mapping_dict = {}
        for col in df_columns:
            mapping_dict[col] = prefix + col

        out_df = in_df.rename(columns = mapping_dict)
        # print("add_prefix_on_dataframe: out_df = {}".format(out_df.head()))
        
        return out_df

    def infer_table_join(self, left_table, right_table):
        """
        推断两表间JOIN的条件
        
        Args:
            left_table:
            right_table:
        Returns:
            left_on:
            right_on:
        """
        left_meta = self.load_table_meta(tbl_name = left_table)
        right_meta = self.load_table_meta(tbl_name = right_table)
        return self.infer_join_conditions(left_meta, right_meta)

    def infer_equivalent_groups(self, left_table, right_table):
        """
        {Description}
    
        Args:
            left_table:
            right_table:
        Returns:
            return1:
            return2:
        """
        t1, t2 = left_table, right_table
        return self.equivalent_idx[(t1, t2)]

    def infer_join_conditions(self, left_meta, right_meta, foreign_mapping = None, allow_empty=True):
        """
        推导连接的条件，目前的代码考虑了所有外键，改动以后考虑foreign_mapping
        的内容
        
        Args:
            left_meta: 左边的元信息
            right_meta: 右边的元信息
        Returns:
            left_on:
            right_on:
        """

        left_on, right_on = [], []
        left_schema, right_schema = left_meta[0], right_meta[0]
        sharing_groups = dict()     # 分析共享的等价组，然后各自任选一列进行连接

        if foreign_mapping is None:
            # 直接用全局的信息
            equivalent_idx = self.equivalent_idx
            equivalent_col_groups = self.equivalent_col_groups
        else:
            # 使用局部的信息
            equivalent_col_groups, equivalent_idx = \
                self.build_local_equivalence_join_group(foreign_mapping = foreign_mapping)
        
        for t1, t2 in product(left_schema, right_schema):
            for idx in equivalent_idx[(t1, t2)]:
                sharing_groups[idx] = t1, t2
        
        for idx, (t1, t2) in sharing_groups.items():
            for item in equivalent_col_groups[idx]:
                if item[0] == t1:
                    left_on.append("{}_{}".format(self.tbl_abbr[t1], item[1]))
                elif item[0] == t2:
                    right_on.append("{}_{}".format(self.tbl_abbr[t2], item[1]))
                else:
                    pass
        
        if len(left_on) == 0 and len(right_on) == 0 and allow_empty == False:
            print(f"infer_join_conditions: sharing_groups = {sharing_groups}. equivalent_col_groups = {equivalent_col_groups}. equivalent_idx = {equivalent_idx}.")
            raise ValueError(f"infer_join_conditions: join condition is empty. left_schema = {left_schema}. right_schema = {right_schema}.")
        
        return left_on, right_on
    

    def get_join_condition(self, left_schema, right_schema):
        """
        获得连接的条件，左右两边都是单表
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        if self.wkld_name == "job":
            if left_schema == "title":
                left_on = "id"
            else:
                left_on = "movie_id"

            if right_schema == "title":
                right_on = "id"
            else:
                right_on = "movie_id"
                
            return left_on, right_on
        elif self.wkld_name == "stats":
            # 同时出现posts和users的情况
            if left_schema == "posts" and right_schema == "users":
                return "OwnerUserId", "Id"
            elif left_schema == "users" and right_schema == "posts":
                return "Id", "OwnerUserId"

            if left_schema in ["posts", "users"]:
                left_on = "Id"
            else:
                left_on = stats_foreign_mapping[left_schema][0]
            
            if right_schema in ["posts", "users"]:
                right_on = "Id"
            else:
                right_on = stats_foreign_mapping[right_schema][0]
            return left_on, right_on
        elif self.wkld_name == "dsb":
            pass
        else:
            raise ValueError("Unsupported workload!")
        
# %%

def get_all_subqueries(workload, alias_list):
    local_manager = DataManager(wkld_name=workload)
    return local_manager.get_subqueries(alias_list)

# %%
