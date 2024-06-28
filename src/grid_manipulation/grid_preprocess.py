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
from data_interaction import data_management, mv_management
from copy import deepcopy, copy
from grid_manipulation import grid_base
from utility import workload_spec, utils
from data_interaction.mv_management import conditions_apply
from utility.utils import predicate_location, predicate_transform
from query import query_construction
# %%


def column_dict2list(column_dict):
    """
    {Description}
    
    Args:
        column_dict:
    Returns:
        selected_columns:
    """
    selected_columns = []

    for k, v in column_dict.items():
        if isinstance(v, (tuple, list)):
            for elem in v:
                selected_columns.append((k, elem))
        elif isinstance(v, str):
            selected_columns.append((k, v))
        else:
            raise TypeError("construct_multi_table_meta: Error! \
                type(v) = {}".format(type(v)))
    return selected_columns
# %%

def construct_multi_table_meta(table_list, column_dict, data_manager: data_management.DataManager):
    """
    {Description}
    
    Args:
        table_list:
        column_dict:
        data_manager:
    Returns:
        src_meta_list:
        selected_columns:
    """
    src_meta_list = []
    selected_columns = []

    for table in table_list:
        curr_meta = data_manager.load_table_meta(tbl_name = table)
        src_meta_list.append(curr_meta)

    for k, v in column_dict.items():
        if isinstance(v, (tuple, list)):
            for elem in v:
                selected_columns.append((k, elem))
        elif isinstance(v, str):
            selected_columns.append((k, v))
        else:
            raise TypeError("construct_multi_table_meta: Error! \
                type(v) = {}".format(type(v)))
    return src_meta_list, selected_columns

# %%

class SingleTableBuilder(object):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, workload: str, data_manager_ref: data_management.DataManager, \
        mv_manager_ref: mv_management.MaterializedViewManager):
        """
        {Description}

        Args:
            workload:
            data_manager_ref:
            mv_manager_ref:
        """
        self.workload = workload
        self.data_manager = data_manager_ref
        self.mv_manager = mv_manager_ref

        self.alias_mapping = deepcopy(data_manager_ref.tbl_abbr)
        self.alias_inverse = {}
        for k, v in self.alias_mapping.items():
            self.alias_inverse[v] = k
        self.data_list = []

    def build_single_table_object(self, src_meta, selected_columns):
        """
        {Description}

        Args:
            src_meta:
            selected_columns:
        Returns:
            return1:
            return2:
        """
        def transform_to_names(selected_columns):
            column_names = ["{}_{}".format(self.alias_mapping[tbl], col) \
                for tbl, col in selected_columns]
            return column_names
        selected_columns = transform_to_names(selected_columns)
        data_df = self.fetch_data(src_meta)
        pruned_df = self.filter_by_columns(data_df, selected_columns)

        return pruned_df, pruned_df.columns

    def fetch_data(self, src_meta):
        """
        获取具体的数据到内存(保存成列表)，根据meta的性质去选择从data_manager
        或者mv_manager中获取数据

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        data_res = None
        def single_meta_judge(meta_info):
            schema_list, filter_list = meta_info
            if len(schema_list) == 1 and len(filter_list) == 0:
                return True
            return False

        if single_meta_judge(src_meta) == True:
            tbl_name = src_meta[0][0]
            data_res = self.data_manager.load_table_with_prefix(tbl_name)        # 从data_manager中进行加载
        else:
            data_res = self.mv_manager.load_mv_from_meta(query_meta = src_meta)  # 从mv_manager中进行加载
        return data_res


    def filter_by_columns(self, data_df, selected_columns):
        """
        {Description}
        
        Args:
            data_df:
            select_columns:
        Returns:
            pruned_df:
        """
        pruned_df = pd.DataFrame([])

        def get_valid_columns(df_columns, selected_columns):
            res_columns = []
            for col in df_columns:
                if col in selected_columns:
                    res_columns.append(col)
            return res_columns

        curr_columns = get_valid_columns(data_df.columns, selected_columns)
        pruned_df = data_df[curr_columns]

        return pruned_df
# %%

class BinsBuilder(object):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, workload, data_manager_ref: data_management.DataManager, \
        mv_manager_ref: mv_management.MaterializedViewManager = None, default_split_budget = 100):
        """
        {Description}

        Args:
            workload:
            data_manager_ref:
            mv_manager_ref:
        """
        self.workload = workload
        self.data_manager = data_manager_ref
        self.mv_manager = mv_manager_ref
        self.bins_cache = {}    # 针对bins的缓存
        self.marginal_cache = {}

        self.default_split_budget = default_split_budget

    def set_default_budget(self, in_budget: int):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        self.default_split_budget = in_budget

    @utils.timing_decorator
    def construct_marginal_dict(self, bins_dict: dict):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        marginal_dict = {}
        for column, bins_list in bins_dict.items():
            tbl_name, col_name = column
            table_df = self.data_manager.load_table(tbl_name=tbl_name)

            marginal_local = grid_base.construct_marginal(\
                data_series = table_df[col_name], input_bins=bins_list)
            marginal_dict[column] = marginal_local

        return marginal_dict

    # @utils.timing_decorator
    def construct_bins_dict(self, selected_columns, split_budget = None):
        """
        构建bins_dict，所有的selected_columns暂时看作是独立的
        TODO:这里存在的一个问题是NULL需要进行解决，暂时看来影响不大，后续待观察

        Args:
            selected_columns:
            split_budget:
        Returns:
            bins_dict:
        """
        bins_dict = {}

        if split_budget is None:
            split_budget = self.default_split_budget

        for tbl_name, col_name in selected_columns:
            if (tbl_name, col_name, split_budget) in self.bins_cache.keys():
                # 如果已经算过了，直接从cache中取数据
                bins_dict[(tbl_name, col_name)] = self.bins_cache[(tbl_name, col_name, split_budget)]
            else:
                table_df = self.data_manager.load_table(tbl_name=tbl_name)
                # 导入column对应的data series
                # print(f"construct_bins_dict: tbl_name = {tbl_name}. col_name = {col_name}.")
                curr_bins = grid_base.construct_bins(table_df[col_name], \
                    bin_num = split_budget)
                bins_dict[(tbl_name, col_name)] = curr_bins
                # 保存结果到cache中
                self.bins_cache[(tbl_name, col_name, split_budget)] = curr_bins
            
        return bins_dict


    def construct_reverse_dict(self, bins_dict):
        """
        {Description}

        Args:
            bins_dict:
        Returns:
            reverse_dict:
        """
        reverse_dict = {}

        for column, bins_list in bins_dict.items():
            reverse_local = {}
            for idx, val in enumerate(bins_list):
                reverse_local[val] = idx
            reverse_dict[column] = reverse_local

        return reverse_dict

# %%

def get_bins_builder_by_workload(workload):
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    data_manager = data_management.DataManager(wkld_name=workload)
    mv_manager = mv_management.MaterializedViewManager(workload=workload)
    bins_builder = BinsBuilder(workload, data_manager, mv_manager)
    return bins_builder

# %%


class MultiTableBuilder(object):
    """
    多表

    Members:
        field1:
        field2:
    """

    def __init__(self, workload: str, data_manager_ref: data_management.DataManager, \
        mv_manager_ref: mv_management.MaterializedViewManager, dynamic_config:dict = {}):
        """
        {Description}

        Args:
            workload:
            data_manager_ref:
            mv_manager_ref:
        """
        self.workload = workload
        self.data_manager = data_manager_ref
        self.mv_manager = mv_manager_ref

        # 新增mv_builder，专门处理多表mv生成的问题
        self.mv_builder = mv_management.MaterializedViewBuilder(workload=workload, data_manager_ref=data_manager_ref, 
                                                                mv_manager_ref=mv_manager_ref)
        
        self.alias_mapping = deepcopy(data_manager_ref.tbl_abbr)
        self.alias_inverse = {}
        for k, v in self.alias_mapping.items():
            self.alias_inverse[v] = k
        self.data_list = []

        # bins相关的信息
        self.bins_global = {}
        self.marginal_global = {}

        # 动态生成mv的相关配置信息
        if dynamic_config == {}:
            self.dynamic_config = {
                "time": 5       # 
            }
        else:
            self.dynamic_config = dynamic_config


    def load_grid_info(self, bins_dict: dict, marginal_dict: dict):
        """
        加载grid相关的信息
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        for k, v in bins_dict.items():
            self.bins_global[k] = v
        
        for k, v in marginal_dict.items():
            self.marginal_global[k] = v

        return self.bins_global, self.marginal_global

    def build_materialized_view_object(self, mv_meta, selected_columns, apply_dynamic = True, cond_bound_dict = None):
        """
        创建物化视图对象
        
        Args:
            mv_meta:
            selected_columns: 被选择的列
        Returns:
            mv_object:
            res_meta:
        """
        if len(mv_meta[0]) <= 2 or apply_dynamic == False:
            # print("build_materialized_view_object: call build_mv_on_meta. mv_meta = {}.".format(mv_meta))
            # 两表join下，不应用dynamic的机制
            mv_object = self.mv_builder.build_mv_on_meta(target_meta=mv_meta)
            return mv_object, mv_meta
        elif cond_bound_dict is not None:
            # print("build_materialized_view_object: call build_dynamic_mv_on_bound. mv_meta = {}.".format(mv_meta))
            # 构建mv时的限制条件
            constraint = {
                "time": 10,
                "cardinality": 1e7
            }
            mv_object, extra_meta_info, _ = self.mv_builder.build_dynamic_mv_on_bound(\
                target_meta=mv_meta, cond_bound_dict=cond_bound_dict, constraint=constraint, use_cache=True)
            # mv_object, extra_meta_info, _ = self.mv_builder.build_dynamic_mv_on_bound(\
            #     target_meta=mv_meta, cond_bound_dict=cond_bound_dict, constraint=constraint)
            print("build_materialized_view_object: extra_meta_info = {}.".format(extra_meta_info))
            res_meta = mv_management.meta_copy(mv_meta)
            for item in extra_meta_info:
                res_meta[1].append(item)

            # 打印meta的限制信息
            print("orgin_meta = {}. res_meta = {}.".format(mv_meta, res_meta))
            return mv_object, res_meta
        else:
            # print("build_materialized_view_object: call build_dynamic_mv_on_constraint. mv_meta = {}.".format(mv_meta))
            
            # 三表join下，自动应用dynamic的机制
            grid_config, growth_config, constraint = {}, {}, {}
            # 填充grid_config以及growth_config的内容
            # TODO: 针对config进行优化，具体来说，如果一个表存在多个column，那么可以考虑选择最均匀的column
            #       来进行切分的操作
            for compound_column in selected_columns:
                # print("build_materialized_view_object")
                origin_column = query_construction.parse_compound_column_name(col_str=compound_column, workload=self.workload)
                grid_config[origin_column] = self.bins_global[origin_column], self.marginal_global[origin_column]
                growth_config[origin_column] = (0, 1/16, 2)     # (start_idx, ratio, rate)，目前全部设置成默认的

            # 出于测试的迭代需求，时间限制到5s
            # constraint = {
            #     "time": 5,
            #   # "time": 15, # 时间限制到15s
            # }
            constraint = self.dynamic_config

            mv_object, extra_meta_info, _ = self.mv_builder.build_dynamic_mv_on_constraint(target_meta=mv_meta,
                grid_config=grid_config, growth_config=growth_config, constraint=constraint)

            print("build_materialized_view_object: extra_meta_info = {}.".format(extra_meta_info))
            res_meta = mv_management.meta_copy(mv_meta)
            for item in extra_meta_info:
                res_meta[1].append(item)

            # 打印meta的限制信息
            print("orgin_meta = {}. res_meta = {}.".format(mv_meta, res_meta))
            return mv_object, res_meta
        

    def build_joined_tables_object(self, src_meta_list, selected_columns, \
            apply_dynamic: bool = True, filter_columns = True, cond_bound_dict = None):
        """
        构造连接tables的对象

        Args:
            src_meta_list: 元数据的列表
            selected_columns: 选择划分的列
            apply_dynamic: 是否应用动态生成机制
        Returns:
            data_df:
            column_list: value_cnt_arr对应的column信息
            merged_meta: 合并的meta信息
            res_meta_list: 处理以后新的元数据列表
        """
        res_meta_list = []

        def transform_to_names(selected_columns):
            column_names = ["{}_{}".format(self.alias_mapping[tbl], col) \
                for tbl, col in selected_columns]
            return column_names
        
        selected_columns = transform_to_names(selected_columns)

        self.data_list, res_meta_list = self.fetch_data_list(src_meta_list, selected_columns\
            =selected_columns, apply_dynamic=apply_dynamic, cond_bound_dict=cond_bound_dict)
        
        # for idx, (local_df, local_meta) in enumerate(zip(self.data_list, res_meta_list)):
        #     print("build_joined_tables_object: idx = {}. local_df = {}. local_meta = {}.".format(idx, local_df, local_meta))
        # src_meta_list此时已被res_meta_list完成替换

        if filter_columns == True:
            # 根据columns的信息过滤掉不必要的数据
            self.data_list = self.filter_by_columns(res_meta_list, self.data_list, selected_columns)

        # 顺序下标列表
        index_list = self.determine_join_order(res_meta_list)

        def column_in_meta(in_column, in_meta):
            # print("in_column = {}.".format(in_column))
            flag = False
            for item in in_meta[0]:
                if in_column[0] == item: 
                    # 只要表名匹配上就可以
                    flag = True
                    break
            return flag
        
        res_data_df, merged_meta = self.execute_join_operation(src_meta_list = res_meta_list,\
            data_list = self.data_list, order_list = index_list)

        # 消去不必要的列，这一步需要优化
        # data_df, column_list = self.eliminate_join_columns(joined_df = res_data_df, \
        #     selected_columns = selected_columns)
        data_df, column_list = res_data_df, selected_columns

        # print("build_joined_tables_object: selected_columns = {}. column_list = {}.".\
        #     format(selected_columns, column_list))
        
        return data_df, column_list, merged_meta, res_meta_list

    def infer_to_join_columns(self, src_meta_list):
        """
        推断出所有参与了join的列
        
        Args:
            src_meta_list:
        Returns:
            join_columns:
        """
        join_columns = []

        def update_columns(col_info):
            if isinstance(col_info, str):
                join_columns.append(col_info)
            elif isinstance(col_info, (list, tuple)):
                for item in col_info:
                    join_columns.append(item)
            else:
                raise TypeError("update_columns: type(col_info) = {}".format(type(col_info)))

        for i in range(len(src_meta_list)):
            for j in range(i + 1, len(src_meta_list)):
                meta1, meta2 = src_meta_list[i], src_meta_list[j]
                left_on, right_on = self.data_manager.infer_join_conditions(\
                    left_meta = meta1, right_meta = meta2)
                print("infer_to_join_columns: meta1 = {}. meta2 = {}. left_on = {}. right_on = {}.".\
                    format(meta1, meta2, left_on, right_on))
                if len(left_on) > 0 and len(right_on) > 0:
                    # 转化成alias_column的形式
                    update_columns(left_on)
                    update_columns(right_on)

        return list(set(join_columns))


    def determine_join_order(self, src_meta_list):
        """
        确定当前对象的连接顺序
        
        Args:
            src_meta_list: 元信息列表
            select_columns: 被选择的列
        Returns:
            index_list:
        """
        index_list = []
        index_all = range(len(src_meta_list))

        # 首先判断两表之前是否可以连通，然后每一次都加一张可以和当前表连接的表
        def build_join_graph(src_meta_list):
            edge_set = set()
            for i in range(len(src_meta_list)):
                for j in range(i + 1, len(src_meta_list)):
                    left_meta, right_meta = src_meta_list[i], src_meta_list[j]
                    left_on, right_on = self.data_manager.infer_join_conditions(\
                        left_meta, right_meta)
                    if left_on is not None and len(left_on) > 0:
                        edge_set.add((i, j))
                        edge_set.add((j, i))
            return edge_set

        edge_set = build_join_graph(src_meta_list)

        def can_connect(idx, edge_set):
            flag = False
            for src_idx in index_list:
                if (src_idx, idx) in edge_set:
                    flag = True
                    break
            return flag

        def find_next_obj_index(index_list):
            for idx in index_all:
                if idx in index_list:
                    # 已经在里面了，直接跳过
                    continue
                elif can_connect(idx, edge_set=edge_set) == True:
                    # 判断能否join起来
                    return idx
            # 全部失败，理论上不可能存在的情况
            return -1

        # 直接添加第一个对象
        index_list.append(0)
        for _ in src_meta_list[1:]:
            # meta_aggregation
            next_obj_index = find_next_obj_index(index_list)
            index_list.append(next_obj_index)
        return index_list


    def fetch_data_list(self, src_meta_list, selected_columns, apply_dynamic = True, cond_bound_dict = None):
        """
        获取具体的数据到内存(保存成列表)，根据meta的性质去选择从data_manager
        或者mv_manager中获取数据

        Args:
            src_meta_list:
            selected_columns:
            apply_dynamic: 是否动态生成mv
        Returns:
            data_list:
            res_meta_list:
        """
        data_list = []
        res_meta_list = []

        def single_meta_judge(meta_info):
            schema_list, filter_list = meta_info
            if len(schema_list) == 1 and len(filter_list) == 0:
                return True
            return False

        for curr_meta in src_meta_list:
            if single_meta_judge(curr_meta) == True:
                tbl_name = curr_meta[0][0]
                # 从data_manager中进行加载，保证可以成功
                data_list.append(self.data_manager.load_table_with_prefix(tbl_name))  
                # 添加元信息
                res_meta_list.append(curr_meta)      
            else:
                # 从mv_manager中进行加载，可能会不成功
                # load_mv可能会出现失败的情况，这种时候应该考虑
                # 先从单表构造object，然后在load进来
                mv_obj = self.mv_manager.load_mv_from_meta(query_meta = curr_meta)
                if mv_obj is None:
                    if len(curr_meta[0]) == 1:
                        # print("mv_builder.fetch_data_list: generate single-table mv_obj")
                        tbl_name = curr_meta[0][0]
                        data_df = self.data_manager.load_table_with_prefix(tbl_name=tbl_name)
                        out_df = conditions_apply(in_df=data_df, filter_list=curr_meta[1])
                        res_meta_list.append(curr_meta)
                        data_list.append(out_df)
                        continue
                    else:
                        # print("mv_builder.fetch_data_list: generate multi-table mv_obj")
                        mv_obj, res_meta = self.build_materialized_view_object(mv_meta=curr_meta, \
                            selected_columns=selected_columns, apply_dynamic=apply_dynamic, cond_bound_dict=cond_bound_dict)
                        res_meta_list.append(res_meta)
                        
                        # 保存mv对象
                        # self.mv_manager.generate_external_mv(ext_df=mv_obj, ext_meta=res_meta)
                        # raise ValueError("mv_obj is None and schema_list > 1")
                else:
                    print("mv_builder.fetch_data_list: find target mv_obj")
                    res_meta_list.append(curr_meta)
                data_list.append(mv_obj) # 从mv_manager中进行加载
        
        # # 观察前后的元信息是否发生变化
        # print("fetch_data_list: src_meta_list = {}.".format(src_meta_list))
        # print("fetch_data_list: res_meta_list = {}.".format(res_meta_list))

        return data_list, res_meta_list  # 额外的meta信息


    def eliminate_join_columns(self, joined_df, selected_columns):
        """
        {Description}
        
        Args:
            joined_df:
            selected_columns:
        Returns:
            pruned_df:
            column_list:
        """
        # pruned_df = pd.DataFrame([])
        filter_columns = []

        for col in joined_df.columns:
            if col in selected_columns:
                filter_columns.append(col)

        # filter_columns在这里表示列的顺序
        pruned_df = joined_df[filter_columns]
        return pruned_df, filter_columns

    def execute_join_operation(self, src_meta_list, data_list, order_list):
        """
        {Description}
        
        Args:
            src_meta_list: 元信息的列表
            data_list: 数据列表
            order_list: 连接顺序列表
        Returns:
            all_tables_df:
            all_tables_meta:
        """
        # all_tables_df = pd.DataFrame([])
        # print("data_list_types = {}".format([type(d) for d in data_list]))

        first_idx = order_list[0]
        all_tables_df = data_list[first_idx]    # 代表已经被join起来的对象，初始为第一张表
        all_tables_meta = src_meta_list[first_idx]

        foreign_mapping = workload_spec.get_spec_foreign_mapping(workload_name=self.workload)
        for curr_index in order_list[1:]:
            curr_meta = src_meta_list[curr_index]

            # 推断两边join的条件
            left_on, right_on = self.data_manager.infer_join_conditions(\
                left_meta = all_tables_meta, right_meta = curr_meta, foreign_mapping=foreign_mapping)
                  
            all_tables_meta = mv_management.meta_merge(all_tables_meta, curr_meta)
            curr_df = data_list[curr_index]     # 获取待join的dataframe

            all_tables_df = self.join_one_step(left_obj = all_tables_df, \
                right_obj = curr_df, left_on = left_on, right_on = right_on)

        return all_tables_df, all_tables_meta
    
    def join_one_step(self, left_obj: pd.DataFrame, right_obj: pd.DataFrame, left_on, right_on, spec = None):
        """
        进行一步的连接，spec是考虑join结果过大
        
        Args:
            left_obj:
            right_obj:
            left_on:
            right_on:
        Returns:
            joined_obj:
        """
        # print("join_one_step: left_on = {}. right_on = {}.".format(left_on, right_on))
        # print("left_obj = {}".format(left_obj.head(n = 2)))
        # print("right_obj = {}".format(right_obj.head(n = 2)))

        # 处理left_obj和right_obj，执行join_key上的dropna操作
        left_obj = left_obj.dropna(how="any", subset=left_on)
        right_obj = right_obj.dropna(how="any", subset=right_on)
        
        joined_obj = pd.merge(left = left_obj, right = right_obj, how="inner", 
            left_on=left_on, right_on=right_on)

        return joined_obj

    def filter_by_columns(self, src_meta_list, data_list, selected_columns):
        """
        {Description}
        
        Args:
            data_list:
            select_columns:
        Returns:
            pruned_data_list:
        """
        pruned_data_list = []
        # self.data_manager.get_join_columns(schema)
        join_columns = self.infer_to_join_columns(src_meta_list)

        print("filter_by_columns: join_columns = {}. selected_columns = {}.".\
            format(join_columns, selected_columns))
        def get_valid_columns(df_columns, join_columns, selected_columns):
            res_columns = []
            for col in df_columns:
                if col in join_columns or col in selected_columns:
                    res_columns.append(col)
            return res_columns

        for df_data in data_list:
            curr_columns = get_valid_columns(df_data.columns, \
                join_columns, selected_columns)
            pruned_data_list.append(df_data[curr_columns])

        return pruned_data_list

# %%

def get_table_builder_by_workload(workload, data_manager = None, mv_manager = None, dynamic_config = {}):
    """
    {Description}

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    if data_manager is None:
        data_manager = data_management.DataManager(wkld_name=workload)

    if mv_manager is None:
        mv_manager = mv_management.MaterializedViewManager(workload=workload)

    table_builder = MultiTableBuilder(workload=workload, data_manager_ref=data_manager, \
                                      mv_manager_ref=mv_manager, dynamic_config=dynamic_config)
    return table_builder

# %%

def get_bins_builder_by_workload(workload, data_manager = None, mv_manager = None):
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    if data_manager is None:
        data_manager = data_management.DataManager(wkld_name=workload)

    if mv_manager is None:
        mv_manager = mv_management.MaterializedViewManager(workload=workload)

    bins_builder = BinsBuilder(workload=workload, \
        data_manager_ref=data_manager, mv_manager_ref=mv_manager)
    return bins_builder

# %% bins_list相关的功能函数

def random_range_on_bins(bins, length = None):
    """
    在bins上进行随机的采样
    
    Args:
        bins:
        length:
    Returns:
        start_val:
        end_val:
    """
    total = len(bins)
    # 在一个bins上随机选择一段长度
    if length is not None:
        start, end = 0, 0
        start = np.random.randint(0, total - length)    # 随机选择起始位置
        end = start + length
        return bins[start] + 1, bins[end]
    else:
        start, end = np.random.randint(0, total), np.random.randint(0, total)
        start, end = min(start, end), max(start, end)
        return bins[start] + 1, bins[end]
    
# %%

def get_marginal_card(start_val, end_val, reverse_dict, marginal_list):
    """
    {Description}

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    start_idx, end_idx = utils.predicate_location(reverse_dict, start_val, end_val)
    return int(np.sum(marginal_list[start_idx: end_idx]))
