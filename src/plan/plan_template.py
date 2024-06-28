#!/usr/bin/env python
# -*- coding: utf-8 -*-


# %%


import numpy as np
import pandas as pd
from os.path import join as p_join
from matplotlib import pyplot as plt
import json, os, time, re, math
import pickle
from copy import deepcopy

from multiprocessing import Pool, Process
from collections import defaultdict

import hashlib, shutil 
from utility import generator, utils, common_config
from data_interaction import data_management, mv_management
from grid_manipulation import grid_construction, grid_preprocess, grid_advance, grid_analysis
from query import query_construction, query_exploration, ce_injection
import base
from itertools import product, combinations
from plan import node_query, plan_analysis
from functools import partial, reduce
import operator, shutil
from algo import bayesian_optimazition
from utility.common_config import get_template_meta_path

def clean_template_dir(workload, method, signature = None, \
        common_dir = "/home/lianyuan/Research/CE_Evaluator/intermediate"):
    """
    清理模版路径

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    if signature is None or len(signature) == 0:
        template_dir = p_join(common_dir, workload, "template_obj", method)
    else:
        # 构造模版路径
        template_dir = p_join(common_dir, workload, "template_obj", method, signature)
        
    shutil.rmtree(template_dir)
    # utils.dump_json()


# %%

@utils.timing_decorator
def get_template_input_elements(workload, selected_columns, split_budget = 100):
    """
    {Description}
    
    Args:
        workload: 
        selected_columns: 
        split_budget:
    Returns:
        data_manager:
        mv_manager:
        bins_builder: 
        bins_dict:
        marginal_dict:
        reverse_dict:
        query_ctrl: 
        table_builder:
    """
    data_manager = data_management.DataManager(wkld_name=workload)
    mv_manager = mv_management.MaterializedViewManager(workload=workload)

    bins_builder = grid_preprocess.BinsBuilder(workload = workload, \
                        data_manager_ref=data_manager, mv_manager_ref=mv_manager)
    bins_builder.set_default_budget(split_budget)

    bins_dict = bins_builder.construct_bins_dict(selected_columns=selected_columns, split_budget=split_budget)
    marginal_dict = bins_builder.construct_marginal_dict(bins_dict=bins_dict)
    reverse_dict = bins_builder.construct_reverse_dict(bins_dict=bins_dict)

    query_ctrl = query_exploration.QueryController(workload=workload)

    table_builder = grid_preprocess.get_table_builder_by_workload(
        workload=workload, data_manager=data_manager, mv_manager=mv_manager
    )

    return data_manager, mv_manager, bins_builder, bins_dict, \
        marginal_dict, reverse_dict, query_ctrl, table_builder

# %%

class TemplateManager(object):
    """
    一批模版的管理者

    Members:
        field1:
        field2:
    """

    def __init__(self, workload, ce_handler: base.BaseHandler = None, inter_path = \
        "/home/lianyuan/Research/CE_Evaluator/intermediate/stats/template_obj", dump_strategy = "remain", \
        split_budget = 100, dynamic_config = {}):
        """
        {Description}

        Args:
            workload:
            ce_handler:
            inter_path:
            dump_strategy: 当结果路径冲突的时候，系统处理的策略
            dynamic_config: 动态mv的生成配置
        """
        self.split_budget = split_budget
        self.dynamic_config = dynamic_config

        self.workload = workload
        if ce_handler is None:
            # 获得内置的处理器
            self.ce_handler = ce_injection.get_internal_handler(workload=workload)
        else:
            self.ce_handler = ce_handler

        self.alias_mapping = query_construction.abbr_option[workload]
        self.alias_inverse = {}
        for k, v in self.alias_mapping.items():
            self.alias_inverse[v] = k

        self.inter_path = inter_path
        # 创建结果路径
        res_path = self.inter_path

        def get_new_path():
            origin_path = self.inter_path
            new_path, cnt = "", 0
            path_tmpl = "{}_{}"
            while True:
                cnt += 1
                new_path = path_tmpl.format(origin_path, cnt)
                if os.path.isdir(new_path) == False: 
                    # 判断路径是否已经存在了
                    break
            return new_path
        
        print(f"TemplateManager.__init__: res_path = {res_path}. dump_strategy = {dump_strategy}.")
        if dump_strategy == "rename":
            # 移动之前的数据
            new_path = get_new_path()
            shutil.move(src=res_path, dst=new_path)
            # 重新创建目录
            os.makedirs(res_path, exist_ok=True)
        elif dump_strategy == "overwrite":
            # 删除之前的数据
            shutil.rmtree(res_path)
            # 重新创建目录
            os.makedirs(res_path, exist_ok=True)
        elif dump_strategy == "remain":
            # 所有内容维持不变
            pass
        else:
            raise ValueError("TemplateManager Unsupported Strategy: {}".format(dump_strategy))
        
        self.template_dict = {}
        self.data_manager = data_management.DataManager(wkld_name=workload)
        self.mv_manager = mv_management.MaterializedViewManager(workload=workload)
        self.bins_builder = grid_preprocess.BinsBuilder(workload = workload, 
            data_manager_ref=self.data_manager, mv_manager_ref=self.mv_manager)

        self.query_ctrl = query_exploration.QueryController(workload=workload)
        self.table_builder = grid_preprocess.get_table_builder_by_workload(
            workload=workload, data_manager=self.data_manager, mv_manager=self.mv_manager)

    def clean_directory(self,):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        print("clean_directory: call function.")
        res_path = self.inter_path
        # 删除之前的数据
        shutil.rmtree(res_path)
        # 重新创建目录
        os.makedirs(res_path, exist_ok=True)

    def set_ce_handler(self, external_handler):
        """
        设置外部的handler
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        if isinstance(external_handler, str):
            # 输入是关键字的case
            # 关键字全部转成小写
            external_handler = external_handler.lower()
            if external_handler == "internal":
                self.ce_handler = ce_injection.PGInternalHandler(workload=self.workload)
            else:
                self.ce_handler = ce_injection.get_ce_handler_by_name(\
                    workload=self.workload, ce_type=external_handler)
            # elif external_handler == "deepdb":
            #     raise NotImplementedError("external_handler == \"deepdb\"未实现")
            # elif external_handler == "neurocard":
            #     raise NotImplementedError("external_handler == \"neurocard\"未实现")
        else:
            # 输入是ce_handler实例的情况
            self.ce_handler = external_handler

    def create_single_template(self, query_meta, selected_columns):
        """
        创建单个模版
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        bins_builder = self.bins_builder
        bins_dict = bins_builder.construct_bins_dict(selected_columns=selected_columns, split_budget=self.split_budget)
        reverse_dict = bins_builder.construct_reverse_dict(bins_dict=bins_dict)
        marginal_dict = bins_builder.construct_marginal_dict(bins_dict=bins_dict)

        template_plan = TemplatePlan(workload=self.workload, query_meta=query_meta, 
            ce_handler=self.ce_handler, bins_dict=bins_dict, marginal_dict=marginal_dict, 
            reverse_dict=reverse_dict, query_ctrl_ref=self.query_ctrl, 
            table_builder_ref=self.table_builder, bins_builder=self.bins_builder, split_budget=self.split_budget)

        return template_plan
    
    def create_batch_templates(self, parameter_list, wait_complete = True):
        """
        {Description}
    
        Args:
            parameter: 入参的列表
            arg2:
        Returns:
            template_dict: 模板计划的字典
            output_path_dict: 输出路径
        """

        template_obj_list = []
        for query_meta, selected_columns, mode in parameter_list:
            template_obj_list.append((self.workload, query_meta, selected_columns, mode))

        result_list, template_dict, output_path_dict = self.parallel_template_creation(\
            template_obj_list=template_obj_list, global_config=self.dynamic_config, wait_complete=wait_complete)
        return template_dict, output_path_dict


    def create_templates_under_cond_bound(self, parameter_list, cond_bound_list, wait_complete = True):
        """
        在cond_bound的条件下创建templates
    
        Args:
            parameter_list:
            cond_bound_list:
        Returns:
            template_dict: 
            output_path_dict:
        """
        template_obj_list = []
        for (query_meta, selected_columns, mode), cond_bound_dict in zip(parameter_list, cond_bound_list):
            # 添加五个变量作为入参
            template_obj_list.append((self.workload, query_meta, selected_columns, mode, cond_bound_dict))
            # print(f"create_templates_under_cond_bound: query_meta = {query_meta}. cond_bound_dict = {cond_bound_dict}")
            
        result_list, template_dict, output_path_dict = self.parallel_template_creation(\
            template_obj_list=template_obj_list, global_config=self.dynamic_config, wait_complete=wait_complete)
        return template_dict, output_path_dict
    

    def create_grids_on_template(self, template_key):
        """
        在选择的模板上创建格点

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        template_curr: TemplatePlan = self.template_dict[template_key]
        template_curr.process_top_query()
        template_curr.process_all_single_tables()
        template_curr.process_all_subqueries()

        return template_curr

    def parallel_template_creation(self, template_obj_list, global_config, wait_complete = True):
        """
        并行的创建多个模版
    
        Args:
            template_obj_list: 模版对象组成的列表
            global_config: 全局的配置信息
            wait_complete: 
        Returns:
            result_list: 结果列表
            template_dict: 模版字典
        """
        python_path = "/home/lianyuan/anaconda3/envs/CE_Evaluator/bin/python"
        script_path = "/home/lianyuan/Research/CE_Evaluator/src/plan/basic_execution_unit.py"
        args_config = {
            "--input_path": "",
            "--output_path": "",
            "--config": str(global_config),
            # 2024-03-13: 修改config_str的构造方式
            # "--config": json.dumps(global_config).replace("\"", "\'"),
            "--split_budget": self.split_budget
        }
        execution_template = "{python_path} {script_path} {args_str}"
        num_process = 10    # 并行程度设置
        cmd_list = []
        alias_mapping = query_construction.abbr_option[self.workload]
        common_dir_path = self.inter_path

        output_path_dict = {}
        shared_ce_handler = ce_injection.get_internal_handler(workload=self.workload)  # 

        for template_obj in template_obj_list:
            # 
            if len(template_obj) == 4:
                workload, query_meta, selected_columns, mode = template_obj
            elif len(template_obj) == 5:
                workload, query_meta, selected_columns, mode, cond_bound_dict = template_obj
            else:
                raise ValueError(f"parallel_template_creation: len(template_obj) = {len(template_obj)}")
            
            curr_template_key = template_repr_key(query_meta=query_meta, \
                selected_columnns=selected_columns, alias_mapping=alias_mapping)
            input_name = "{}_input.pkl".format(curr_template_key)
            output_name = "{}_output.pkl".format(curr_template_key)
            # 输入路径和输出路径
            input_path = p_join(common_dir_path, input_name)
            output_path = p_join(common_dir_path, output_name)

            # print(f"parallel_template_creation: input_path = {input_path}")
            # print(f"parallel_template_creation: output_path = {output_path}")

            args_config["--input_path"] = input_path
            args_config["--output_path"] = output_path
            # 路径上添加引号
            args_str = " ".join(["{} \"{}\"".format(k, v) for k, v in args_config.items()])

            local_execution_cmd = execution_template.format(python_path=python_path,
                                                            script_path=script_path,
                                                            args_str=args_str)
            
            output_path_dict[curr_template_key] = output_path

            # 判断输出结果的文件是否存在
            if os.path.isfile(output_path) == False:
                # 保存intermediate的变量
                utils.dump_pickle(template_obj, input_path)
                # 添加执行命令
                # cmd_list.append(local_execution_cmd)
                cmd_list.append((local_execution_cmd, output_path))
            else:
                print("输出结果已经存在，路径为 {}".format(output_path))
        
        # 
        utils.clean_mv_cache(workload=self.workload)

        # print("cmd_list = \n{}".format("\n".join([str(item) for item in cmd_list])))
        if wait_complete == True:
            # 2024-03-12: 等待生成过程结束
            # 并行执行模版生成程序
            with Pool(num_process) as p:
                result_list = p.map(exec_script, cmd_list)

            # 读取intermidate的结果
            valid_cnt, output_path_valid = 0, {}
            for k, output_path in output_path_dict.items():
                local_template_plan: TemplatePlan = utils.load_pickle(output_path)
                if local_template_plan is None:
                    # 考虑创建可能失败的问题，直接跳过
                    continue

                local_template_plan.set_ce_handler(external_handler=shared_ce_handler)  # 设置共有的ce_handler
                self.template_dict[k] = local_template_plan
                output_path_valid[k] = output_path
                valid_cnt += 1

            if valid_cnt == 0:
                raise ValueError("parallel_template_creation: none template creation!")
            else:
                print(f"total_template_num = {len(output_path_dict)}. create_template_num = {valid_cnt}.")

            # return result_list, self.template_dict, output_path_dict
            return result_list, self.template_dict, output_path_valid
        else:
            # 2024-03-12: 不等待生成过程结束，直接返回
            print(f"parallel_template_creation: wait_complete = False. len(cmd_list) = {len(cmd_list)}.")
            # with Pool(num_process) as p:
            #   
            #     p.map_async(exec_script, cmd_list)
            p = Process(target = exec_script_parallel, args = (cmd_list, ))
            p.start()
            result_list = [True for _ in cmd_list]  # 默认执行可以成功
            return result_list, self.template_dict, output_path_dict


    def save_current_state(self,):
        """
        保存当前的状态
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        dump_path = p_join(self.inter_path, "manager_state.pkl")
        utils.dump_pickle(res_obj=self.template_dict, data_path=dump_path)

    def load_historical_state(self, signature):
        """
        加载历史的状态
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        load_path = p_join(self.inter_path, signature, "manager_state.pkl")
        self.template_dict = utils.load_pickle(data_path=load_path)

    def load_historical_template(self, template_path):
        """
        加载单个历史上的模版
        
        Args:
            template_path:
            arg2:
        Returns:
            res1:
            res2:
        """
        template_obj: TemplatePlan = utils.load_pickle(template_path)
        alias_mapping = self.alias_mapping
        query_meta = template_obj.query_meta
        selected_columns = template_obj.selected_columns
        template_key = template_repr_key(query_meta=query_meta, \
            selected_columnns=selected_columns, alias_mapping=alias_mapping)
        self.template_dict[template_key] = template_obj
        return template_obj

    def load_historical_templates_in_directory(self, template_dir = "/home/lianyuan/CE_Evaluator/intermediate/template_obj"):
        """
        从一个目录下加载所有的template并保存
        
        Args:
            template_dir:
            arg2:
        Returns:
            res1:
            res2:
        """
        file_list = os.listdir(template_dir)
        for f_name in file_list:
            if f_name.endswith(".pkl"):
                curr_path = p_join(template_dir, f_name)
                print("curr_path = {}.".format(curr_path))
                template_obj = utils.load_pickle(curr_path)
                if isinstance(template_obj, TemplatePlan) == False:
                    continue
                template_obj: TemplatePlan = template_obj
                alias_mapping = self.alias_mapping
                query_meta = template_obj.query_meta
                selected_columns = template_obj.selected_columns
                template_key = template_repr_key(query_meta=query_meta, \
                    selected_columnns=selected_columns, alias_mapping=alias_mapping)
                self.template_dict[template_key] = template_obj

    def template_status_evaluation(self,):
        """
        评价template的大小
        
        Args:
            None
        Returns:
            res1:
            res2:
        """
        for k, v in self.template_dict.items():
            print("k = {}.".format(k))
            v.show_grid_info()


def exec_script(input_params: tuple):
    """
    {Description}

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    try:
        cmd, out_path = input_params
        print("cmd = {}.".format(cmd))
        # 2024-03-13: 休眠程序，用做测试
        time.sleep(3)
        os.system(cmd)

        if os.path.isfile(out_path) == False:
            # 发现找不到output，再次执行生成template的命令
            os.system(cmd)
            # 执行完之后再次判断
            if os.path.isfile(out_path) == False:
                print(f"exec_script: create template error. out_path = {out_path}")
    except Exception as e:
        print(f"exec_script: meet error. {e}.")

    return True



def exec_script_parallel(input_list):
    # 并发创建模版
    with Pool(common_config.template_parallel_num) as pool:
        pool.map(exec_script, input_list)

# %%

def template_repr_key(query_meta, selected_columnns, alias_mapping) -> str:
    """
    将模板转成字典的键，明文保留schema和selected_columns的信息，
    然后其他所有信息转Hash

    Args:
        query_meta: 查询的元信息
        selected_columnns: 选择的列
        alias_mapping: 别名映射
    Returns:
        repr_key:
    """
    sorted_schema = sorted(query_meta[0])
    sorted_columns = sorted(list(selected_columnns))
    sorted_filters = sorted(query_meta[1])

    repr_key_template = "{schema_str}#{column_str}#{hash_str}"
    schema_str = "&".join(sorted_schema)
    column_str = "&".join("{}_{}".format(alias_mapping[i], j) for i,j in sorted_columns)
    m = hashlib.sha256()
    m.update(str(sorted_filters).encode("utf-8"))

    hash_str = m.hexdigest()[:6]
    repr_key = repr_key_template.format(schema_str=schema_str,
                column_str=column_str, hash_str=hash_str)
    return repr_key


def convert2df_col(alias_mapping, column_list):
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    df_col_list = ["{}_{}".format(alias_mapping\
        [col[0]], col[1]) for col in column_list]
    
    return df_col_list

# %%

class TemplatePlan(object):
    """
    一个模版的查询计划实例，包含了顶层查询的模版，也包含了所有子查询的模版

    Members:
        field1:
        field2:
    """

    # @utils.timing_decorator
    def __init__(self, workload, query_meta, mode, bins_dict:dict, reverse_dict:dict, 
        marginal_dict:dict, query_ctrl_ref: query_exploration.QueryController, 
        table_builder_ref: grid_preprocess.MultiTableBuilder, ce_handler: base.BaseHandler, 
        bins_builder: grid_preprocess.BinsBuilder, extra_info: dict = {}, split_budget = 100):
        """
        {Description}

        Args:
            workload: 负载
            query_meta:
            mode: 
            bins_dict:
            reverse_dict:
            marginal_dict:
            query_ctrl_ref:
            table_builder_ref: 
            extra_info: 额外的信息注入，包括列的限制
        """
        assert mode in ("over-estimation", "under-estimation")
        assert workload in ("job", "stats", "dsb")

        cond_bound_dict = extra_info.get('cond_bound', None)
        self.workload, self.mode = workload, mode

        # query_meta在这里是输入的meta信息，但是存在的问题是
        self.query_meta = query_meta
        self.extra_column_list = []

        # 处理alias相关信息
        self.alias_mapping = query_construction.abbr_option[self.workload]
        self.alias_inverse = {}

        for k, v in self.alias_mapping.items():
            self.alias_inverse[v] = k

        # 对于placeholder做提取处理
        out_meta, selected_columns = self.process_meta(in_meta = query_meta)
        self.top_meta = out_meta
        self.selected_columns = selected_columns    # 选择切分的列

        # 查询处理对象赋值
        self.query_ctrl = query_ctrl_ref
        self.ce_handler = ce_handler
        self.table_builder = table_builder_ref
        self.bins_builder = bins_builder

        # 加载grid相关的信息
        self.bins_global = bins_dict
        self.marginal_global = marginal_dict
        self.reverse_global = reverse_dict

        # 将grid相关信息导入到table_builder中去
        self.table_builder.load_grid_info(bins_dict=bins_dict, marginal_dict=marginal_dict)
        self.table_num = len(query_meta[0])
        self.column_list = selected_columns

        # 
        self.grid_plan_dict = {}        # 已经创建好的grid_plan
        self.grid_plan_info = {}        # grid_plan探索的结果

        # 额外列组合的集合
        self.existing_extra_columns = set()     

        self.subquery_df_dict = {}      # 子查询对应的
        self.single_table_df_dict = {}  

        self.convert2df_col = partial(convert2df_col, alias_mapping = self.alias_mapping)

        # 处理顶层查询
        self.process_top_query(cond_bound_dict)
        # 处理所有的子查询
        self.process_all_subqueries()
        # 处理所有的单表信息
        self.process_all_single_tables()

        self.infer_extra_columns()
        # print("TemplatePlan.__init__: create_default_grid_plan")
        self.create_default_grid_plan()     # 构造默认的grid_plan
        self.bind_grid_plan()               # 绑定默认的grid plan
        print("TemplatePlan.__init__: set extra_info_dict")
        self.extra_info_dict = self.default_grid_plan.extra_info_dict

        # 最小/最大的grid数目
        # self.grid_num_min, self.grid_num_max = 1000, 130000000
        self.grid_num_min, self.grid_num_max = 1000, 10000000   # 用于测试
        self.split_budget = split_budget
    

    def bind_template_plan(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        result_list = []
        for grid_plan_id, grad_plan_instance in self.grid_plan_dict.items():
            grad_plan_instance: GridPlan = grad_plan_instance
            grad_plan_instance.add_template_plan_ref(self)
            result_list.append(grid_plan_id)

        return result_list
    
    def get_mode(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        return self.mode

    def epsilon_greedy_selection(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        def linear_iter_gen(start_val, end_val, total_step):
            delta = (start_val - end_val) / total_step
            # 生成linear epsilon的迭代器
            def local_func(in_val):
                if in_val - delta <= end_val:
                    out_val = in_val
                else:
                    out_val = in_val - delta
                return out_val

            return local_func
    
        def exponential_iter_gen(start_val, end_val, total_step):
            if end_val <= 0.0:
                # 避免出现0的情况
                end_val = 1e-3
            factor = (end_val / start_val) ** (1 / total_step)
            # print("factor = {}.".format(factor))
            # 生成exponential epsilon的迭代器
            def local_func(in_val):
                if in_val * factor <= end_val:
                    out_val = in_val
                else:
                    out_val = in_val * factor

                return out_val

            return local_func
            
        if hasattr(self, "curr_epsilon") == False:
            # 变量初始化
            self.curr_epsilon = 1.0
            start_val, end_val, total_step = 1.0, 0.1, 20
            self.iter_func = exponential_iter_gen(start_val, end_val, total_step)

        # 基于历史收益，利用epsilon-greedy策略去选择
        indicator = np.random.uniform(0, 1)
        valid_id_list = [item for item in self.grid_plan_info.keys() if \
            self.grid_num_min <= self.grid_plan_info[item]['grid_num'] <= self.grid_num_max]
        
        if indicator < self.curr_epsilon:
            # 随机选grid_id
            grid_id = np.random.choice(valid_id_list)
        else:
            # 选history_rewards中最大的
            # val_pair_list = [(grid_id, self.grid_plan_info[grid_id]) for grid_id in valid_id_list]
            val_pair_list = []
            for grid_id in valid_id_list:
                local_list = self.grid_plan_info[grid_id]['history_rewards']
                if len(local_list) == 0:
                    val_pair_list.append((grid_id, 1e8))
                else:
                    val_pair_list.append((grid_id, max(local_list)))

            val_pair_list.sort(key=lambda a: a[1], reverse=True)
            grid_id = val_pair_list[0][0]

        self.curr_epsilon = self.iter_func(self.curr_epsilon)
        return grid_id
    
    def select_grid_plan(self, mode = "max_grid"):
        """
        默认选择grid_num最大的grid_plan
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        grid_id = 0
        if mode == "max_grid":
            # 直接选grid_num最多的，但其实不一定收益最高
            grid_id = max(self.grid_plan_info.keys(), key=lambda a: self.grid_plan_info[a]['grid_num'])
        elif mode == "random":
            # 在合法的grid_num范围内随机选择
            valid_id_list = [item for item in self.grid_plan_info.keys() if \
                self.grid_num_min <= self.grid_plan_info['grid_num'] <= self.grid_num_max]
            grid_id = np.random.choice(valid_id_list)
        elif mode == "history":
            grid_id = self.epsilon_greedy_selection()
            # raise NotImplementedError("select_grid_plan: history mode没有实现")
        else:
            raise ValueError(f"select_grid_plan: mode = {mode}.")

        return grid_id

    def set_grid_plan_benefit(self, grid_plan_id, benefit):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        self.grid_plan_info[grid_plan_id]["history_rewards"].append(benefit)

    def get_id_list(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        return list(self.grid_plan_info.keys())

    def create_extra_grid_plans(self, plan_num):
        """
        创建额外的grid_plan
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        target_columns_list = self.generate_extra_columns(comb_num=plan_num)
        plan_list = []
        for target_columns in target_columns_list:
            new_grid_plan = self.create_grid_plan(\
                new_column_list=target_columns, split_budget=self.split_budget)
            plan_list.append(new_grid_plan)

        return plan_list

    def construct_extra_bins(self, split_budget = 100):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        extra_bins_dict = self.bins_builder.construct_bins_dict(\
            self.extra_column_list, split_budget=split_budget)
        return extra_bins_dict

    def list_repr(self, in_list):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        return sorted(in_list)

    def eval_existence(self, column_list):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        col_list_repr = self.list_repr(column_list)
        return col_list_repr in self.existing_extra_columns


    def add_to_set(self, column_list):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        col_list_repr = self.list_repr(column_list)
        self.existing_extra_columns.add(col_list_repr)
        return self.existing_extra_columns
    
    def set_cost(self, bins_num_dict, column_list):
        return reduce(operator.mul, [bins_num_dict[i] for i in column_list], 1)
    

    def generate_all_column_list(self, bins_num_dict, input_columns, min_budget, max_budget):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        exist_cost = self.default_grid_plan.value_cnt_arr.size   # 已经存在的代价
        print(f"generate_all_column_list: exist_cost = {exist_cost}.")
        min_budget /= exist_cost
        max_budget /= exist_cost
        column_list_set = set()     # 最终的结果

        def set_cost(column_list):
            # return reduce(operator.mul, [bins_num_dict[i] for i in column_list], 1.0)
            data_list = [bins_num_dict[i] for i in column_list]
            # print(f"set_cost: data_list = {data_list}")
            return np.prod(data_list)
        
        # 枚举所有组合，直接筛选
        for curr_num in range(1, len(bins_num_dict) + 1):
            local_set = set(combinations(input_columns, curr_num))
            selected_set = set()

            for i in local_set:
                # if set_cost(i) <= total_budget:
                curr_cost = set_cost(i)
                # print(f'generate_all_column_list: i = {i}. curr_cost = {curr_cost:.2f}')
                if curr_cost >= min_budget and curr_cost <= max_budget:
                    selected_set.add(i)

            column_list_set.update(selected_set)
        return column_list_set

    def generate_extra_columns(self, comb_num, min_budget = None, max_budget = None):
        """
        生成额外的列
    
        Args:
            comb_num:
            min_budget:
            max_budget:
        Returns:
            column_list_set:
            return2:
        """
        bins_origin_dict = self.bins_builder.construct_bins_dict(\
            self.extra_column_list, split_budget=self.split_budget)
        bins_num_dict = {k: len(v) for k, v in bins_origin_dict.items()}
        print(f"generate_extra_columns: bins_num_dict = {bins_num_dict}")
        input_columns = self.extra_column_list
        
        min_budget = min_budget if min_budget is not None else self.grid_num_min
        max_budget = max_budget if max_budget is not None else self.grid_num_max

        # print(f"generate_extra_columns: bins_num_dict = {bins_num_dict}. input_columns = {input_columns}. total_budget = {total_budget}.")
        target_columns_set = self.generate_all_column_list(\
            bins_num_dict, input_columns, min_budget, max_budget)

        if comb_num < len(target_columns_set):
            # 选择性返回
            # print(f"generate_extra_columns: comb_num = {comb_num}. target_columns_set = {target_columns_set}.")
            index_list = np.random.choice(range(len(target_columns_set)), comb_num, replace=False)
            column_subset = utils.list_index(list(target_columns_set), index_list)
            return column_subset
        else:
            # 直接返回所有结果
            return list(target_columns_set)

    def bind_grid_plan(self, grid_plan_id = "default"):
        """
        绑定当前的grid_plan
    
        Args:
            grid_plan_id:
            arg2:
        Returns:
            return1:
            return2:
        """
        if grid_plan_id == "default":
            self.curr_plan: GridPlan = self.default_grid_plan
        else:
            self.curr_plan: GridPlan = self.grid_plan_dict[grid_plan_id]

        # 函数重新绑定
        self.get_plan_cardinalities = self.curr_plan.get_plan_cardinalities
        self.explore_init_query = self.curr_plan.explore_init_query
        self.explore_init_query_multiround = self.curr_plan.explore_init_query_multiround
        self.grid_info_adjust = self.curr_plan.grid_info_adjust
        self.set_ce_handler = self.curr_plan.set_ce_handler
        self.value_cnt_arr = self.curr_plan.value_cnt_arr

        # 20240303: 新增的函数/成员绑定
        self.generate_random_samples = self.curr_plan.generate_random_samples
        return self.curr_plan

    def clean_table_buffer(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        self.table_builder.data_list = []               # 清空datalist
        self.bins_builder.data_manager.empty_cache()    # 清空缓存

    def load_bins_info(self, column):
        """
        加载新的bins信息
    
        Args:
            column:
            arg2:
        Returns:
            return1:
            return2:
        """
        local_bins = self.bins_builder.construct_bins_dict(\
            selected_columns = [column,], split_budget = self.split_budget)
        local_marginal = self.bins_builder.construct_marginal_dict(bins_dict=local_bins)
        local_reverse = self.bins_builder.construct_reverse_dict(bins_dict=local_bins)

        # self.bins_global[column] = local_bins_dict[column]
        self.update_bins_info(local_bins, local_marginal, local_reverse)

        return self.bins_global
    
    # @utils.timing_decorator
    def infer_extra_columns(self,):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """

        df_col_list = self.subquery_df_dict[self.top_key][0].columns
        origin_column_list = [query_construction.parse_compound_column_name(col_str=col_str, \
                workload=self.workload, alias_reverse=self.alias_inverse) for col_str in df_col_list]

        extra_column_list = list(set(origin_column_list).difference(set(self.column_list)))
        self.extra_column_list = extra_column_list

    def show_candidate_columns(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        extra_column_list = self.extra_column_list
        print(f"show_candidate_columns: query_meta = {self.top_meta}. workload = {self.workload}.")

        for schema_name, column_name in extra_column_list:
            print("schema_name = {}. column_name = {}.".format(schema_name, column_name))

        return extra_column_list

    def update_bins_info(self, local_bins: dict, local_marginal: dict, local_reverse: dict):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        for k, v in local_bins.items():
            self.bins_global[k] = v

        for k, v in local_marginal.items():
            self.marginal_global[k] = v
        
        for k, v in local_reverse.items():
            self.reverse_global[k] = v


    def get_bins_info(self, column):
        """
        获得分桶bounds的基本信息
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        if column not in self.bins_global.keys():
            self.load_bins_info(column)

        return self.bins_global[column]

    def release_df_memory(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        del self.subquery_df_dict
        del self.single_table_df_dict

    def release_grid_memory(self,):
        """
        释放grid相关内存
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        for k, v in self.grid_plan_dict.items():
            try:
                grid_plan: GridPlan = v
                grid_plan.release_grid_memory()
            except Exception as e:
                continue


    # @utils.timing_decorator
    def create_grid_plan(self, new_column_list, split_budget = 100):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        data_df_dict = {}
        subquery_ref, single_table_ref = {}, {}

        # curr_column_list = self.column_list + new_column_list

        # 处理subquery的信息
        for k, v in self.subquery_df_dict.items():
            df_in, curr_meta, column_exist = v

            column_add = self.column_list_match(query_meta=curr_meta, column_list = new_column_list)
            column_local = column_add + column_exist

            df_local = self.null_filting_by_columns(df_in, column_add)
            df_local = self.df_column_clipping(df_local, column_local)

            data_df_dict[k] = df_local

            new_meta = self.add_columns_on_meta(exist_meta=curr_meta, column_list=new_column_list)
            subquery_ref[k] = new_meta, column_local


        # 处理single_table的信息
        for k, v in self.single_table_df_dict.items():
            df_in, curr_meta, column_exist = v

            column_add = self.column_list_match(query_meta=curr_meta, column_list = new_column_list)
            column_local = column_add + column_exist

            df_local = self.null_filting_by_columns(df_in, column_add)
            df_local = self.df_column_clipping(df_local, column_local)

            data_df_dict[k] = df_local

            new_meta = self.add_columns_on_meta(exist_meta=curr_meta, column_list=new_column_list)
            single_table_ref[k] = new_meta, column_local

        grid_top_meta = self.add_columns_on_meta(exist_meta=self.top_meta, \
                        column_list=new_column_list, mode = "value")
        grid_query_meta = self.add_columns_on_meta(exist_meta=self.query_meta, \
                        column_list=new_column_list, mode = "placeholder")

        # 创建grid_plan(同步模式)
        bins_input = {}
        column_total = (self.column_list + list(new_column_list))
        # print(f"create_grid_plan: column_list = {self.column_list}. new_column_list = {new_column_list}")

        for col in column_total:
            # bins_input[col] = self.bins_global[col]
            bins_input[col] = self.get_bins_info(column=col)

        reverse_input = self.bins_builder.construct_reverse_dict(bins_input)
        # print(f"create_grid_plan: column_total = {column_total}. reverse_input = {reverse_input}")

        grid_plan = GridPlan(workload = self.workload, query_meta = grid_query_meta, top_meta = grid_top_meta, \
                mode=self.mode, data_df_dict = data_df_dict, subquery_ref = subquery_ref, single_table_ref = single_table_ref, \
                bins_global=bins_input, reverse_global=reverse_input, alias_mapping=self.alias_mapping, \
                alias_inverse=self.alias_inverse)
        
        grid_plan_id = 1 + len(self.grid_plan_dict)
        grid_plan.set_grid_plan_id(grid_plan_id)
        self.grid_plan_dict[grid_plan_id] = grid_plan
        query_text = query_construction.construct_origin_query(query_meta=grid_top_meta, workload=self.workload)    # 

        self.grid_plan_info[grid_plan_id] = {
            "query": query_text,
            "meta": grid_top_meta,
            "column_list": column_total,
            "grid_num": grid_plan.value_cnt_arr.size,
            "visit_time": 0,
            "history_rewards": []
        }

        return self.grid_plan_dict[grid_plan_id], self.grid_plan_info[grid_plan_id]
    

    @utils.timing_decorator
    def create_default_grid_plan(self,):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        res, _ = self.create_grid_plan(new_column_list=[])
        res: GridPlan = res
        self.default_grid_plan = res
        return res

    def add_columns_on_meta(self, exist_meta, column_list, mode = "value"):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        new_meta = mv_management.meta_copy(exist_meta)
        filter_list = []
        for col in column_list:
            alias_name, column_name = self.alias_mapping[col[0]], col[1]
            if mode == "value":
                bins_local = self.get_bins_info(col)
                # lower_val, uppper_val = bins_local[0], bins_local[-1]
                lower_val, uppper_val = utils.predicate_transform(bins_local, 0, len(bins_local) - 1)

                filter_list.append((alias_name, column_name, lower_val, uppper_val))
            elif mode == "placeholder":
                filter_list.append((alias_name, column_name, "placeholder", "placeholder"))

        new_meta = mv_management.meta_filter_append(new_meta, filter_list)
        return new_meta

    
    def generate_column_combinations(self, column_add_num, comb_num, mode):
        """
        生成列的组合，用于创建grid
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        pass


    def column_list_match(self, query_meta, column_list):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        def column_check(column, query_meta):
            schema_list = query_meta[0]
            return column[0] in schema_list
        
        column_sublist = []

        for col in column_list:
            if column_check(col, query_meta) == True:
                column_sublist.append(col)

        return column_sublist


    @utils.timing_decorator
    def df_column_clipping(self, data_df: pd.DataFrame, column_list):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        df_col_list = self.convert2df_col(column_list=column_list)
        return data_df[df_col_list]
    
    @utils.timing_decorator
    def null_filting_by_columns(self, data_df: pd.DataFrame, column_list):
        """
        根据column进行数据过滤
    
        Args:
            arg1:
            arg2:
        Returns:
            out_df:
            return2:
        """
        df_col_list = self.convert2df_col(column_list=column_list)
        out_df = data_df
        out_df = out_df.dropna(subset=df_col_list, how='any')

        return out_df
    

    def get_template_key(self,):
        """
        获得模版对应的字典键
        
        Args:
            None
        Returns:
            key_str:
        """
        return template_repr_key(query_meta=self.query_meta, selected_columnns=\
                                 self.selected_columns, alias_mapping=self.alias_mapping)

    def set_ce_handler(self, external_handler):
        """
        设置外部的handler
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        if isinstance(external_handler, str):
            # 输入是关键字的case
            # 关键字全部转成小写
            external_handler = external_handler.lower()
            if external_handler == "internal":
                self.ce_handler = ce_injection.PGInternalHandler(workload=self.workload)
            else:
                self.ce_handler = ce_injection.get_ce_handler_by_name(\
                    workload=self.workload, ce_type=external_handler)
        else:
            # 输入是ce_handler实例的情况
            self.ce_handler = external_handler

        self.curr_plan.set_ce_handler(external_handler=self.ce_handler)

    def show_status(self,):
        """
        展示当前的状态
    
        Args:
            None
        Returns:
            return1:
            return2:
        """
        print("show current template_plan info:")
        top_query = query_construction.construct_origin_query(query_meta=self.top_meta, workload=self.workload)
        print("top-level query = {}.".format(top_query))
        print("selected_columns = {}.".format(self.selected_columns))
        print("value_cnt_arr's shape = {}.".format(self.value_cnt_arr.shape))
        print("subquery_info's keys = {}.".format(self.subquery_info.keys()))
        for k, v in self.subquery_info.items():
            print("k = {}. arr_shape = {}. column_list = {}.".format(k, v[0].shape, v[1]))

        if hasattr(self, "single_table_info"):
            # 打印单表的结果
            print("single_table_info's keys = {}.".format(self.single_table_info.keys()))
            for k, v in self.single_table_info.items():
                try:
                    print("k = {}. arr_shape = {}. column_list = {}.".format(k, v[0].shape, v[1]))
                except AttributeError:
                    print("k = {}. array is None. No column exists.".format(k))


    def construct_template_query(self, template_meta, value_dict):
        """
        根据模板和value_dict生成具体的查询实例
        
        Args:
            template_meta: 查询模板的元信息
            value_dict: 谓词值的字典
        Returns:
            result_meta:
            result_query:
        """
        result_meta = mv_management.meta_copy(template_meta)

        filter_new = [] # 新的filter

        for item in result_meta[1]:
            if item[3] == "placeholder":
                curr_column = (self.alias_inverse[item[0]], item[1])
                start_val, end_val = value_dict[curr_column]
                filter_new.append((item[0], item[1], start_val, end_val))
            else:
                filter_new.append(item)

        result_meta = result_meta[0], filter_new
        result_query = query_construction.construct_origin_query(\
            query_meta=result_meta, workload=self.workload)
        
        return result_meta, result_query


    def make_info_dict(self,):
        """
        制作用在保存在meta里的字典
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        def tuple2str(in_tuple):
            return "_".join(in_tuple)
        
        # res_dict = {}
        col_idx_dict = {}

        total_grid_num = 1
        for k, v in self.extra_info_dict.items():
            # print("column_name = {}. start_idx = {}. end_idx = {}.".\
            #       format(k, v[0], v[1]))
            col_idx_dict[tuple2str(k)] = (int(v[0]), int(v[1]))
            total_grid_num *= (v[1] - v[0])
        
        # 补充新的信息
        res_dict = {
            "columns": col_idx_dict,
            "grid_num": int(total_grid_num),
            "cnt_num": int(np.sum(self.value_cnt_arr)),
            "mode": self.mode,
            "grid_plan": {}
        }

        # 补充所属的grid_plan信息
        for k, v in self.grid_plan_info.items():
            res_dict['grid_plan'][k] = {
                "column_list": str(v["column_list"]),
                "grid_num": int(v["grid_num"])
            }

        return res_dict

    @utils.timing_decorator
    def process_top_query(self, cond_bound_dict = None):
        """
        处理最顶层的查询
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        data_df, column_list, res_meta = self.construct_target_df(in_meta=self.top_meta, \
            selected_columns=self.selected_columns, apply_dynamic=True, cond_bound_dict=cond_bound_dict)

        # self.top_meta = res_meta    # 真正的顶层查询元信息
        top_key = self.get_meta_repr(in_meta=res_meta)
        self.top_key, self.top_meta = top_key, res_meta
        self.subquery_df_dict[top_key] = data_df, self.top_meta, column_list

        return data_df, column_list, res_meta
    
    def get_meta_repr(self, in_meta):
        """
        {Description}
        
        Args:
            in_meta:
        Returns:
            repr_str:
        """
        # print("get_meta_repr: in_meta = {}.".format(in_meta))
        repr_str = mv_management.meta_repr(in_meta=in_meta, workload=self.workload)
        return repr_str

    @utils.timing_decorator
    def process_all_single_tables(self, ):
        """
        针对所有单表的信息处理
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        single_table_meta_list = []
        single_table_meta_list = mv_management.meta_decompose(in_meta=self.top_meta, workload=self.workload)

        for single_meta in single_table_meta_list:
            meta_str = self.get_meta_repr(in_meta=single_meta)

            # 获得需要切分的列
            subset_columns = self.get_subset_columns(\
                selected_columns=self.selected_columns, subquery_meta=single_meta)
            
            self.submeta_dict[meta_str] = single_meta   # 保存单表的meta信息

            data_df, column_list, res_meta = self.construct_target_df(single_meta, \
                    selected_columns=subset_columns, apply_dynamic=False)  # 创建目标df
            self.single_table_df_dict[meta_str] = (data_df, res_meta, column_list)        

        return self.single_table_df_dict

    @utils.timing_decorator
    def process_all_subqueries(self,):
        """
        处理所有的子查询，生成对应的grid并且保存到字典中
    
        Args:
            arg1:
            arg2:
        Returns:
            subquery_info: 所有子查询的信息
        """
        sub_meta_list = self.get_all_submeta()
        for sub_meta in sub_meta_list:
            if len(sub_meta[0]) == len(self.top_meta[0]):
                # 顶层查询的情况去除，不需要进行重复的处理
                continue

            # print("process_all_subqueries: sub_meta = {}.".format(sub_meta))
            meta_str = self.get_meta_repr(in_meta=sub_meta)

            # 获得需要切分的列
            subset_columns = self.get_subset_columns(selected_columns=self.selected_columns, subquery_meta=sub_meta)
            # 子查询一律不使用动态mv的技术，但是这可能会造成问题(三表的话应该还可以，三表以上不好说)
            data_df, column_list, res_meta = self.construct_target_df(
                in_meta=sub_meta, selected_columns=subset_columns, apply_dynamic=False
            )
            self.subquery_df_dict[meta_str] = data_df, res_meta, column_list

        return self.subquery_df_dict

    def incremental_process(self,):
        """
        增量的处理数据
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        pass


    def result_merge(self,):
        """
        结果合并
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        pass
    

    def construct_default_query(self, ):
        """
        构造默认的查询
        
        Args:
            None
        Returns:
            query_text: 查询文本
        """
        default_meta = self.query_meta[0], []
        default_query = query_construction.construct_origin_query(\
            query_meta = default_meta, workload=self.workload)
        return default_query
    
    def get_subset_columns(self, selected_columns, subquery_meta):
        """
        获得需要切分的列的子集
    
        Args:
            selected_columns:
            subquery_meta:
        Returns:
            subset_columns:
        """
        subset_columns = []

        for item in selected_columns:
            schema_name, column_name = item
            if schema_name in subquery_meta[0]:
                subset_columns.append(item)

        return subset_columns

    
    def get_all_submeta(self,):
        """
        获得所有子查询的meta信息，为后续的子查询grid生成做基础，并且构建submeta_dict

        Args:
            None
        Returns:
            sub_meta_list:
        """
        self.submeta_dict = {}

        default_query = self.construct_default_query()
        # 设置查询实例
        # self.query_ctrl.set_query_instance(query_text=default_query, query_meta=self.query_meta)
        # 在这里query_meta改成top_meta
        self.query_ctrl.set_query_instance(query_text=default_query, query_meta=self.top_meta)

        sub_relation_list = self.query_ctrl.get_all_sub_relations()
        sub_meta_list = []
        for sub_relation in sub_relation_list:
            # sub_meta_list.append(mv_management.meta_subset(in_meta=self.query_meta, \
            #     schema_subset=sub_relation, abbr_mapping=self.workload))
            # 在这里query_meta改成top_meta
            sub_meta = mv_management.meta_subset(in_meta=self.top_meta, \
                schema_subset=sub_relation, abbr_mapping=self.workload)
            sub_meta_list.append(sub_meta)
            
            # 保存meta
            meta_str = self.get_meta_repr(in_meta=sub_meta)
            self.submeta_dict[meta_str] = sub_meta
            
        return sub_meta_list

    def process_meta(self, in_meta):
        """
        处理meta信息，输出table_builder所需要的meta和selected_columns
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        out_meta = in_meta[0], []
        selected_columns = []

        for item in in_meta[1]:
            if item[3] == "placeholder":
                selected_columns.append((self.alias_inverse[item[0]], item[1]))
            else:
                out_meta[1].append(item)

        return out_meta, selected_columns


    def construct_target_df(self, in_meta, selected_columns, apply_dynamic = True, cond_bound_dict = None):
        """
        {Description}
        
        Args:
            in_meta: 
            selected_columns: 
            apply_dynamic:
        Returns:
            data_df: 
            column_list: 
            res_meta:
        """
        # out_meta, selected_columns = self.process_meta(in_meta)
        out_meta = in_meta

        # 构建多表的对象
        src_meta_list = [out_meta, ]  # 对象元信息组成的列表
        data_df, column_list, merged_meta, res_meta_list = self.table_builder.build_joined_tables_object(
            src_meta_list=src_meta_list, selected_columns=selected_columns, apply_dynamic=apply_dynamic, \
            filter_columns = False, cond_bound_dict=cond_bound_dict
        )
        res_meta = res_meta_list[0] # 新的meta信息


        data_manager = self.table_builder.data_manager
        # 删除join_column, primary_key_column，优化存储空间
        valid_column_list = reduce(lambda a, b: a+b, [[(s, c) for c in data_manager.get_valid_columns(\
                        schema_name=s)] for s in in_meta[0]], [])
        valid_column_df = [f"{self.alias_mapping[s]}_{c}" for s, c in valid_column_list]
        selected_column_df = [col for col in data_df.columns if col in valid_column_df]

        # print(f"construct_target_df: before column_num = {len(data_df.columns)}. after column_num = {len(selected_column_df)}.")
        data_df = data_df[selected_column_df]

        column_origin = [query_construction.parse_compound_column_name(col_str=col_str, \
                        workload=self.workload) for col_str in column_list]
        # print(f"construct_target_df: valid_column_list = {valid_column_list}")
        # print(f"construct_target_df: column_origin = {column_origin}")
        # print(f"construct_target_df: column_df = {data_df.columns}")

        return data_df, column_origin, res_meta
        

    def build_grid_analyzer(self,):
        """
        构建grid分析器，用于查询生成和分析，由于dynamic_mv的问题，这里需要做预处理
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        value_cnt_arr = self.value_cnt_arr
        marginal_value_arr = None
        bins_list = []
        for col in self.column_list:
            bins_list.append(col)
        query_meta = self.top_meta
        params_meta = None
        method = "internal"
        workload = self.workload
        local_analyzer = grid_analysis.GridAnalyzer(value_cnt_arr=value_cnt_arr, marginal_value_arr=marginal_value_arr,
                                                    bins_list=bins_list, query_meta=query_meta, params_meta=params_meta,
                                                    method=method, workload=workload, load_history=False)
        
        return local_analyzer


# %%
def get_template_plan_in_cache(path) -> TemplatePlan:
    """
    从cache中获得template_plan

    Args:
        path: 模板计划的路径
    Returns:
        template_plan:
        return2:
    """
    return utils.load_pickle(data_path=path)


# %%

class GridPlan(object):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    # def __init__(self, query_meta, data_df_dict, bins_list, columns_list):
    # @utils.timing_decorator
    def __init__(self, workload, query_meta, top_meta, mode, data_df_dict: dict, subquery_ref: dict, single_table_ref: dict, \
                 bins_global: dict, reverse_global: dict, alias_mapping: dict, alias_inverse: dict):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        # print(f"GridPlan.__init__: query_meta = {query_meta}. \nbins_global = {bins_global}. reverse_global = {reverse_global}.")
        self.workload = workload
        self.alias_mapping, self.alias_inverse = alias_mapping, alias_inverse

        self.top_meta, self.query_meta = top_meta, query_meta
        self.mode = mode        # 基数估计错误模式

        self.bins_global = bins_global
        self.reverse_global = reverse_global

        self.top_key = max(subquery_ref.keys(), key=lambda a:len(a))
        # print(f"GridPlan.__init__: top_meta = {self.top_meta}. top_key = {self.top_key}.")

        assert mode in ("over-estimation", "under-estimation")     # 目前只支持两种模式
        # self.template_plan_ref = template_plan_ref
        self.data_df_dict = data_df_dict

        # 子查询和单表的信息
        self.subquery_info = {}
        self.single_table_info = {}
        self.submeta_dict = {}  # 子查询元信息的字典

        # 创建所有的value_cnt_arr(子查询和单表的)
        self.create_value_cnt_arraies(data_df_dict, subquery_ref, single_table_ref)
        self.convert2df_col = partial(convert2df_col, alias_mapping = self.alias_mapping)

        self.grid_info_adjust()     # 调整grid_info
        self.grid_plan_id = -1      # 当前GridPlan在TemplatePlan下的ID
        
        # 直接释放内存
        self.release_grid_memory()
        self.ce_handler = None    # ce_handler默认设置成None
        self.template_plan: TemplatePlan = None

    def set_grid_plan_id(self, new_id):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        self.grid_plan_id = new_id

    def add_template_plan_ref(self, template_plan: TemplatePlan):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        self.template_plan: TemplatePlan = template_plan

    # @utils.timing_decorator
    def create_value_cnt_arraies(self, data_df_dict: dict, subquery_ref: dict, single_table_ref: dict):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        for key, value in subquery_ref.items():
            data_df = data_df_dict[key]
            meta_local, columns_local = value
            value_cnt_arr, column_list = \
                self.construct_grid_array(data_df=data_df, column_list=columns_local)
            # print(f"create_value_cnt_arraies: value_cnt_arr.shape = {value_cnt_arr.shape}. column_list = {column_list}")

            self.subquery_info[key] = (value_cnt_arr, column_list)

        for key, value in single_table_ref.items():
            data_df = data_df_dict[key]
            meta_local, columns_local = value
            value_cnt_arr, column_list = \
                self.construct_grid_array(data_df=data_df, column_list=columns_local)
            
            self.single_table_info[key] = (value_cnt_arr, column_list)

        # 特殊处理一下top_key的信息
        self.value_cnt_arr, self.column_list = self.subquery_info[self.top_key]
        
        return self.subquery_info, self.single_table_info
        

    def grid_info_adjust(self, origin_meta = None, res_meta = None) -> dict:
        """
        调整grid相关信息，主要是bins的信息在dynamic_mv的时候更新过了，
        直接沿用原来的会出现很多个0
    
        Args:
            origin_meta: 输入查询的元信息
            res_meta: 输出查询的元信息
        Returns:
            extra_info_dict: key是column，value是(start_idx, end_idx)组成的元组
        """
        if origin_meta is None:
            origin_meta = self.query_meta

        if res_meta is None:
            res_meta = self.top_meta
        # print("grid_info_adjust.top_query = {}.".format(query_construction.construct_origin_query(\
        #     self.top_meta, workload=self.workload)))
        # print("grid_info_adjust.origin_meta = {}. res_meta = {}. total_card = {}.".\
        #       format(origin_meta, res_meta, np.sum(self.value_cnt_arr)))

        extra_info_dict = {}
        _, filter_origin = origin_meta
        _, filter_res = res_meta

        for item in filter_res:
            alias_name, column_name, start_val, end_val = item
            schema_name = self.alias_inverse[alias_name]
            curr_key = schema_name, column_name
            if item not in filter_origin:
                try:
                    reverse_local = self.reverse_global[curr_key]
                    # print(f"grid_info_adjust: curr_key = {curr_key}. reverse_local = {reverse_local}. start_val = {start_val}. end_val = {end_val}")
                    start_idx, end_idx = utils.predicate_location(\
                        reverse_dict=reverse_local, start_val=start_val, end_val=end_val)
                    extra_info_dict[curr_key] = start_idx, end_idx
                except KeyError as e:
                    print(f"grid_info_adjust: meet KeyError. reverse_global.keys = {self.reverse_global.keys()}.")
                    raise e


        self.extra_info_dict = extra_info_dict
        return extra_info_dict
    

    def construct_column_elements(self,):
        """
        {Description}
    
        Args:
            None
        Returns:
            column_order:
            column_info:
        """
        meta_repr_key = self.get_meta_repr(in_meta=self.top_meta)
        _, column_order = self.subquery_info[meta_repr_key]
        column_info = {}

        # marginal_dict = self.bins_builder.construct_marginal_dict(bins_dict=bins_dict)
        bins_builder = self.template_plan.bins_builder
        marginal_dict = bins_builder.construct_marginal_dict(\
            bins_dict=self.bins_global)

        for col in column_order:
            column_info[col] = {
                "bins_list": self.bins_global[col],
                "marginal_list": marginal_dict[col],
                "reverse_dict": self.reverse_global[col]
            }

        return column_order, column_info
    
    def construct_bayes_selector(self, external_card_func = None):
        """
        {Description}
    
        Args:
            external_card_func: 入参为(query_text, query_meta)
            arg2:
        Returns:
            return1:
            return2:
        """
        meta_repr_key = self.get_meta_repr(in_meta=self.top_meta)
        _, column_order = self.subquery_info[meta_repr_key]
        target_func = self.build_target_function(external_card_func)
        column_info = {}

        # marginal_dict = self.bins_builder.construct_marginal_dict(bins_dict=bins_dict)
        if self.template_plan is None:
            bins_builder = grid_preprocess.get_bins_builder_by_workload(self.workload)
        else:
            bins_builder = self.template_plan.bins_builder
        marginal_dict = bins_builder.construct_marginal_dict(\
            bins_dict=self.bins_global)

        for col in column_order:
            column_info[col] = {
                "bins_list": self.bins_global[col],
                "marginal_list": marginal_dict[col],
                "reverse_dict": self.reverse_global[col]
            }

        bayes_selector = bayesian_optimazition.BayesSelector(\
            self.workload, self.mode, target_func, column_info, column_order)
        
        return bayes_selector
    
    @utils.timing_decorator
    def explore_by_bayes_optim(self, min_card, max_card, sample_num, construct_bayes_selector = None):
        """
        {Description}
    
        Args:
            min_card: 
            max_card:
            sample_num:
        Returns:
            query_text: 
            query_meta: 
            true_card: 
            estimation_card:
        """
        if hasattr(self, "bayes_selector") == False:
            self.bayes_selector = self.construct_bayes_selector(construct_bayes_selector)

        range_dict, true_card, estimation_card = \
            self.bayes_selector.explore_query(sample_num)

        query_meta = self.range_dict2query_meta(range_dict)
        query_text = query_construction.construct_origin_query(query_meta, self.workload)
        return query_text, query_meta, true_card, estimation_card

    def explore_by_random_sample(self, min_card, max_card, sample_num, target):
        """
        随机化探索最优结果
        
        Args:
            min_card:
            max_card:
            sample_num:
            target:
        Returns:
            query_text: 
            query_meta: 
            true_card: 
            estimation_card:
        """
        query_list, meta_list, label_list, estimation_list = \
            self.generate_random_samples(min_card, max_card, sample_num)
        target_res = self.select_best_result(query_list, meta_list, \
                        label_list, estimation_list)
        return target_res


    def generate_random_samples(self, min_card, max_card, sample_num, with_time = False):
        """
        {Description}
    
        Args:
            min_card:
            max_card:
            sample_num:
            with_time:
        Returns:
            query_list: 
            meta_list: 
            label_list: 
            estimation_list:
            time_list(optional):
        """
        query_list, meta_list, label_list, estimation_list = \
            [], [], [], []
        
        sample_limit = sample_num * 5   # 总采样数的限制
        invalid_list, label_total = [], []

        if with_time == True:
            time_start, time_list = time.time(), []

        for _ in range(sample_limit):
            test_query, test_meta = self.generate_random_query()    
            label = self.get_true_cardinality(in_meta=test_meta, query_text=test_query, mode="subquery")
            # print(f"explore_by_random_sample: test_meta = {test_meta}. label = {label}.")

            label_total.append(label)
            if min_card <= label <= max_card:
                # 只有在label满足条件的情况下才会被添加
                query_list.append(test_query)
                meta_list.append(test_meta)
                label_list.append(label)
            else:
                # print("label = {}.".format(label))
                invalid_list.append(label)  # 添加到非法列表中去

            if with_time == True:
                # 每次获取估计基数
                est_card = self.ce_handler.get_cardinalities([test_query,])[0]
                estimation_list.append(est_card)
                time_curr = time.time()
                time_list.append(time_curr - time_start)
                
            if len(label_list) >= sample_num:
                break
        
        # print("len(query_list) = {}. len(label_list) = {}.".\
        #       format(len(query_list), len(label_list)))
        # print(f"explore_by_random_sample: label_total = {label_total}")

        if with_time == False:
            # 不记录时间的话，采用常规的获取基数方法
            estimation_list = self.ce_handler.get_cardinalities(query_list=query_list)

        if with_time == True:
            return query_list, meta_list, label_list, estimation_list, time_list
        else:
            return query_list, meta_list, label_list, estimation_list

    def select_best_result(self, query_list, meta_list, label_list, estimation_list):
        """
        从一批结果中选择最优的结果
        
        Args:
            query_list:
            meta_list: 
            label_list: 
            estimation_list:
            target: 优化目标，目前支持["over", "under"]这几种目标，含义如下
                    over:
                    under:
        Returns:
            query_text:
            query_meta:
            true_card: 
            estimation_card:
        """
        if len(query_list) == 0:
            query_text, query_meta = "", ([], [])
            true_card, estimation_card = 1.0, 1.0
            return query_text, query_meta, true_card, estimation_card

        # print("select_best_result: label_list = {}.".format(label_list))
        # print("select_best_result: estimation_list = {}.".format(estimation_list))
        # print("select_best_result: query_len = {}. meta_len = {}. label_len = {}. estimation_len = {}.".\
        #       format(len(query_list), len(meta_list), len(label_list), len(estimation_list)))
        
        composite_list, metrics_list = [], []

        for true_card, estimation_card in zip(label_list, estimation_list):
            if self.mode == "over-estimation":
                metrics_list.append(estimation_card / (1.0 + true_card))
            elif self.mode == "under-estimation":
                metrics_list.append(true_card / (1.0 + estimation_card))
        
        composite_list = list(zip(query_list, meta_list, label_list, estimation_list, metrics_list))
        composite_list.sort(key=lambda a:a[4], reverse=True)

        # print("select_best_result: len(composite_list) = {}. len(composite_list[0]) = {}.".\
        #       format(len(composite_list), len(composite_list[0])))
        
        # print("select_best_result(true_card, est_card): mode = {}. best_case = ({}, {}). worst_case = ({}, {}).".\
        #       format(self.mode, composite_list[0][2], composite_list[0][3], composite_list[-1][2], composite_list[-1][3]))
        query_text, query_meta, true_card, estimation_card = \
            composite_list[0][0], composite_list[0][1], composite_list[0][2], composite_list[0][3]
        
        return query_text, query_meta, true_card, estimation_card


    def range_dict2query_meta(self, range_dict):
        # query_text = ""
        local_meta = mv_management.meta_copy(in_meta=self.query_meta)
        filter_new = [] # 新的filter

        for item in local_meta[1]:
            if item[3] == "placeholder":
                curr_column = (self.alias_inverse[item[0]], item[1])
                start_val, end_val = range_dict[curr_column]
                filter_new.append((item[0], item[1], start_val, end_val))
            else:
                # 不可变的条件
                filter_new.append(item)
        final_meta = local_meta[0], filter_new
        return final_meta

    def build_target_function(self, external_card_func = None):
        """
        {Description}
    
        Args:
            external_card_func:
            arg2:
        Returns:
            return1:
            return2:
        """
        def adjust_func(true_card, est_card):
            true_card += 1.0 
            est_card += 1.0
            penalty_factor = (100 / true_card) + 1.0
            return 1 / penalty_factor

        def target_func(range_input):
            # 目标函数，支持单个/多个的输入
            if isinstance(range_input, list):
                val_list = range_input
            elif isinstance(range_input, dict):
                val_list = [range_input,]
            else:
                raise TypeError(f"target_func: type(range_input) = {type(range_input)}")
            
            query_list, meta_list = [], []
            true_card_list, est_card_list = [], []

            for range_dict in val_list:
                in_meta = self.range_dict2query_meta(range_dict)
                in_query = query_construction.construct_origin_query(in_meta, self.workload)
                query_list.append(in_query), meta_list.append(in_meta)
                if external_card_func is None:
                    card_true = self.get_true_cardinality(in_meta=in_meta, query_text=in_query, mode="subquery")
                else:
                    card_true = external_card_func(in_query, in_meta)

                true_card_list.append(card_true)

            est_card_list = self.ce_handler.get_cardinalities(query_list)

            # a: true_card, b: est_card
            if self.mode == "under-estimation":
                val_func = lambda a, b: (a + 1.0) / (b + 1.0)
            elif self.mode == "over-estimation":
                val_func = lambda a, b: (b + 1.0) / (a + 1.0)

            q_error_list = [val_func(true_card, est_card) for \
                true_card, est_card in zip(true_card_list, est_card_list)]

            log_q_error_list = np.log(q_error_list)
            adjust_factor_list = [adjust_func(true_card, est_card) for \
                true_card, est_card in zip(true_card_list, est_card_list)]

            metrics_list = [(error * factor) for error, factor in \
                            zip(log_q_error_list, adjust_factor_list)]
            
            # print(f"adjust_factor_list = {adjust_factor_list}.")
            # print(f"metrics_list = {metrics_list}.")

            # return q_error_list, true_card_list, est_card_list
            return metrics_list, true_card_list, est_card_list
        
        return target_func

    def get_true_cardinality(self, in_meta, query_text, mode = "subquery"):
        """
        获得查询真实的基数
        
        Args:
            in_meta: 查询的元信息
            query_text: 查询的文本信息
            mode: ["subquery", "single_table"]，子查询模式和单表模式
        Returns:
            true_card:
        """
        meta_repr_key = self.get_meta_repr(in_meta=in_meta)
        # 从subquery_info中获取信息
        if mode == "subquery":
            value_cnt_arr, column_order = self.subquery_info[meta_repr_key]
        elif mode == "single_table":
            value_cnt_arr, column_order = self.single_table_info[meta_repr_key]
        else:
            raise ValueError("Unsupported mode: {}".format(mode))
        
        _, reverse_dict = self.construct_local_bins(schema_list=in_meta[0])
        alias_reverse = self.alias_inverse

        # print("get_true_cardinality: column_order = {}.".format(column_order))

        # 推测真实基数
        true_card = grid_advance.infer_true_cardinality(query_meta=in_meta, value_cnt_arr=value_cnt_arr,\
            column_order=column_order, reverse_dict=reverse_dict, alias_reverse=alias_reverse)
        # print("query_text = {}. true_card = {}.".format(query_text, true_card))

        return int(true_card)


    def get_plan_subqueries(self, in_meta, query_text):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """

        subquery_dict = {}
        single_table_dict = {}
        top_meta_repr = self.get_meta_repr(in_meta=self.top_meta)

        for k, v in self.subquery_info.items():
            # curr_meta = self.submeta_dict[k]    # curr_meta设置错误
            curr_meta = mv_management.meta_subset(in_meta=in_meta, 
                schema_subset=list(k), abbr_mapping=self.workload)  # 还是采用取子集的方法获得meta信息
            # print("k = {}. curr_meta = {}. in_meta = {}.".format(k, curr_meta, in_meta))
            curr_query = query_construction.construct_origin_query(query_meta=curr_meta, workload=self.workload)
            subquery_dict[k] = curr_query

        # 填充single_table的内容
        if hasattr(self, 'single_table_info'):
            # 存在单表基数的情况
            for k, v in self.single_table_info.items():
                curr_meta = mv_management.meta_subset(in_meta=in_meta,
                    schema_subset=k, abbr_mapping=self.workload)
                curr_query = query_construction.construct_origin_query(
                    query_meta=curr_meta, workload=self.workload)
                # print("single_table_info: curr_meta = {}. curr_query = {}.".
                #       format(curr_meta, curr_query))
                single_table_dict[k] = curr_query

        subquery_dict[top_meta_repr] = query_construction.construct_origin_query(\
            query_meta=self.top_meta, workload=self.workload)
        return subquery_dict, single_table_dict

    # @utils.timing_decorator
    def get_plan_cardinalities(self, in_meta, query_text):
        """
        获得一个查询对应的计划搜索中全部需要用到的基数
        
        Args:
            in_meta: 元信息
            query_text: 查询文本
        Returns:
            subquery_dict:
            single_table_dict:
        """
        top_cardinality = self.get_true_cardinality(in_meta=in_meta, query_text=query_text)
        top_meta_repr = self.get_meta_repr(in_meta=self.top_meta)

        subquery_dict = {}
        single_table_dict = {}

        # 填充subquery的内容
        for k, v in self.subquery_info.items():
            # curr_meta = self.submeta_dict[k]    # curr_meta设置错误
            curr_meta = mv_management.meta_subset(in_meta=in_meta, 
                schema_subset=list(k), abbr_mapping=self.workload)  # 还是采用取子集的方法获得meta信息
            # print("k = {}. curr_meta = {}. in_meta = {}.".format(k, curr_meta, in_meta))
            curr_query = query_construction.construct_origin_query(query_meta=curr_meta, workload=self.workload)
            
            card = self.get_true_cardinality(in_meta=curr_meta, query_text=curr_query)
            subquery_dict[k] = card            

        # 填充single_table的内容
        if hasattr(self, 'single_table_info'):
            # 存在单表基数的情况
            for k, v in self.single_table_info.items():
                curr_meta = mv_management.meta_subset(in_meta=in_meta,
                    schema_subset=k, abbr_mapping=self.workload)
                curr_query = query_construction.construct_origin_query(
                    query_meta=curr_meta, workload=self.workload)
                # print("single_table_info: curr_meta = {}. curr_query = {}.".
                #       format(curr_meta, curr_query))
                card = self.get_true_cardinality(in_meta=curr_meta, query_text=curr_query, mode="single_table")
                # single_table_dict[k] = card
                single_table_dict[k[0]] = card      # 修改key的形态

        subquery_dict[top_meta_repr] = top_cardinality  # 顶层的基数

        # print("get_plan_cardinalities: subquery_info = {}. single_table_info = {}".\
        #       format(self.subquery_info, self.single_table_info))
        
        return subquery_dict, single_table_dict

    def explore_init_query_multiround(self, external_info: dict, update = True):
        """
        探索用于初始根节点的查询，返回单个结果，external_info代表外部的输入信息，
        引导TemplatePlan选择更优的结果

        Args:
            external_info:
            update_num: 
        Returns:
            selected_query: 被选择的查询文本
            selected_meta: 被选择的查询元信息
            true_card: 真实基数
            estimation_card: 估计的基数
        """
        if hasattr(self, "query_cache") == False:
            self.query_cache = []
        
        truncate_threshold = 100
        # cmp_func = lambda a, b: max((a / (b + 1.0)), (b / (a + 1.0)))
        assert self.mode in ("over-estimation", "under-estimation")
        if self.mode == "over-estimation":
            cmp_func = lambda a, b: b / (a + 1.0)
        elif self.mode == "under-estimation":
            cmp_func = lambda a, b: a / (b + 1.0)

        def merge_result(instance_list):
            # 合并之前的结果
            self.query_cache.extend(instance_list)
            # item[2]: true_card, item[3]: estimation_card
            self.query_cache.sort(key=lambda item:cmp_func(item[2], item[3]), reverse=True)

            if len(self.query_cache) > truncate_threshold:
                self.query_cache = self.query_cache[:truncate_threshold]
            return self.query_cache
        
        def select_best_instance():
            # 每次弹出一个最好的结果
            assert len(self.query_cache) > 0
            res_item = self.query_cache[0]
            self.query_cache = self.query_cache[1:]
            return res_item
        
        inf = 1e10
        lower_bound = external_info.get("min_card", -inf)
        upper_bound = external_info.get("max_card", inf)
        test_num = external_info.get("num", 20)

        if update == True:
            # 生成新的实例
            query_list, meta_list, label_list, estimation_list = \
                self.generate_random_samples(lower_bound, upper_bound, test_num)
            assert len(query_list) == len(meta_list) == len(label_list) == len(estimation_list)
            instance_list = zip(query_list, meta_list, label_list, estimation_list)
            merge_result(instance_list)
        
        query_text, query_meta, true_card, estimation_card = select_best_instance()
        return query_text, query_meta, true_card, estimation_card

    # @utils.timing_decorator
    def explore_init_query(self, external_info: dict):
        """
        探索用于初始根节点的查询，返回单个结果，external_info代表外部的输入信息，
        引导TemplatePlan选择更优的结果

        在一开始采用随机探索的策略，然后选择估计器最坏的结果，之后会考虑Column上的其他特征
        Args:
            external_info: 外部的信息输入，目前考虑bayesian和sample-based两种方法
            arg2:
        Returns:
            selected_query: 被选择的查询文本
            selected_meta: 被选择的查询元信息
            true_card: 真实基数
            estimation_card: 估计的基数
        """
        external_info['mode'] = external_info.get("mode", "sample-based")
        # external_info['target'] = external_info.get("target", "over")

        if self.mode == "over-estimation":
            target = "over"
        elif self.mode == "under-estimation":
            target = "under"
        else:
            raise ValueError(f"explore_init_query: self.mode = {self.mode}")

        test_num = external_info.get("num", 20)
        inf = 1e10
        lower_bound = external_info.get("min_card", -inf)
        upper_bound = external_info.get("max_card", inf)
        if external_info['mode'] == "sample-based":
            # 随机产生20个查询，再组装结果
            # if 'num' not in external_info[]
            
            # 有基数范围的查询探索
            # print("explore_init_query: lower_bound = {}. upper_bound = {}. test_num = {}.".\
            #       format(lower_bound, upper_bound, test_num))
            query_text, query_meta, true_card, estimation_card = self.explore_by_random_sample(\
                min_card=lower_bound, max_card=upper_bound, sample_num=test_num, target=target)
        elif external_info['mode'] == "bayesian":
            query_text, query_meta, true_card, estimation_card = self.explore_by_bayes_optim(\
                min_card=lower_bound, max_card=upper_bound, sample_num=test_num)
        else:
            raise ValueError("explore_init_query: Unsupported mode: {}. available list = ['sample-based', 'bayesian']"\
                             .format(external_info['mode']))
        # print(f"explore_init_query: query_meta = {query_meta}. q_error = {true_card / estimation_card:.3f}.")
        return query_text, query_meta, true_card, estimation_card


    # def grid_infer_verification(self,):
    #     """
    #     验证dynamic_mv以后grid的正确性
    
    #     Args:
    #         arg1:
    #         arg2:
    #     Returns:
    #         flag: 验证是否正确
    #         return2:
    #     """
    #     # 针对top-level query的验证
    #     extra_info_dict = self.grid_info_adjust(origin_meta=self.query_meta, res_meta=self.top_meta)
    #     print("grid_verification: extra_info_dict = {}.".format(extra_info_dict))
    #     # 构造value_cnt_arr中特定区域的index
    #     local_idx_list = []

    #     def build_index_list(column_list, info_dict):
    #         local_idx_list = []
    #         for compound_col in column_list:
    #             print("grid_verification: compound_col = {}.".format(compound_col))
    #             schema_name, column_name = query_construction.\
    #                 parse_compound_column_name(col_str=compound_col, workload=self.workload)
    #             curr_key = (schema_name, column_name)
    #             if curr_key in info_dict.keys():
    #                 start_idx, end_idx = info_dict[curr_key]
    #                 local_idx_list.append(slice(start_idx, end_idx))
    #             else:
    #                 local_idx_list.append(Ellipsis)
    #         return local_idx_list
        
    #     local_idx_list = build_index_list(column_list=self.column_list, info_dict=extra_info_dict)
    #     total_sum_count, specified_sum_count = 0, 0

    #     value_cnt_arr = self.value_cnt_arr
    #     total_sum_count = np.sum(value_cnt_arr)
    #     specified_sum_count = np.sum(value_cnt_arr[tuple(local_idx_list)])

    #     print("grid_verification: local_idx_list = {}.".format(local_idx_list))
    #     print("grid_verification: total_sum = {}. specified_sum = {}.".\
    #           format(total_sum_count, specified_sum_count))
        
    #     top_flag = total_sum_count == specified_sum_count

    #     # 针对子查询的验证
    #     subquery_flag = {}
    #     for k, v in self.subquery_info.items():
    #         value_cnt_arr, column_list = v  # 注意两个变量的顺序

    #         local_idx_list = build_index_list(
    #             column_list=column_list, info_dict=extra_info_dict)
    #         total_sum_count, specified_sum_count = 0, 0

    #         total_sum_count = np.sum(value_cnt_arr)
    #         specified_sum_count = np.sum(value_cnt_arr[tuple(local_idx_list)])
    #         flag = total_sum_count == specified_sum_count
    #         subquery_flag[k] = flag

    #     # TODO: 针对单表的验证
    #     single_table_flag = {}
    #     # for k, v in self.single_table_info.items():
    #     #     pass

    #     return top_flag, subquery_flag, single_table_flag


    # def value_count_correctness_verification(self, test_num, mode):
    #     """
    #     验证value_count的正确性，直接根据value_count生成对应的query，
    #     然后在数据库中验证，每一个查询实际上对应于value_cnt_arr的一个格点
        
    #     Args:
    #         test_num: 正确性测试样例的数目
    #         mode: 测试模式，目前支持的模式为"random", "sequential"，分别
    #         代表随机测试和顺序测试。
    #     Returns:
    #         query_list: 查询列表
    #         cardinality_list: 基数列表
    #     """
    #     value_cnt_arr = self.value_cnt_arr
    #     total_size = value_cnt_arr.size
    #     selected_idx_list = []
    #     query_list, cardinality_list = [], []


    #     if hasattr(self, "extra_info_dict") == False:        
    #         # 直接从全局出发
    #         if test_num == "all":
    #             selected_idx_list = [i for i in range(total_size)]
    #         else:
    #             if mode == "random":
    #                 selected_idx_list = np.random.choice(total_size, size=test_num, replace=False)
    #             elif mode == "sequential":
    #                 selected_idx_list = [i for i in range(test_num)]
    #             else:
    #                 raise ValueError("value_count_correctness_verification: unsupported mode \"{}\"".format(mode))
    #     else:
    #         # 得从selected_region中出发
    #         global_start_list, global_end_list = [], []
    #         for col_str in self.column_list:
    #             schema_name, column_name = query_construction.parse_compound_column_name(\
    #                 col_str=col_str, workload=self.workload)
    #             start_idx, end_idx = self.extra_info_dict[(schema_name, column_name)]
    #             global_start_list.append(start_idx)
    #             global_end_list.append(end_idx)
            
    #         # global_start_idx = np.ravel_multi_index(tuple(global_start_list), dims=self.value_cnt_arr.shape)
    #         # global_end_idx = np.ravel_multi_index(tuple(global_end_list), dims=self.value_cnt_arr.shape)
    #         # print("global_start_idx = {}. global_end_idx = {}.".format(global_start_idx, global_end_idx))
    #         available_idx_list = list(product(*[range(start, end) for start, end in zip(global_start_list, global_end_list)]))
    #         print("available_idx_list = {}.".format(available_idx_list))
    #         available_idx_list = [np.ravel_multi_index(item, dims=value_cnt_arr.shape) for item in available_idx_list]
    #         global_start_idx, global_end_idx = 0, len(available_idx_list)

    #         if test_num == "all":
    #             # 选择所有测试案例的情况
    #             selected_idx_list = available_idx_list
    #         else:
    #             if mode == "random":
    #                 local_idx_list = np.random.choice(range(global_start_idx, global_end_idx), size=test_num, replace=False)
    #                 selected_idx_list = []
    #                 for idx in local_idx_list:
    #                     selected_idx_list.append(available_idx_list[idx])
    #             elif mode == "sequential":
    #                 selected_idx_list = available_idx_list[:test_num]
    #             else:
    #                 raise ValueError("value_count_correctness_verification: unsupported mode \"{}\"".format(mode))

    #         print("selected_idx_list = {}.".format(selected_idx_list))

    #     for idx in selected_idx_list:
    #         selected_grid_idx = np.unravel_index(idx, shape=value_cnt_arr.shape)
            
    #         value_dict = {}
    #         for col, curr_idx in zip(self.column_list, selected_grid_idx):
    #             schema, column = query_construction.parse_compound_column_name(col_str=col, workload=self.workload)

    #             bins_local = self.bins_global[(schema, column)]
    #             start_idx, end_idx = curr_idx, curr_idx + 1
    #             start_val, end_val = grid_preprocess.predicate_transform(bins_list=bins_local, 
    #                                     start_idx=start_idx, end_idx=end_idx)
    #             # value_dict[col] = start_val, end_val
    #             value_dict[(schema, column)] = start_val, end_val
                
    #         result_meta, result_query = self.construct_template_query(
    #             template_meta=self.query_meta, value_dict=value_dict)
    #         query_list.append(result_query)
    #         cardinality_list.append(value_cnt_arr[selected_grid_idx])
        
    #     return query_list, cardinality_list
    
    def set_ce_handler(self, external_handler):
        """
        设置外部的handler
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        if isinstance(external_handler, str):
            # 输入是关键字的case
            # 关键字全部转成小写
            external_handler = external_handler.lower()
            if external_handler == "internal":
                self.ce_handler = ce_injection.PGInternalHandler(workload=self.workload)
            else:
                self.ce_handler = ce_injection.get_ce_handler_by_name(\
                    workload=self.workload, ce_type=external_handler)
        else:
            # 输入是ce_handler实例的情况
            self.ce_handler = external_handler

    def get_meta_repr(self, in_meta):
        """
        {Description}
        
        Args:
            in_meta:
        Returns:
            repr_str:
        """
        # print("get_meta_repr: in_meta = {}.".format(in_meta))
        repr_str = mv_management.meta_repr(in_meta=in_meta, workload=self.workload)
        return repr_str
    
    def specific_condition(self, left_bound, right_bound, desired_size = None):
        """
        根据左右的边界确定大小
        
        Args:
            left_bound: 左边界
            right_bound: 右边界
            desired_size: 期望大小
        Returns:
            res1:
            res2:
        """
        # print("specific_condition: left_bound = {}. right_bound = {}.".format(left_bound, right_bound))
        if desired_size is None:
            desired_size = np.random.randint(1, right_bound - left_bound + 1)

        start = np.random.randint(left_bound, right_bound - desired_size + 1)
        end = start + desired_size
        return start, end

    def random_condition(self, total_size, desired_size = None):
        """
        随机生成一个条件
    
        Args:
            desired_size: 目标大小
            total_size: 总的大小
        Returns:
            start:
            end:
        """
        if desired_size is None:
            desired_size = np.random.randint(1, total_size)

        start = np.random.randint(0, total_size - desired_size)
        end = start + desired_size
        return start, end


    def grid_compression(self, extra_info_dict:dict) -> dict:
        """
        针对grid的结果进行压缩
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        pass
    

    def generate_random_query(self, ):
        """
        生成随机查询
        
        Args:
            None
        Returns:
            query_text: 查询的元信息
            query_meta: 查询的文本
        """
        extra_info_dict = self.extra_info_dict  # 设置额外的信息

        query_text = ""
        local_meta = mv_management.meta_copy(in_meta=self.query_meta)
        filter_new = [] # 新的filter

        for item in local_meta[1]:
            if item[3] == "placeholder":
                # 可变的条件
                curr_column = (self.alias_inverse[item[0]], item[1])
                bins_local = self.bins_global[curr_column]
                schema_name, column_name = self.alias_inverse[item[0]], item[1]
                if (schema_name, column_name) not in extra_info_dict.keys():
                    start_idx, end_idx = self.random_condition(total_size=len(bins_local))
                else:
                    left_bound, right_bound = extra_info_dict[(schema_name, column_name)]
                    start_idx, end_idx = self.specific_condition(left_bound=left_bound, right_bound=right_bound)

                start_val, end_val = grid_preprocess.predicate_transform(bins_list=bins_local,\
                                                            start_idx=start_idx, end_idx=end_idx)
                filter_new.append((item[0], item[1], start_val, end_val))
            else:
                # 不可变的条件
                filter_new.append(item)

        final_meta = local_meta[0], filter_new
        query_text = query_construction.construct_origin_query(\
            query_meta=final_meta, workload=self.workload)
        
        return query_text, final_meta
    

    def generate_desired_query(self, column_config):
        """
        生成符合期望的查询，这里主要是指column的大小限制
        
        Args:
            column_config: 配置要求的字典
            arg2:
        Returns:
            query_text: 查询文本
            query_meta: 查询元信息
        """
        if column_config is None or len(column_config):
            return self.generate_random_query()

    def generate_spec_query(self, column_idx_dict: dict):
        """
        注入所有bins的信息，生成具体查询
        
        Args:
            column_idx_dict:
            arg2:
        Returns:
            query_text:
            query_meta:
        """
        query_text = ""
        local_meta = mv_management.meta_copy(in_meta=self.query_meta)
        filter_new = [] # 新的filter

        for item in local_meta[1]:
            if item[3] == "placeholder":
                curr_column = (self.alias_inverse[item[0]], item[1])
                bins_local = self.bins_global[curr_column]
                schema_name, column_name = self.alias_inverse[item[0]], item[1]
                # 从外部读入start_idx和end_idx
                start_idx, end_idx = column_idx_dict[(schema_name, column_name)]
                start_val, end_val = grid_preprocess.predicate_transform(bins_list=bins_local,\
                                                            start_idx=start_idx, end_idx=end_idx)
                filter_new.append((item[0], item[1], start_val, end_val))
            else:
                filter_new.append(item)

        final_meta = local_meta[0], filter_new
        query_text = query_construction.construct_origin_query(\
            query_meta=final_meta, workload=self.workload)
        
        return query_text, final_meta
    

    def explore_target_queries(self, num, target_config):
        """
        探索目标查询，这里包含可扩展的要求
        目前考虑one-loop的生成策略，之后考虑改成多轮的结果
    
        Args:
            num: 生成查询的数目
            target_config: 目标查询的配置信息，字典的键值配置如下
            {   
                "error_type": "over/under",         # 到底是估大了还是估小了
                "cardinality": (lower, upper),      # 真实基数的范围
                "estimation": (lower, upper),       # 估计基数的范围
                "columns": {                        # 列的信息
                
                }
            }
        Returns:
            query_list:
            meta_list: 
            info_list: 额外信息的列表
        """
    
        # 生成一批查询
        query_list, meta_list, info_list = [], [], []

        # 过滤cardinality/estimation不符合要求的查询
        pass

    def show_grid_info(self,):
        """
        展示格点的信息
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        if hasattr(self, "extra_info_dict") == False:
            self.grid_info_adjust()

        total_grid_num = 1
        for k, v in self.extra_info_dict.items():
            print("column_name = {}. start_idx = {}. end_idx = {}.".\
                  format(k, v[0], v[1]))
            total_grid_num *= (v[1] - v[0])
        
        print("total_grid_num = {}.".format(total_grid_num))
        print("total_cnt_num = {}.".format(np.sum(self.value_cnt_arr)))

        return self.extra_info_dict, total_grid_num

    # @utils.timing_decorator
    def construct_grid_array(self, data_df: pd.DataFrame, column_list: list):
        """
        {Description}

        Args:
            selected_columns: 选择切分的列
            apply_dynamic: 是否应用动态生成mv的技术
        Returns:
            value_cnt_arr: 值计数矩阵
            column_list: 所选择拆分的列
        """
        # 获得bins_dict的相关信息
        bins_list = []  # 函数需要的是由bins组成的列表
        for col in column_list:
            bins_list.append(self.bins_global[col])

        # print("construct_grid_array: data_df = {}. column_list = {}.".\
        #       format(data_df.head(), column_list))

        # 构建grid
        ts = time.time()
        data_df_arr = data_df.values
        te = time.time()
        # print(f"construct_grid_array: convert_arr time = {te - ts:.2f}")
        
        distinct_list, marginal_list, value_cnt_arr = \
            grid_construction.make_grid_data(data_arr=data_df_arr, input_bins = bins_list, process_num=10)
        
        return value_cnt_arr, column_list

    # def construct_grid_elements(self, in_meta, selected_columns):
    #     """
    #     返回构建grid相关所需的元素
    
    #     Args:
    #         in_meta:
    #         selected_columns:
    #     Returns:
    #         value_cnt_arr: 
    #         column_list: 
    #         res_meta: 
    #         distinct_list: 
    #         marginal_list: 
    #         data_df:
    #     """
    #     out_meta = in_meta

    #     # 构建多表的对象
    #     src_meta_list = [out_meta, ]  # 对象元信息组成的列表
    #     data_df, column_list, merged_meta, res_meta_list = self.table_builder.build_joined_tables_object(
    #         src_meta_list=src_meta_list, selected_columns=selected_columns, apply_dynamic=False
    #     )   # 强制不应用mv dynamic的机制

    #     res_meta = res_meta_list[0] # 新的meta信息
        
    #     # 获得bins_dict的相关信息
    #     bins_local, _ = self.construct_local_bins(in_meta[0])
    #     bins_list = []  # 函数需要的是由bins组成的列表

    #     origin_columns = [query_construction.parse_compound_column_name(col_str=col_str, workload=self.workload) \
    #                       for col_str in column_list]
    #     print("bins_local keys = {}. origin_columns = {}.".format(bins_local.keys(), origin_columns))

    #     for col in origin_columns:
    #         bins_list.append(bins_local[col])

    #     # 构建grid  
    #     distinct_list, marginal_list, value_cnt_arr = \
    #         grid_construction.make_grid_data(data_arr=data_df.values, input_bins = bins_list)
        
    #     return value_cnt_arr, column_list, res_meta, distinct_list, marginal_list, data_df
    

    def grid_construction_verification(self, alias_list):
        """
        grid在构建过程中和实际存在不一致的情况，在此进行正确性的验证
    
        Args:
            alias_list: 别名列表
            arg2:
        Returns:
            value_cnt_arr:
            return2:
        """
        sub_meta = mv_management.meta_subset(in_meta=self.top_meta, 
                schema_subset=alias_list, abbr_mapping=self.workload)  # 去一个子集
        meta_str = self.get_meta_repr(in_meta=sub_meta)

        # 获得需要切分的列
        subset_columns = self.get_subset_columns(selected_columns=self.selected_columns, subquery_meta=sub_meta)
        # 子查询一律不使用动态mv的技术，但是这可能会造成问题(三表的话应该还可以，三表以上不好说)
        value_cnt_arr, column_list, res_meta, distinct_list, marginal_list, data_df = self.construct_grid_elements(
            in_meta=sub_meta, selected_columns=subset_columns) # 不应用动态机制

        # 直接返回结果
        return value_cnt_arr, column_list, res_meta, distinct_list, marginal_list, data_df
    
    def release_grid_memory(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        # print("GridPlan: call release_grid_memory.")
        del self.data_df_dict

    def construct_local_bins(self, schema_list):
        """
        构建局部的bins信息
        
        Args:
            schema_list: 查询包含的所有schema
        Returns:
            bins_local:
            reverse_local:
        """
        bins_local, reverse_local = {}, {}

        for tbl, col in self.bins_global.keys():
            if tbl in schema_list:
                # table匹配成功
                bins_local[(tbl, col)] = self.bins_global[(tbl, col)]
                reverse_local[(tbl, col)] = self.reverse_global[(tbl, col)]

        return bins_local, reverse_local
    
# %%
def get_spec_template(workload, method, signature, id, template_dir = "/home/lianyuan/Research/CE_Evaluator/intermediate"):
    """
    直接获得单个template，用于实验测试
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    tmpl_meta_path = p_join(template_dir, workload, \
        "template_obj", method, signature, "meta_info.json")
    tmpl_meta_dict = utils.load_json(tmpl_meta_path)
    tmpl_obj_path = tmpl_meta_dict[id]['info']['path']
    tmpl_obj = utils.load_pickle(tmpl_obj_path)
    return tmpl_obj

# %%
