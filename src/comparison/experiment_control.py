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
from pprint import pprint
import shutil

# %% 单进程的baseline

from baseline.feedback_based import feedback_based_search
from baseline.heuristic import init_heuristic, final_heuristic
from baseline.random import random_search

# %% 多进程的baseline

from baseline.feedback_based import feedback_based_parallel
from baseline.heuristic import init_heuristic_parallel, final_heuristic_parallel
from baseline.random import random_search_parallel
from baseline.reinforcement_learning import rl_search
# %%
from utility import utils, common_config
from baseline.utility import util_funcs
from itertools import product

import random, string

# %%
from plan import plan_template
from experiment import search_wrapping, parallel_wrapping, stateful_wrapping, benefit_oriented_wrapping
from utility import workload_spec

# %%
"""
整一个实验控制的元信息由三个文件组成

单个实验的实例，表示每一次单个方法探索的结果，这样的好处是可以支持多个实验的结果比较

result_instance
{
    "query_list": [],
    "meta_list: [],
    "result_list": [],
    "card_dict_list": []
}

result_meta
{   
    # key是单调递增的ID
    "0": {
        "search_method": "探索的方法",
        "dataset": "数据集",
        "ce_type": "基数估计的方法",
        "total_time": "总的探索时间",
        "query_num": "查询数目",
        "obj_path": "",
        "constraint": {
        
        }
    },
    "1": {
    
    }
}

实验的配置
expt_config_meta
{
    "0": {
        "search_method": [],
        "dataset": [],
        "ce_type": []
        "total_time": []
    },
    "1": {
    
    }
}

实验的执行结果
expt_exec_meta
{
    "0": {
        "expt_config": {
            # 总体的实验配置
            ... ...
        },
        "instance_list": [
            {
                "instance_config": {    # 当前实验的配置信息
                    "search_method": "",
                    "dataset": "",
                    "ce_type": "",
                    "total_time": ""
                }
                "result_id": ""
            },
            {
            
            }
        ]
    }
}

"""
# %%

def generate_random_string(length):
    letters = string.ascii_letters + string.digits  # 包含大小写字母和数字的字符集合
    return ''.join(random.choice(letters) for _ in range(length))

# %%

def merge_config(curr_config, ref_config):
    """
    合并两个配置信息
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    print(f"merge_config: curr_config = {curr_config}. ref_config = {ref_config}.")
    out_config = {}
    merge_keys = set(list(curr_config.keys()) + list(ref_config.keys()))
    for k in merge_keys:
        if k == "reference_id":
            continue
        if k in curr_config and k in ref_config:
            if isinstance(curr_config[k], dict) == True:
                out_config[k] = deepcopy(ref_config[k])
                for kk, vv in curr_config[k].items():
                    out_config[k][kk] = vv
            else:
                out_config[k] = curr_config[k]
        elif k in curr_config:
            out_config[k] = curr_config[k]
        elif k in ref_config:
            out_config[k] = ref_config[k]

    return out_config
# %%

class Controller(object):
    """
    基线实验对比的控制器，用于标准化的执行实验。

    Members:
        field1:
        field2:
    """

    def __init__(self, workload, out_dir = "/home/lianyuan/Research/CE_Evaluator/intermediate", result_meta = "result_meta.json", \
                 expt_config_meta = "expt_config_meta.json", expt_exec_meta = "expt_exec_meta.json"):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        assert workload in ("job", "stats", "dsb")
        self.time_gap = 10                                      # 间隔设置成10s
        self.workload = workload
        # self.report_ctor = ReportConstructor(result_dir="")
        self.out_dir = out_dir
        self.with_time_info = False
        self.load_meta_info(out_dir, result_meta, expt_config_meta, expt_exec_meta)

    def set_time_info(self, flag: bool):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        self.with_time_info = flag

    def load_meta_info(self, out_dir, result_meta, expt_config_meta, expt_exec_meta):
        """
        {Description}
        
        Args:
            out_dir:
            result_meta:
            expt_config_meta:
            expt_exec_meta:
        Returns:
            res1:
            res2:
        """
        self.result_meta_path = p_join(out_dir, self.workload, "experiment_obj", result_meta) 
        self.result_meta_dict = utils.load_json(data_path=self.result_meta_path)
        
        self.expt_config_meta_path = p_join(out_dir, self.workload, "experiment_obj", expt_config_meta) 
        self.expt_config_meta_dict = utils.load_json(data_path=self.expt_config_meta_path)

        self.expt_exec_meta_path = p_join(out_dir, self.workload, "experiment_obj", expt_exec_meta) 
        self.expt_exec_meta_dict = utils.load_json(data_path=self.expt_exec_meta_path)


    def update_meta_info(self,):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        utils.dump_json(self.result_meta_dict, self.result_meta_path)
        utils.dump_json(self.expt_exec_meta_dict, self.expt_exec_meta_path)
        # utils.dump_json(self.expt_config_meta_dict, self.expt_config_meta_path)

    def add_result_instance(self, search_method, ce_type, total_time, query_num, \
            constraint, res_obj, workload, time_info, split_budget, extra_info = None):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            id:
            return2:
        """
        next_id = self.get_next_id(in_dict=self.result_meta_dict)
        
        # 直接拿时间戳作为名字
        out_name = "{}_{}.pkl".format(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()), generate_random_string(5))
        out_path = p_join(self.out_dir, self.workload, "experiment_obj", out_name)

        self.result_meta_dict[next_id] = {
            "search_method": search_method,
            "dataset": workload,
            "estimation_method": ce_type,
            "total_time": total_time,
            "query_num": query_num,
            "obj_path": out_path,
            "constraint": constraint,
            "split_budget": split_budget,
            "time_info": time_info
        }

        # 
        if extra_info is not None:
            self.result_meta_dict[next_id]["extra_info"] = extra_info

        self.update_meta_info()
        utils.dump_pickle(res_obj=res_obj, data_path=out_path)
        return next_id, self.result_meta_dict[next_id]


    def add_expt_exec_list(self, expt_config, out_list):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        next_id = self.get_next_id(in_dict=self.expt_exec_meta_dict)
        self.expt_exec_meta_dict[next_id] = {
            "expt_config": expt_config,
            "instance_list": out_list
        }
        self.update_meta_info()

        return next_id

    def add_expt_exec_instance(self, instance_local):
        """
        {Description}
    
        Args:
            instance_local:
            arg2:
        Returns:
            next_id:
            exec_config:
        """
        instance = instance_local
        # print("Controller.add_expt_exec_instance: instance = {}.".format(instance))
        config_local = {}
        # args_list = ["search_method", "estimation_method", "total_time", "constraint", "res_obj", "workload"]
        args_list = ["search_method", "ce_type", "total_time", "res_obj", "workload", "split_budget"]

        for key in args_list:
            config_local[key] = instance[key]
        
        config_local["query_num"] = len(instance['res_obj'][0]) # 设置生成查询的数目
        config_local["constraint"] = ""
        
        # 直接改成用字典传参数
        search_method = config_local["search_method"]

        # 更新extra_info的记录方法
        if search_method in ["polling_based_parallel", "epsilon_greedy_parallel", "correlated_MAB_parallel"]:
            extra_keys = ["generate_join_order", "generate_ref_query", "generate_template"]
            extra_info = {}

            for k in extra_keys:
                extra_info[k] = instance['custom_options'][k]
                
            config_local['extra_info'] = extra_info

        if search_method in ["stateful_parallel", ]:
            extra_keys = ["generate_join_order", "generate_ref_query", "generate_template",
                "estimator", "schedule_mode", "action_selection_mode", "root_selection_mode", "template_config"]
            extra_info = {}

            for k in extra_keys:
                if k in instance['custom_options']:
                    extra_info[k] = instance['custom_options'][k]
                
            config_local['extra_info'] = extra_info

        # 确认是否包含数据
        config_local['time_info'] = self.with_time_info

        # 添加结果实例
        # print("Controller.add_expt_exec_instance: config_local = {}.".format(config_local))
        next_id, res_dict = self.add_result_instance(**config_local)
        
        del config_local['res_obj']
        # out_list.append({"instance_config": config_local, "result_id": next_id })
        self.update_meta_info()
        return next_id, {"instance_config": config_local, "result_id": next_id }


    def instance_config_enumeration(self, config_dict):
        """
        通过config_dict枚举实验的配置
        
        Args:
            config_dict:
            arg2:
        Returns:
            config_list:
            res2:
        """

        key_list, val_list = utils.dict2list(config_dict)
        # print("Controller.instance_config_enuemration: val_list = {}.".format(val_list))
        iter_list = []  # 用于进行迭代的列表
        for item in val_list:
            if isinstance(item, list):
                iter_list.append(item)
            elif isinstance(item, (int, str, float)):
                iter_list.append([item])
            elif isinstance(item, dict):
                iter_list.append([item,])
            else:
                raise TypeError("Controller.instance_config_enumeration: item_type = {}.".format(type(item)))
            
        val_comb_list = product(*iter_list)
        config_list = []
        for val_instance_list in val_comb_list:
            config_instance = util_funcs.list2dict(key_list, val_instance_list)
            # result = self.run_experiment(config_instance)
            config_list.append(config_instance)

        return config_list

    def get_next_id(self, in_dict: dict):
        """
        {Description}
        
        Args:
            in_dict:
        Returns:
            out_id:
        """
        # print("experiment_control.get_next_id: in_dict = {}.".format(in_dict))
        int_id_list = [int(k) for k in in_dict.keys()]
        if len(int_id_list) == 0:
            return str(0)
        return str(max(int_id_list) + 1)
    
    def get_expt_id_by_config(self, info_dict: dict):
        """
        {Description}
    
        Args:
            info_dict:
            arg2:
        Returns:
            return1:
            return2:
        """
        candidate_list = []
        def dict_match(info_dict: dict, ref_dict: dict):
            flag = True
            for k, v in info_dict.items():
                # if isinstance(v, (float, int, str)):
                if isinstance(v, (float, int)):
                    if v != ref_dict[k]:
                        print(f"get_expt_id_by_config.dict_match: v = {v}. ref_dict[k] = {ref_dict[k]}.")
                        flag = False
                        break
                elif isinstance(v, str):
                    # string类型不区分大小写
                    if v.lower() != ref_dict[k].lower():
                        print(f"get_expt_id_by_config.dict_match: v = {v}. ref_dict[k] = {ref_dict[k]}.")
                        flag = False
                        break
                if isinstance(v, (list, tuple)):
                    if set(v) != set(ref_dict[k]):
                        print(f"get_expt_id_by_config.dict_match: v = {v}. ref_dict[k] = {ref_dict[k]}.")
                        flag = False
                        break
            return flag

        for expt_id, expt_config in self.expt_config_meta_dict['execution'].items():
            if dict_match(info_dict, expt_config) == True:
                candidate_list.append(expt_id)

        if len(candidate_list) == 0:
            return None
        elif len(candidate_list) > 1:
            return None
        else:
            return candidate_list[0]


    def parse_expt_config(self, expt_config):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        if isinstance(expt_config, (int, str)):
            # ID转成字符串类型的
            # expt_config = self.expt_config_meta_dict['execution'][str(expt_config)]
            # 考虑带有reference_id的情况
            expt_config = self.expt_config_meta_dict['execution'][str(expt_config)]
            if "reference_id" in expt_config:
                ref_config = self.parse_reference_config(expt_config["reference_id"])
                del expt_config["reference_id"]
            else:
                ref_config = {}
            # for k, v in ref_config.items():
            #     if k not in expt_config:
            #         expt_config[k] = v
            expt_config = merge_config(expt_config, ref_config)
            # print(f"parse_expt_config: expt_config = {expt_config}")
        elif isinstance(expt_config, dict):
            expt_config = expt_config
        else:
            raise ValueError("Controller.launch_experiment_process: Unsupported expt_config type({})".\
                             format(type(expt_config)))
        
        # 删除无关的条件
        if "description" in expt_config:
            del expt_config['description']

        # pprint(expt_config)
        return expt_config

    def parse_reference_config(self, ref_id):
        """
        加载引用的实验配置
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        # print(f"parse_reference_config: ref_id = {ref_id}.")
        curr_config: dict = self.expt_config_meta_dict['execution'][ref_id]
        if "reference_id" in curr_config.keys():
            ref_config = self.parse_reference_config(curr_config['reference_id'])
        else:
            ref_config = {}

        # res_config = {}
        # for k, v in ref_config.items():
        #     res_config[k] = v

        # for k, v in curr_config.items():
        #     if k == "reference_id":
        #         continue
        #     res_config[k] = v
        # 优化：直接调用合并函数
        res_config = merge_config(curr_config, ref_config)
        # print(f"parse_reference_config: res_config = {res_config}")
        return res_config

    def launch_experiment_process(self, expt_config, ignore_error = False, \
            save_result = True, extra_params:dict = {}):
        """
        启动一次完整的实验
    
        Args:
            expt_config:
            ignore_error:
            save_result:
            extra_params:
        Returns:
            return1:
            return2:
        """
        print(f"self.expt_config_meta_dict = {self.expt_config_meta_dict}")
        expt_config = self.parse_expt_config(expt_config)
        
        print("launch_experiment_process: expt_config = {}.".format(expt_config))
        config_list = self.instance_config_enumeration(config_dict=expt_config)
        print("launch_experiment_process: config_list = {}.".format(config_list))

        expt_result_list = []
        # instance_list = []
        out_list = []
        for conf in config_list:
            try:
                print(f"launch_experiment_process: conf_before = {conf}.")
                # 将extra_params加到每一个conf中去
                for k, v in extra_params.items():
                    # 2024-03-20: 支持dict_merge
                    if isinstance(v, dict):
                        conf[k].update(v)
                    else:
                        conf[k] = v
                print(f"launch_experiment_process: conf_after = {conf}")
                result_local = self.run_single_experiment(config_dict=conf)
                expt_result_list.append(result_local)
                instance_local = deepcopy(conf)
                instance_local['res_obj'] = result_local
                #
                if save_result == True:
                    next_id, exec_config = self.add_expt_exec_instance(instance_local)
                    print(f"launch_experiment_process: finish one expt. conf = {conf}. next_id = {next_id}.")
                    out_list.append(exec_config)

            except Exception as e:
                if ignore_error == False:
                    raise e
                print(f"launch_experiment_process: In config_list iteration. meet exception = {e}.")
                time.sleep(60)

            # 暂停程序以释放资源
            time.sleep(self.time_gap)

        if save_result == True:
            self.add_expt_exec_list(expt_config, out_list)
        return out_list
    

    def get_experiment_object(self, expt_config, extra_params:dict = {}, expt_idx = 0):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        # print(f"self.expt_config_meta_dict = {self.expt_config_meta_dict}")
        expt_config = self.parse_expt_config(expt_config)
        
        # print("launch_experiment_process: expt_config = {}.".format(expt_config))
        config_list = self.instance_config_enumeration(config_dict=expt_config)
        # print("launch_experiment_process: config_list = {}.".format(config_list))

        conf = config_list[expt_idx]
        print(f"launch_experiment_process: conf_before = {conf}.")
        # 将extra_params加到每一个conf中去
        for k, v in extra_params.items():
            # 2024-03-20: 支持dict_merge
            if isinstance(v, dict):
                conf[k].update(v)
            else:
                conf[k] = v
        print(f"launch_experiment_process: conf_after = {conf}")
        # expt_obj = self.run_single_experiment(config_dict=conf)
        expt_obj = self.construct_stateful_object(config_dict=conf)

        return expt_obj    

    def show_experiment_configurations(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        def config_info_str(in_dict):
            res_str = f"method_list: {in_dict['search_method']}\n"\
                f"total_time: {in_dict['total_time']}. "\
                f"ce_type: {in_dict['ce_type']}. "\
                f"workload: {in_dict['workload']}"
            return res_str

        print("Show Execution Configurations:")
        for k, v in self.expt_config_meta_dict['execution'].items():
            self.parse_expt_config(expt_config=k)
            print("ID: {}. Config: \n{}.".format(k, config_info_str(v)))

        print("Show Comparison Configurations:")
        for k, v in self.expt_config_meta_dict['execution'].items():
            print("ID: {}. Config: {}.".format(k, v))
        

    def run_baseline_experiment(self, config_dict):
        """
        运行baseline的实验
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        print("Controller.run_single_experiment: config_dict = {}.".format(json.dumps(config_dict, indent=4)))

        search_method: str = config_dict['search_method']
        expt_instance = None
        
        workload = config_dict['workload']

        if config_dict['schema_total'] == "all":
            schema_total = workload_spec.total_schema_dict[workload]
        else:
            schema_total = config_dict['schema_total']

        print(f"run_baseline_experiment: schema_total = {schema_total}.")
        
        expt_init_config = {
            "schema_total": schema_total, 
            "workload": workload, 
            "time_limit": config_dict['time_limit'],
            "ce_type": config_dict['ce_type']
        }


        if search_method.lower() == "random":
            # 概率转化成确定的结果
            expt_init_config["table_num_dist"] = {
                config_dict['table_num']: 1.0
            }
            expt_instance = random_search.RandomPlanSearcher(**expt_init_config)
        elif search_method.lower() == "init_heuristic":
            expt_instance = init_heuristic.InitGreedyPlanSearcher(**expt_init_config)
        elif search_method.lower() == "final_heuristic":
            expt_instance = final_heuristic.FinalGreedyPlanSearcher(**expt_init_config)
        elif search_method.lower() == "feedback_based":
            # 使用默认["schema", "column"]的模式
            expt_init_config['mode'] = ["schema", "column"]
            expt_instance = feedback_based_search.FBBasedPlanSearcher(**expt_init_config)
        elif search_method.lower() == "random_parallel":
            expt_init_config["table_num_dist"] = { config_dict['table_num']: 1.0 }
            expt_instance = random_search_parallel.RandomParallelSearcher(**expt_init_config)
        elif search_method.lower() == "init_heuristic_parallel":
            expt_instance = init_heuristic_parallel.InitGreedyParallelSearcher(**expt_init_config)
        elif search_method.lower() == "final_heuristic_parallel":
            expt_instance = final_heuristic_parallel.FinalGreedyParallelSearcher(**expt_init_config)
        elif search_method.lower() == "feedback_based_parallel":
            expt_init_config['mode'] = ["schema", "column"]
            expt_instance = feedback_based_parallel.FBBasedParallelSearcher(**expt_init_config)
        else:
            raise ValueError("Controller.run_experiment: Unsupported search_method({})".format(search_method))
        
        if self.with_time_info == False:
            expt_run_config = {
                "total_time": config_dict['total_time'],
                "with_start_time": False
            }
            query_list, result_list, meta_list, card_dict_list = \
                expt_instance.launch_search_process(**expt_run_config)

            # 保存结果
            return query_list, result_list, meta_list, card_dict_list
        else:
            expt_run_config = {
                "total_time": config_dict['total_time'],
                "with_start_time": True
            }
            query_list, result_list, meta_list, card_dict_list, time_info = \
                expt_instance.launch_search_process(**expt_run_config)

            # 保存结果
            return query_list, result_list, meta_list, card_dict_list, time_info


    def run_RL_experiment(self, config_dict):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        print("Controller.run_RL_experiment: config_dict = {}.".format(json.dumps(config_dict, indent=4)))

        search_method: str = config_dict['search_method']
        expt_instance = None
        
        workload = config_dict['workload']

        if config_dict['schema_total'] == "all":
            schema_total = workload_spec.total_schema_dict[workload]
        else:
            schema_total = config_dict['schema_total']

        print(f"run_baseline_experiment: schema_total = {schema_total}.")
        
        expt_init_config = {
            "schema_total": schema_total, 
            "workload": workload, 
            "time_limit": config_dict['time_limit'],
            "ce_type": config_dict['ce_type'],
            "table_num_limit": config_dict['table_num']
        }

        if search_method.lower() == "learnedsql":
            expt_instance = rl_search.RLBasedPlanSearcher(**expt_init_config)

        if self.with_time_info == False:
            expt_run_config = {
                "total_time": config_dict['total_time'],
                "with_start_time": False
            }
            query_list, result_list, meta_list, card_dict_list = \
                expt_instance.launch_search_process(**expt_run_config)

            # 保存结果
            return query_list, result_list, meta_list, card_dict_list
        else:
            expt_run_config = {
                "total_time": config_dict['total_time'],
                "with_start_time": True
            }
            query_list, result_list, meta_list, card_dict_list, time_info = \
                expt_instance.launch_search_process(**expt_run_config)

            # 保存结果
            return query_list, result_list, meta_list, card_dict_list, time_info
        

    def run_custom_experiment(self, config_dict):
        """
        运行自己设计的实验
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        search_method = config_dict['search_method']
        query_list, result_list, meta_list, card_dict_list = [], [], [], []
        expt_init_config = {}

        for param in ["schema_total", "workload", "time_limit", "ce_type"]:
            expt_init_config[param] = config_dict[param]

        if search_method == "epsilon_greedy":
            expt_init_config['expt_config'] = {
                "table_num": 5,
                "namespace": "history",
                "sample_queries": False,
                "generate_template": False,
                "explore_mode": "epsilon_greedy"  
            }
        elif search_method == "polling_based":
            expt_init_config['expt_config'] = {
                "table_num": 5,
                "namespace": "history",
                "sample_queries": False,
                "generate_template": False,
                "explore_mode": "polling_based"
            }
            
        expt_instance = search_wrapping.StateAwareSearcher(**expt_init_config)

        search_result = expt_instance.launch_search_process(total_time=config_dict['total_time'])
        query_list, result_list, meta_list, card_dict_list = search_result

        return query_list, result_list, meta_list, card_dict_list

    def run_advance_experiment(self, config_dict):
        """
        {Description}
    
        Args:
            config_dict:
            arg2:
        Returns:
            query_list: 
            result_list: meta_list, card_dict_list
        """
        workload = config_dict['workload']
        if config_dict['schema_total'] == "all":
            config_dict['schema_total'] = workload_spec.total_schema_dict[workload]

        search_method, ce_type = config_dict['search_method'], config_dict['ce_type']
        query_list, result_list, meta_list, card_dict_list = [], [], [], []
        expt_init_config = {}

        for param in ["schema_total", "workload", "time_limit", "ce_type"]:
            expt_init_config[param] = config_dict[param]
        
        custom_options = config_dict['custom_options']

        try:
            signature = custom_options['signature']
            meta_path = plan_template.get_template_meta_path(workload, ce_type, signature)
            expt_init_config['tmpl_meta_path'] = meta_path
        except KeyError: 
            meta_path = plan_template.get_template_meta_path(workload, ce_type, "")
            expt_init_config['tmpl_meta_path'] = meta_path

        # expt_instance = parallel_wrapping.StateAwareParallelSearcher(**expt_init_config)
        expt_instance = parallel_wrapping.FeedbackAwareParallelSearcher(**expt_init_config)

        config_input = utils.get_sub_dict(custom_options, \
            ["generate_join_order", "generate_ref_query", "generate_template"])
        
        config_input["explore_mode"] = search_method
        expt_instance.set_config(**config_input)

        search_result = expt_instance.launch_search_process(total_time=config_dict['total_time'])
        query_list, result_list, meta_list, card_dict_list = search_result

        return query_list, result_list, meta_list, card_dict_list
    

    def construct_stateful_object(self, config_dict: dict):
        """
        获得stateful搜索器对象
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        workload = config_dict['workload']
        if config_dict['schema_total'] == "all":
            config_dict['schema_total'] = workload_spec.total_schema_dict[workload]

        search_method, ce_type = config_dict['search_method'], config_dict['ce_type']
        query_list, result_list, meta_list, card_dict_list = [], [], [], []
        expt_init_config = {}

        for param in ["schema_total", "workload", "time_limit", "ce_type", "split_budget"]:
            expt_init_config[param] = config_dict[param]

        custom_options = config_dict['custom_options']

        try:
            signature = custom_options['signature']
            meta_path = plan_template.get_template_meta_path(workload, ce_type, signature)
            template_dir = common_config.get_template_dir(workload, ce_type, signature)
            expt_init_config['tmpl_meta_path'] = meta_path
        except KeyError: 
            meta_path = plan_template.get_template_meta_path(workload, ce_type, "")
            template_dir = common_config.get_template_dir(workload, ce_type, signature)
            expt_init_config['tmpl_meta_path'] = meta_path

        # 设置expl_estimator和resource_config变量
        estimator_mapping = {
            "graph_based": "graph_corr_based",
            "external": "external"
        }
        resource_mapping = {
            "static": common_config.default_resource_config,
            "dynamic": common_config.dynamic_resource_config
        }
 
        def process_custom():
            option_dict = {
                "estimator": ("card_est_input", lambda a: estimator_mapping[a]),
                "schedule_mode": ("resource_config", lambda a: resource_mapping[a]),
                "action_selection_mode": ("action_selection_mode", lambda a: a),
                "root_selection_mode": ("root_selection_mode", lambda a: a),
                "exploration_est": ("expl_estimator", lambda a: a),
                "noise_parameters": ("noise_parameters", lambda a: tuple(a)),
                "template_config": ("template_config", lambda a: a),
                "sample_config": ("sample_config", lambda a: a)
            }
            nonlocal expt_init_config
            for option_name, (config_name, proc_func) in option_dict.items():
                if option_name in custom_options:
                    expt_init_config[config_name] = proc_func(custom_options[option_name])

        process_custom()

        try:
            config_input = utils.get_sub_dict(custom_options, \
                ["generate_join_order", "generate_ref_query", "generate_template"])
        except KeyError as e:
            print(f"run_stateful_experiment: meet KeyError. custom_options = {custom_options}")
            raise e
        
        # 对于config_input进行特判
        if config_input["generate_template"] == True:
            print(f"run_stateful_experiment: generate_template = True. template_dir = {template_dir}.")
            if os.path.isdir(template_dir) == True:
                shutil.rmtree(template_dir)

        # 2024-03-28: 
        if "warm_up_config" in custom_options.keys():
            local_config = custom_options["warm_up_config"]
            max_expl_step, warm_up_num = local_config["max_expl_step"], local_config["warm_up_num"]
            expt_init_config["max_step"] = max_expl_step
            common_config.warm_up_num = warm_up_num
            print(f"run_stateful_experiment: warm_up_config. max_expl_step = {max_expl_step}. warm_up_num = {warm_up_num}.")

        # 创建搜索器实例
        if search_method == "stateful":
            expt_instance = stateful_wrapping.StatefulParallelSearcher(**expt_init_config)
        elif search_method == "benefit_oriented":
            expt_instance = benefit_oriented_wrapping.BenefitOrientedSearcher(**expt_init_config)
        else:
            raise ValueError(f"construct_stateful_object: invalid search_method = {search_method}.")
        
        config_input["explore_mode"] = search_method
        expt_instance.set_config(**config_input)

        return expt_instance

    def run_stateful_experiment(self, config_dict: dict):
        """
        {Description}
    
        Args:
            config_dict:
            arg2:
        Returns:
            return1:
            return2:
        """
        workload = config_dict['workload']
        if config_dict['schema_total'] == "all":
            config_dict['schema_total'] = workload_spec.total_schema_dict[workload]

        search_method, ce_type = config_dict['search_method'], config_dict['ce_type']
        query_list, result_list, meta_list, card_dict_list = [], [], [], []
        expt_init_config = {}

        for param in ["schema_total", "workload", "time_limit", "ce_type", "split_budget"]:
            expt_init_config[param] = config_dict[param]

        custom_options = config_dict['custom_options']

        try:
            signature = custom_options['signature']
            meta_path = plan_template.get_template_meta_path(workload, ce_type, signature)
            template_dir = common_config.get_template_dir(workload, ce_type, signature)
            expt_init_config['tmpl_meta_path'] = meta_path
        except KeyError: 
            meta_path = plan_template.get_template_meta_path(workload, ce_type, "")
            template_dir = common_config.get_template_dir(workload, ce_type, signature)
            expt_init_config['tmpl_meta_path'] = meta_path

        # 设置expl_estimator和resource_config变量
        estimator_mapping = {
            "graph_based": "graph_corr_based",
            "external": "external"
        }
        resource_mapping = {
            "static": common_config.default_resource_config,
            "dynamic": common_config.dynamic_resource_config
        }
 
        def process_custom():
            # 处理个性化配置
            dummy_func = lambda a: a
            option_dict = {
                "estimator": ("card_est_input", lambda a: estimator_mapping[a]),
                "schedule_mode": ("resource_config", lambda a: resource_mapping[a]),
                "action_selection_mode": ("action_selection_mode", dummy_func),
                "root_selection_mode": ("root_selection_mode", dummy_func),
                "exploration_est": ("expl_estimator", dummy_func),
                "noise_parameters": ("noise_parameters", lambda a: tuple(a)),
                "template_config": ("template_config", dummy_func),
                "sample_config": ("sample_config", dummy_func)
            }
            nonlocal expt_init_config
            for option_name, (config_name, proc_func) in option_dict.items():
                if option_name in custom_options:
                    expt_init_config[config_name] = proc_func(custom_options[option_name])

        process_custom()

        try:
            config_input = utils.get_sub_dict(custom_options, \
                ["generate_join_order", "generate_ref_query", "generate_template"])
        except KeyError as e:
            print(f"run_stateful_experiment: meet KeyError. custom_options = {custom_options}")
            raise e
        
        # 对于config_input进行特判
        if config_input["generate_template"] == True:
            print(f"run_stateful_experiment: generate_template = True. template_dir = {template_dir}.")
            if os.path.isdir(template_dir) == True:
                shutil.rmtree(template_dir)

        # 2024-03-28: 
        if "warm_up_config" in custom_options.keys():
            local_config = custom_options["warm_up_config"]
            max_expl_step, warm_up_num = local_config["max_expl_step"], local_config["warm_up_num"]
            expt_init_config["max_step"] = max_expl_step
            common_config.warm_up_num = warm_up_num
            print(f"run_stateful_experiment: warm_up_config. max_expl_step = {max_expl_step}. warm_up_num = {warm_up_num}.")

        # 创建不同类型的搜索器实例
        if search_method == "stateful_parallel":
            expt_instance = stateful_wrapping.StatefulParallelSearcher(**expt_init_config)
        elif search_method == "benefit_oriented":
            expt_instance = benefit_oriented_wrapping.BenefitOrientedSearcher(**expt_init_config)
        else:
            raise ValueError(f"run_stateful_experiment: search_method = {search_method}.")
        
        config_input["explore_mode"] = search_method
        expt_instance.set_config(**config_input)

        if self.with_time_info == False:
            expt_run_config = {
                "total_time": config_dict['total_time'],
                "with_start_time": False,
                "template_only": config_dict.get('template_only', False)
            }
            search_result = expt_instance.launch_search_process(**expt_run_config)
            query_list, result_list, meta_list, card_dict_list = search_result
            # 保存结果
            return query_list, result_list, meta_list, card_dict_list
        else:
            expt_run_config = {
                "total_time": config_dict['total_time'],
                "with_start_time": True,
                "template_only": config_dict.get('template_only', False)
            }
            search_result = expt_instance.launch_search_process(**expt_run_config)
            query_list, result_list, meta_list, card_dict_list, time_info = search_result
            # 保存结果
            return query_list, result_list, meta_list, card_dict_list, time_info


    def run_single_experiment(self, config_dict: dict):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        # exit(-1)    # 直接退出，用做测试

        # 用来对比的基线方法
        baseline_list = ["init_heuristic", "final_heuristic", "random", "feedback_based", \
            "init_heuristic_parallel", "final_heuristic_parallel", "random_parallel", "feedback_based_parallel"]
        # 用来对比的强化学习方法
        RL_list = ["learnedsql"]
        # 消融实验列表
        ablation_list = [""]
        # 单进程自己的方法
        custom_list = ["polling_based", "epsilon_greedy", "correlated_MAB"]
        # 多进程自己的方法
        advance_list = ["polling_based_parallel", "epsilon_greedy_parallel", "correlated_MAB_parallel"]
        # 基于状态的方法
        stateful_list = ["stateful_parallel", "benefit_oriented"]

        search_method = config_dict["search_method"]

        if search_method in baseline_list:
            expt_res = self.run_baseline_experiment(config_dict=config_dict)
        elif search_method in RL_list:
            expt_res = self.run_RL_experiment(config_dict=config_dict)
        elif search_method in custom_list:
            expt_res = self.run_custom_experiment(config_dict=config_dict)
        elif search_method in advance_list:
            expt_res = self.run_advance_experiment(config_dict=config_dict)
        elif search_method in stateful_list:
            expt_res = self.run_stateful_experiment(config_dict=config_dict)
        else:
            raise ValueError(f"run_single_experiment: unsupported search_method. search_method = {search_method}.")
        
        return expt_res


# %%
