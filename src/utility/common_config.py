#!/usr/bin/env python
# -*- coding: utf-8 -*-


# %%
from utility import workload_spec
from os.path import join as p_join
import time, os
# %%

default_resource_config = {
    "total_proc_num": 30, "proc_num_per_task": 5,   
    "short_proc_num": 20, "long_proc_num": 0,
    "assign_mode": "stable", "long_task_threshold": 1e8,
}

# 
dynamic_resource_config = {
    "total_proc_num": 30, "proc_num_per_task": 5,   
    "short_proc_num": 20, "long_proc_num": 10,
    "assign_mode": "dynamic", "shift_factor": 2,
    "long_task_threshold": 10,
}

def get_template_meta_path(workload, method, signature = None, \
        common_dir = "/home/lianyuan/Research/CE_Evaluator/intermediate"):
    """
    获得模版元数据的路径

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    if signature is None or len(signature) == 0:
        meta_path = p_join(common_dir, workload, "template_obj", method, "meta_info.json")
    else:
        meta_path = p_join(common_dir, workload, "template_obj", \
                        method, signature, "meta_info.json")
    return meta_path


def get_template_dir(workload, method, signature = None, \
        common_dir = "/home/lianyuan/Research/CE_Evaluator/intermediate"):
    """
    {Description}

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    if signature is None or len(signature) == 0:
        res_dir = p_join(common_dir, workload, "template_obj", method)
    else:
        res_dir = p_join(common_dir, workload, "template_obj", \
                        method, signature)
    return res_dir


# %%

tree_config_dict = {
    "job": { "max_depth":6, "timeout": 120000 },
    "stats": { "max_depth":8, "timeout": 120000 },
    "dsb": { "max_depth":8, "timeout": 120000 }
}
# %%

parallel_base_config = {
    "explorer_params": {
        "workload": "job",
        "expt_config": {
            "selected_tables": workload_spec.total_schema_dict["job"],
            "ce_handler": "DeepDB_jct"
        },
        "expl_estimator": "graph_corr_based",
        "resource_config": default_resource_config,
        "max_expl_step": 10,
        # "tmpl_meta_path": "/home/lianyuan/Research/CE_Evaluator/intermediate/job/"
        "tmpl_meta_path": get_template_meta_path(workload="job", method="DeepDB_jct", signature="test_1002")
    },
    "generation_params": {
        "template_id_list": "all",
        "root_config": {    # 创建根节点的配置
            "min_card": 5000, 
            "max_card": 10000000000
        }, 
        "tree_config": {
            "max_depth": 6,
        },
        "search_config": {
            "max_step": 10,
            "return": "full"
        },
        "total_time": 120
    }
}

# %%
def construct_spec_params(config_name = "parallel_base_config", \
    workload = None, method = None, signature = "test_1002", total_time = None):
    """
    {Description}
    
    Args:
        config_name:
        workload:
        method:
        total_time:
    Returns:
        explorer_params: 
        generation_params:
    """
    config_dict = globals()[config_name]

    explorer_params = config_dict['explorer_params']
    generation_params = config_dict['generation_params']
    
    if workload is not None:
        explorer_params['workload'] = workload
        explorer_params['expt_config']["selected_tables"] = \
            workload_spec.total_schema_dict[workload]
    
    if method is not None:
        explorer_params['expt_config']["ce_handler"] = method

    if signature is not None:
        workload = explorer_params['workload']
        method = explorer_params['expt_config']['ce_handler']
        explorer_params['tmpl_meta_path'] = \
            get_template_meta_path(workload, method, signature)
    
    if total_time is not None:
        generation_params['total_time'] = total_time
    return explorer_params, generation_params


# %% 计划收益估计器的参数

default_strategy = {
    "option": "dummy"
}

equal_diff_strategy = {
    "option": "equal_diff",
    "start": 0.5,
    "end": 2,
    "number": 4
}

equal_ratio_strategy = {
    "option": "equal_ratio",
    "start": 0.5,
    "end": 2,
    "number": 4
}

option_collections = {
    "default": {
        "estimator": "built-in",
        "strategy": default_strategy,
    },
    "equal_diff": {
        "estimator": "built-in",
        "strategy":equal_diff_strategy,
    },
    "equal_ratio": {
        "estimator": "built-in",
        "strategy": equal_ratio_strategy
    },
    "graph_corr_based": {
        "estimator": "graph_corr",
        "strategy": {
            "mode": "kde"
        }
    },
    "external": {
        "estimator": "external",
        "model_type": "dropout_bayes",
        "internal": "graph_corr_based",
        "url": "http://101.6.96.160:30007",
        "sample_num": 5
    }
}

# %% HTTP请求的超时限制
http_timeout = 10       # 最低的timeout设置为10s
unit_timeout = 0.4      # 单位查询的超时

def cal_timeout(query_num):
    return max(http_timeout, int(query_num * unit_timeout))

# %% 全局列划分的预算

global_split_budget = 200
def set_split_budget(new_budget):
    assert 10 <= new_budget <= 1000
    global global_split_budget
    global_split_budget = new_budget


def get_split_budget():
    return global_split_budget

# %% StatefulExploration.explorate_query_on_template中的参数

tree_mode = "normal"

def set_tree_mode(new_mode: str):
    global tree_mode
    assert new_mode in ("normal", "advance")
    tree_mode = new_mode

exploration_mode, exploration_eval_num = "hybrid-split", 5
# exploration_mode, exploration_eval_num = "card-split", 5
card_distance, meta_distance, hybrid_alpha = 2.0, 0.1, 0.5

def set_exploration_params(new_mode: str = None, new_eval_num: int = None):
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    global exploration_mode
    global exploration_eval_num

    if new_mode is not None:
        assert new_mode in ("random", "greedy", "card-split", "meta-split", "hybrid-split", "feedback-based")
        exploration_mode = new_mode

    if new_eval_num is not None:
        exploration_eval_num = new_eval_num

def set_exploration_distances(card_new = None, meta_new = None, hybrid_new = None):
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    if card_new is not None:
        global card_distance
        card_distance = card_new
    
    if meta_new is not None:
        global meta_distance
        meta_distance = meta_new

    if hybrid_new is not None:
        global hybrid_alpha
        hybrid_alpha = hybrid_new


# %% StatefulExploration中相关配置
# parallel_workload_generation中策略设置

warmup_barrier = False      # 预热屏障，指warm up阶段是否允许task_selector启动新的任务

def set_warmup_barrier(flag: bool):
    global warmup_barrier
    warmup_barrier = flag

warm_up_num = 3
warmup_timeout = 60000      # 超时缩短到20s

# %% PlanAnalyzer Evaluation过程相关分析参数，主要是平衡探索质量和探索时间

eval_config = {
    "meta_num": 3,
    "total_num": 20,
    "loop_num": 3,
    "target_num": 3
}

# 初始化动作的策略
init_strategy = "multi-loop"

def set_eval_param(k: str, v: int):
    global eval_config
    assert k in eval_config

    eval_config[k] = v

# %% BenefitEstimation相关参数，主要是平衡估计质量和估计时间

benefit_config = {
    # "": "",
    # "": "",
    "plan_sample_num": 50
}

def set_benefit_param(k: str, v):
    global benefit_config
    assert k in benefit_config

    benefit_config[k] = v

# init condition estiamtion下的参数
global_sample_num, global_scale = 0, 0
out_case_num = 4
reference_num = 5   # 收益参考比对的最大个数

def set_noise_parameters(new_num: int, new_scale: float):
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    global global_sample_num
    global global_scale

    assert new_num >= 0, f"set_noise_parameters: new_num = {new_num}"
    assert new_scale >= 0.0, f"set_noise_parameters: new_scale = {new_scale}"

    global_sample_num = new_num
    global_scale = new_scale

# %% asynchronous相关参数

# 每次实验生成一个时间戳，作为asynchronous结果保存的目录
async_dir = "/home/lianyuan/Research/CE_Evaluator/intermediate/async"
async_prefix = time.strftime("%Y%m%d%H%M%S", time.localtime())
# print(f"common_config: async_prefix = {async_prefix}.")
os.makedirs(p_join(async_dir, async_prefix), exist_ok=True)

# %% task_management相关参数

interval_max, interval_min = 5.0, 0.5     # 最长的

def calculate_gap_time(eval_time):
    # eval_time指当前处理状态消耗的时间
    if eval_time > interval_max - interval_min:
        return interval_min
    else:
        return interval_max - eval_time

# %% template创建的相关参数

template_table_num = 3
template_parallel_num = 3

dynamic_creation_config = {
    'time': 30, 
    'cardinality': 20000000
}

enable_new_template = True
# %% Root Selector的相关配置

num_limit = 5
error_threshold = 1.0       # 这里的threshold感觉不一定需要

# %% merge_all_tables的时间限制

merge_time_limit = 60

# %% extend过程查询计划比较

comparison_mode = "leading"

def set_comparison_mode(new_mode):
    assert new_mode in ("leading", "plan")
    global comparison_mode
    comparison_mode = new_mode

# %% 用于迭代处理结果的基本信息

workload_list = ["job", "stats", "dsb"]
search_method_list = ["feedback_based_parallel", "final_heuristic_parallel", 
    "init_heuristic_parallel", "random_parallel", "learnedsql"]
ce_method_list = ["DeepDB_jct", "DeepDB_rdc", "FCN", "FCNPool", "internal", "MSCN", "SQLServer"]

# %%
