#!/usr/bin/env python
# -*- coding: utf-8 -*-


# %%
from os.path import join as p_join
from experiment import template_evaluation, parallel_exploration, stateful_exploration
from itertools import product
from utility import workload_spec, utils
from data_interaction import data_management, mv_management
from result_analysis import case_analysis
from estimation import case_based_estimation
from grid_manipulation import grid_preprocess

# %%

def get_template_evaluator():
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    workload = "job"
    method = "DeepDB_jct"
    signature = "test_1002"
    common_dir = "/home/lianyuan/Research/CE_Evaluator/intermediate"
    # eval_time = 600     # 一次探索10min
    eval_time = 600     # 一次探索2min，用于测试

    # 创建parallel_explorer
    # 相关实验配置
    expt_config = { 
        "selected_tables": workload_spec.total_schema_dict[workload],
        "ce_handler": method
    }
    meta_path = p_join(common_dir, workload, "template_obj", method, signature, "meta_info.json")
    print(f"eval_multi_template_exploration: workload = {workload}. method = {method}. meta_path = {meta_path}")

    resource_config = parallel_exploration.default_resource_config

    #
    init_query_config = {
        "min_card": 500,
        "max_card": 1000000,
        # "mode": "bayesian"     # 采用贝叶斯探索
        "mode": "sample-based",     # 采用贝叶斯探索
        "num": 100
    }

    tree_config = {
        "max_depth": 6, "timeout": 60000
    }
    # 
    parallel_expt = parallel_exploration.ParallelForestExploration(\
        workload=workload, expt_config=expt_config, resource_config=\
        resource_config, tmpl_meta_path=meta_path, init_query_config=init_query_config,
        tree_config=tree_config)

    tmpl_evaluator = template_evaluation.TemplateEvaluator(test_explorer=parallel_expt)
    return tmpl_evaluator

# %%

def get_variation_test_objects():
    """
    获得变换测试所需要的对象
    
    Args:
        arg1:
        arg2:
    Returns:
        multi_case_analyzer: 
        case_based_estimator: 
        data_manager: 
        bins_builder:
        target_case:
    """
    data_path = "./test_data/new_template_1_instance_list1.pkl"
    workload = "job"
    res_format = ("list", "tree", ("query", "meta", "card_dict", "target_table"))

    multi_case_analyzer = case_analysis.MultiCaseAnalyzer(workload, res_format)
    multi_case_analyzer.load_result(data_path)
    # multi_case_analyzer.show_result()

    # 选择table修改condition

    case_based_estimator = case_based_estimation.CaseBasedEstimator(workload)
    data_manager = data_management.DataManager(wkld_name = workload)
    mv_manager = mv_management.MaterializedViewManager(workload = workload)
    bins_builder = grid_preprocess.BinsBuilder(workload, data_manager, mv_manager)

    #
    index_list = (0, 4)
    target_case: case_analysis.CaseAnalyzer = multi_case_analyzer.get_obj_by_index(index_list)
    target_case.case_repr()
    # 
    case_based_estimator.add_new_case(query_meta=target_case.meta, card_dict=target_case.card_dict)

    return multi_case_analyzer, case_based_estimator, data_manager, bins_builder, target_case

# %%

def get_stateful_explorer(workload = "job", method = "DeepDB_jct", template_table_num = None, warm_up_num = 5):
    """
    获得stateful_explorer实例

    Args:
        arg1:
        arg2:
    Returns:
        stateful_explorer:
        return2:
    """
    # if template_table_num is None:
    #     signature = f"{method.lower()}_{workload}_test"
    # else:
    #     signature = f"{method.lower()}_{workload}_test_{template_table_num}"

    if template_table_num is None:
        signature = f"{method}_{workload}_test"
    else:
        signature = f"{method}_{workload}_test_{template_table_num}"

    # signature = f"{method}_{workload}_test"
    common_dir = "/home/lianyuan/Research/CE_Evaluator/intermediate"

    # workload = "job"
    
    expt_config = {
        "selected_tables": workload_spec.total_schema_dict[workload],
        "ce_handler": method
    }
    resource_config = stateful_exploration.default_resource_config
    expl_estimator, card_est_input, max_expl_step = "linear", "graph_corr_based", 100

    init_query_config = { "target": "under","min_card": 500,
        "max_card": 1000000, "mode": "sample-based", "num": 100 }
    tree_config = { "max_depth": 20, "timeout": 60000 }
    init_strategy = "multi-loop"
    
    tmpl_meta_path: str = p_join(common_dir, workload, "template_obj", method, signature, "meta_info.json")

    # res_explorer = stateful_exploration.StatefulExploration(workload, \
    #     expt_config, expl_estimator, resource_config, max_expl_step, tmpl_meta_path,\
    #     init_query_config, tree_config, init_strategy, warm_up_num)

    res_explorer = stateful_exploration.StatefulExploration(workload, expt_config, expl_estimator, 
        resource_config, max_expl_step, tmpl_meta_path, init_query_config, 
        tree_config, init_strategy, warm_up_num, card_est_input, 
        action_selection_mode = "global", root_selection_mode="advance", noise_parameters = None)
    
    res_explorer.set_search_config()
    return res_explorer

# %%
