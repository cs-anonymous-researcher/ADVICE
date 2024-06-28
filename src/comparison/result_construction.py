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
from utility import utils, workload_parser
from comparison import result_base, external_memory
from sklearn.linear_model import LinearRegression
from matplotlib.ticker import MultipleLocator
from collections import defaultdict

# %%
from comparison.aggregation_func import *
# %%

"""
根据config文件来生成测试结果集

{
    "out_dir": "timestamp",
    "result_list": [result_id1, result_id2],
    "metrics_list": [
        {
            "name": "",
            "params_dict": {}
        },
        {
            "name": "",
            "params_dict": {}
        }
    ],
    "mode": "stable/growth" 
}
"""
# %%

class ReportConstructor(result_base.ResultBase):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, workload, intermediate_dir, result_dir, \
            config_dir = "/home/lianyuan/Research/CE_Evaluator/evaluation/config"):
        """
        {Description}

        Args:
            workload:
            intermediate_dir:
            result_dir: 结果保存路径
        """
        super(ReportConstructor, self).__init__(workload, intermediate_dir, result_dir, config_dir)

        self.workload = workload
        self.intermediate_dir = intermediate_dir
        self.result_dir = result_dir
        self.result_meta_dict, self.metrics_meta_dict = self.load_meta()

        self.result_metrics_timed = {}

    def load_meta(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        # 加载结果的元信息
        meta_path = p_join(self.intermediate_dir, self.workload, "experiment_obj", "result_meta.json")
        meta_dict = utils.load_json(data_path=meta_path)

        # 加载metrics_list
        metrics_meta_dict = utils.load_json(self.metrics_path)
        return meta_dict, metrics_meta_dict


    def parse_config(self, config: dict, verify_complete = False, \
            verify_estimation = False, out_format = ("query", "meta", "result", "card_dict")):
        """
        解析配置，获得相应的查询结果
    
        Args:
            config:
            arg2:
        Returns:
            out_dir:
            result_val_dict:
        """
        result_val_dict = {}

        out_dir = config["out_dir"]
        if out_dir == "timestamp":
            out_dir = time.strftime("%Y%m%d%H%M%S", time.localtime())


        name_idx_dict = {
            "query": 0, "meta": 1,
            "result": 2, "card_dict": 3
        }

        def load_result_by_id(result_id):
            try:
                instance_meta = self.result_meta_dict[str(result_id)]
            except KeyError as e:
                print(f"parse_config.load_result_by_id: meet KeyError. result_meta_dict = {self.result_meta_dict.keys()}.")
                raise e
            
            obj_path = instance_meta['obj_path']
            result_obj = utils.load_pickle(data_path=obj_path)

            if result_obj is None:
                # 尝试新的路径
                obj_dir = os.path.dirname(obj_path)
                obj_name = os.path.basename(obj_path)
                new_path = p_join(obj_dir, "history_pickle", obj_name)
                result_obj = utils.load_pickle(data_path=new_path)
            assert result_obj is not None, f"construct_instance: new_path = {new_path}"

            # self.filter_invalid_cases(zip(result_obj)
            method_name = instance_meta['search_method']

            if len(result_obj) == 5:
                result_obj = result_obj[:4]

            # result_obj = self.filter_wrap_func(result_obj, card_complete = verify_complete, 
            #     card_estimation = verify_estimation, ce_type = method_name)
            result_obj = self.filter_wrap_func(result_obj, card_top = True, 
                card_complete = True, card_estimation = False, ce_type = method_name)
            
            # result_val = result_obj[2]
            pack_list = []

            for name in out_format:
                pack_list.append(result_obj[name_idx_dict[name]])

            if len(pack_list) == 1:
                return method_name, pack_list[0]
            else:
                return method_name, tuple(pack_list)
            

        if "result_list" in config:
            result_list = config["result_list"]
            for result_id in result_list:
                method_name, res_out = load_result_by_id(result_id)
                result_val_dict[method_name] = res_out
                    
                # if len(pack_list) == 1:
                #     result_val_dict[method_name] = pack_list[0]
                # else:
                #     result_val_dict[method_name] = tuple(pack_list)
        elif "result_dict" in config:
            for ce_method, result_list in config["result_dict"].items():
                for result_id in result_list:
                    method_name, res_out = load_result_by_id(result_id)
                    result_val_dict[(ce_method, method_name)] = res_out
        else:
            raise ValueError(f"ReportConstructor.parse_config: config = {config}.")

        return out_dir, result_val_dict

    def evaluate_instance(self, config: dict):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        result_list = config["result_list"]
        result_all = []
        for result_id in result_list:
            instance_meta = self.result_meta_dict[result_id]
            obj_path = instance_meta['obj_path']

            result_obj = self.load_object(obj_path)
            if len(result_obj) == 5:
                # 有time信息的情况
                result_obj = result_obj[:4]

            instance_list = list(zip(*result_obj))
            ce_type = instance_meta['estimation_method']
            instance_out = self.filter_invalid_cases(instance_list, \
                card_complete = True, card_estimation = False, ce_type = ce_type)
            result_local = self.filter_cost_error_cases(instance_out)
            result_all.extend(result_local)
 
        return result_all
    
    def construct_instance(self, config: dict, verify_complete = False, verify_estimation = False, 
            verify_top = False, dict_key_mode = "search_method"):
        """
        构造整体结果实例
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        assert dict_key_mode in ("search_method", "result_id")
        result_val_dict = {}

        out_dir = config["out_dir"]
        if out_dir == "timestamp":
            out_dir = time.strftime("%Y%m%d%H%M%S", time.localtime())

        result_list = config["result_list"]

        for result_id in result_list:
            instance_meta = self.result_meta_dict[str(result_id)]
            obj_path = instance_meta['obj_path']
            result_obj = utils.load_pickle(data_path=obj_path)

            if result_obj is None:
                # 尝试新的路径
                obj_dir = os.path.dirname(obj_path)
                obj_name = os.path.basename(obj_path)
                new_path = p_join(obj_dir, "history_pickle", obj_name)
                result_obj = utils.load_pickle(data_path=new_path)
                assert result_obj is not None, f"construct_instance: new_path = {new_path}"

            # print(f"construct_instance: result_id = {result_id}. len(result_obj) = {len(result_obj)}.")
            #
            if len(result_obj) == 5:
                # 有time信息的情况，直接删除
                result_obj = result_obj[:4]
                
            method_name = instance_meta['search_method']
            estimator_name = instance_meta['estimation_method']

            result_obj = self.filter_wrap_func(result_obj, verify_complete, \
                verify_estimation, verify_top, estimator_name)

            result_val = result_obj[2]
            print(f"construct_instance: method_name = {method_name}. estimator_name = {estimator_name}. "\
                  f"len(result_obj) = {len(result_obj)}. len(result_val) = {len(result_val)}.")
            
            # 2024-03-26: 添加结果
            self.append_to_result_dict(estimator_name, result_val)
            
            if dict_key_mode == "search_method":
                result_val_dict[method_name] = result_val
            elif dict_key_mode == "result_id":
                result_val_dict[result_id] = result_val


        def subinfo_dict(val_dict, mode):
            out_dict = {}
            if mode == "p_error":
                for k, v in val_dict.items():
                    out_dict[k] = [item[0] for item in v]

            elif mode == "value_pair":
                for k, v in val_dict.items():
                    out_dict[k] = [(item[1], item[2]) for item in v]

            return out_dict
        
        metrics_list = config["metrics_list"]
        if isinstance(metrics_list, str):
            # str类型，做一层额外的转换
            metrics_list = self.metrics_meta_dict[metrics_list]
        metrics_list = self.complement_missing_info(metrics_list)   # 

        func_mapping, subinfo_mode_mapping = self.get_metrics_information()

        result_metrics_dict = {}
        for metrics in metrics_list:
            n = metrics['type']
            test_func = func_mapping[n]
            params_dict = metrics['params_dict']
            params_dict['val_dict'] = subinfo_dict(val_dict=\
                result_val_dict, mode=subinfo_mode_mapping[n])

            # result_metrics_dict[metrics['name']] = test_func(**params_dict)
            result_metrics_dict[metrics['name']] = {
                "data": test_func(**params_dict),          # 具体数据
                "position": metrics['position']            # 图的位置
            }
        self.construct_format_output(result_metrics_dict=result_metrics_dict, mode="text")
        self.result_metrics_dict = result_metrics_dict
        return result_metrics_dict
    
    def complement_missing_info(self, metrics_list: list):
        """
        补充缺失的metrics信息，主要为了做代码兼容
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        metrics_out = []
        total_metrics_num = len(metrics_list)
        row_num, col_num = math.ceil(total_metrics_num / 2), 2
        def get_position(idx):
            col_idx = idx % col_num
            row_idx = math.floor(idx / col_num)
            return row_idx, col_idx
        
        for idx, item in enumerate(metrics_list):
            print(f"complement_missing_info: idx = {idx}. item = {item}.")
            flag1, flag2, flag3 = "name" in item, "type" in item, "position" in item
            if (not flag1) and (not flag2):
                # 如果name和type缺失了一个，就用另一个补全
                raise ValueError(f"complement_missing_info: name and type all miss. idx = {idx}. item = {item}")

            if flag1 and (not flag2):
                item['type'] = item['name']

            if (not flag1) and flag2:
                item['name'] = item['type']

            if not flag3:
                item['position'] = get_position(idx)

            metrics_out.append(item)

        return metrics_out


    def construct_format_output(self, result_metrics_dict, metrics_order = [],
        method_order = [], mode = "text"):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        if len(metrics_order) == 0:
            metrics_order = list(result_metrics_dict.keys())

        if len(method_order) == 0:
            method_order = list(result_metrics_dict[metrics_order[0]]['data'].keys())

        print("metrics_order = {}. method_order = {}.".format(metrics_order, method_order))
        line_list = []

        if mode == "text":
            # 打印文本信息
            for metrics in metrics_order:
                line = "metrics: {}.".format(metrics)
                line_list.append(line)
                print(line)
                try:
                    for method in method_order:
                        line = "    {}: {:.3f}.".format(method, result_metrics_dict[metrics]['data'][method])
                        line_list.append(line)
                        print(line)
                except Exception as e:
                    print(f"func_name: meet Error. {result_metrics_dict[metrics]['data'][method]}.")
                    raise e

                            
        elif mode == "latex":
            # 生成latex表格
            header_template = ""
            line_template = ""
        elif mode == "markdown":
            # 生成markdown的表格
            pass

        return metrics_order, method_order, line_list
    

    def display_result_list(self, config: dict, verify_complete = False, \
        verify_estimation = False, estimation_subset = None, search_subset = None):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        result_val_dict = {}

        out_dir = config["out_dir"]
        if out_dir == "timestamp":
            out_dir = time.strftime("%Y%m%d%H%M%S", time.localtime())

        result_list = config["result_list"]

        for result_id in result_list:
            instance_meta = self.result_meta_dict[result_id]
            obj_path = instance_meta['obj_path']
            # result_obj = utils.load_pickle(data_path=obj_path)
            result_obj = self.load_object(obj_path)
            print(f"construct_instance: result_id = {result_id}. len(result_obj) = {len(result_obj)}.")
            #
            if len(result_obj) == 5:
                # 有time信息的情况，直接删除
                result_obj = result_obj[:4]
                
            method_name = instance_meta['search_method']
            estimator_name = instance_meta['estimation_method']

            if estimation_subset is not None and estimator_name not in estimation_subset:
                continue

            if search_subset is not None and method_name not in search_subset:
                continue

            result_obj = self.filter_wrap_func(result_obj, \
                verify_complete, verify_estimation, estimator_name)

            result_val = result_obj[2]
            # print(f"construct_instance: len(result_obj) = {len(result_obj)}. len(result_val) = {len(result_val)}.")
            result_val_dict[(method_name, estimator_name)] = result_val

        def subinfo_dict(val_dict, mode):
            out_dict = {}
            if mode == "p_error":
                for k, v in val_dict.items():
                    out_dict[k] = [item[0] for item in v]

            elif mode == "value_pair":
                for k, v in val_dict.items():
                    out_dict[k] = [(item[1], item[2]) for item in v]

            return out_dict
        
        metrics_list = config["metrics_list"]
        
        if isinstance(metrics_list, str):
            # str类型，做一层额外的转换
            metrics_list = self.metrics_meta_dict[metrics_list]

        func_mapping, subinfo_mode_mapping = self.get_metrics_information()

        result_metrics_dict = {}
        for metrics in metrics_list:
            n = metrics['type']
            test_func = func_mapping[n]
            params_dict = metrics['params_dict']
            params_dict['val_dict'] = subinfo_dict(val_dict=\
                result_val_dict, mode=subinfo_mode_mapping[n])

            # result_metrics_dict[metrics['name']] = test_func(**params_dict)
            result_metrics_dict[metrics['name']] = {
                "data": test_func(**params_dict),          # 具体数据
                "position": metrics['position']            # 图的位置
            }

        self.construct_format_output(result_metrics_dict=result_metrics_dict, mode="text")
        self.result_metrics_dict = result_metrics_dict
        return result_metrics_dict
    
    def append_to_result_dict(self, ce_method, result_list):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        error_list = [item[0] for item in result_list]
        external_memory.result_error_dict[(self.workload, ce_method)].extend(error_list)

    def get_metrics_information(self,):
        """
        获得metrics的相关信息
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        func_mapping = {
            "max_p_error": max_p_error,
            "quantile_p_error": quantile_p_error,
            "90th_p_error": ninetieth_p_error, 
            "topk_p_error": topk_p_error,
            "median_p_error": median_p_error,
            "cumulative_workload_p_error": cumulative_workload_p_error,
            "number_of_notable_instances": number_of_notable_instances
        }

        subinfo_mode_mapping = {
            "max_p_error": "p_error",
            "quantile_p_error": "p_error",
            "90th_p_error": "p_error",
            "topk_p_error": "p_error",
            "median_p_error": "p_error",
            "cumulative_workload_p_error": "value_pair",
            "number_of_notable_instances": "p_error"
        }
        return func_mapping, subinfo_mode_mapping

    def get_plot_information(self, row_num = 2, col_num = 2):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            fig:
            axes:
            plot_mapping:
            name_abbr:
        """
        # fig, axes = plt.subplots(2, 2)
        fig, axes = plt.subplots(row_num, col_num)

        # 测试方法的顺序
        method_order = []

        # 感觉这玩意得自己来调度
        plot_mapping = {    
            "max_p_error": (0, 0),
            "median_p_error": (0, 1),
            "topk_p_error": (1, 0),
            "cumulative_workload_p_error": (1, 1)
        }

        name_abbr = {
            # 基础方法
            "random": "random",
            "init_heuristic": "init_heur.",
            "final_heuristic": "final_heur.",
            "feedback_based": "feedback",
            
            # 并行方法
            "random_parallel": "random",
            "init_heuristic_parallel": "init_heur.",
            "final_heuristic_parallel": "final_heur.",
            "feedback_based_parallel": "feedback",

            # 学习型的方法
            "learnedsql": "learnedsql", 
            #
            "stateful_parallel": "stateful(ours)"
        }

        # return fig, axes, plot_mapping, name_abbr
        return fig, axes, name_abbr


    def construct_metrics_string(self, metrics_order = ["max_p_error", \
            "median_p_error", "topk_p_error", "cumulative_workload_p_error"]):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        print(f"construct_metrics_string: result_metrics_dict = {self.result_metrics_dict}")

        metrics_list_dict, metrics_str_dict = defaultdict(list), {}
        # for metrics_name, val_dict in self.result_metrics_dict.items():

        for metrics in metrics_order:
            val_dict: dict = self.result_metrics_dict[metrics]['data']
            for method, value in val_dict.items():
                metrics_list_dict[method].append(value)

        for k, v_list in metrics_list_dict.items():
            metrics_str_dict[k] = " / ".join([f"{v:.2f}" for v in v_list])

        return metrics_str_dict

    def construct_bar_chart(self, result_metrics_dict = None):
        """
        构建柱状图
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        if result_metrics_dict is None:
            result_metrics_dict = self.result_metrics_dict

        fig, axes, name_abbr = self.get_plot_information()

        # for metrics_name, val_dict in result_metrics_dict.items():
        for metrics_name, item_dict in result_metrics_dict.items():
            val_dict = item_dict['data']
            position_idx = item_dict['position']
            # selected_idx = plot_mapping[metrics_name]
            # ax = axes[selected_idx[0]][selected_idx[1]]
            ax = axes[position_idx[0]][position_idx[1]]

            name_list, val_list = utils.dict2list(val_dict)
            name_list = [name_abbr[item] for item in name_list]

            data_range = range(len(val_list))
            hatch_styles = ['/', '\\', '|', '-', '+', 'x']

            bars = ax.bar(data_range, val_list, ec='k', color='white', linewidth=2)
            for bar, hatch in zip(bars, hatch_styles):
                bar.set_hatch(hatch)
                
            ax.tick_params(labelrotation=20)
            ax.set_xticks(data_range)
            ax.set_xticklabels(name_list)
            ax.set_ylabel('error_value')
            ax.set_title(metrics_name)

        plt.tight_layout()
        return fig, axes

    def construct_timed_instance(self, config, verify_complete, verify_estimation, verify_top, granularity = 300):
        """
        构造附带时间的实例
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        result_val_dict = {}

        out_dir = config["out_dir"]
        if out_dir == "timestamp":
            out_dir = time.strftime("%Y%m%d%H%M%S", time.localtime())

        result_list = config["result_list"]
        calc_func = self.construct_aggregate_function(granularity)

        for idx, result_id in enumerate(result_list):
            instance_meta = self.result_meta_dict[result_id]
            obj_path = instance_meta['obj_path']
            result_obj = self.load_object(obj_path)

            print(f"construct_timed_instance: result_id = {result_id}. len(result_obj) = {len(result_obj)}.")
            #
            assert len(result_obj) == 5
            result_obj, delta_time = result_base.process_time_info(result_obj)

            method_name = instance_meta['search_method']
            estimator_name = instance_meta['estimation_method']

            result_obj = self.filter_wrap_func(result_obj, \
                verify_complete, verify_estimation, verify_top, method_name)

            self.append_to_result_dict(estimator_name, result_obj[2])
            instance_list = list(zip(*result_obj))
            test_item = instance_list[0]
            print(f"construct_timed_instance: test_item[0] = {type(test_item[0])}. test_item[1] = {type(test_item[1])}. "\
                  f"test_item[2] = {type(test_item[2])}. test_item[3] = {type(test_item[3])}. test_item[4] = {type(test_item[4])}.")
            # result_val = result_obj[2]

            if idx == 0:
                batch_list, time_boundary, index_info = calc_func(instance_list, time_end=delta_time)
            else:
                batch_list, _, _ = calc_func(instance_list, time_boundary, index_info)

            print(f"construct_timed_instance: len(result_obj) = {len(result_obj)}. len(result_val) = {len(batch_list)}.")
            result_val_dict[method_name] = batch_list

        external_memory.set_current_list(self.workload, estimator_name)
        # 提取val_dict的子信息
        def subinfo_dict(val_dict, mode):
            out_dict = {}
            if mode == "p_error":
                for k, v in val_dict.items():
                    out_dict[k] = [[item[0] for item in local_list] for local_list in v]

            elif mode == "value_pair":
                for k, v in val_dict.items():
                    out_dict[k] = [[(item[1], item[2]) for item in local_list] for local_list in v]

            return out_dict
        
        metrics_list = config["metrics_list"]
        if isinstance(metrics_list, str):
            # str类型，做一层额外的转换
            metrics_list = self.metrics_meta_dict[metrics_list]

        func_mapping, subinfo_mode_mapping = self.get_metrics_information()

        result_metrics_timed = {}
        for metrics in metrics_list:
            n = metrics['type']
            test_func = func_mapping[n]
            params_dict = metrics['params_dict']
            params_dict['val_dict'] = subinfo_dict(val_dict=\
                result_val_dict, mode=subinfo_mode_mapping[n])

            # result_metrics_timed[metrics['name']] = test_func(**params_dict)
            result_metrics_timed[metrics['name']] = {
                "data": test_func(**params_dict),          # 具体数据
                "position": metrics['position']            # 图的位置
            }

        self.result_metrics_timed = result_metrics_timed
        return result_metrics_timed

    def construct_aggregate_function(self, granularity = 180):
        """
        构造聚集函数
    
        Args:
            granularity:
            arg2:
        Returns:
            cal_func:
            return2:
        """
        # granularity *= 1e6
        def split_func(time_end):
            # print(f"construct_aggregate_function.split_func: time_list = {time_list}.")
            # time_start, time_end = np.min(time_list), np.max(time_list)
            time_start = 0.0
            print(f"split_func: time_end = {time_end}. granularity = {granularity}.")
            time_boundary = np.arange(time_start, time_end, granularity)
            time_boundary[-1] = time_end + 1.0
            index_info = {}
            for idx, (time_start, time_end) in enumerate(zip(time_boundary[:-1], time_boundary[1:])):
                index_info[idx] = time_start, time_end

            return time_boundary, index_info
        
        def cal_func(instance_list, time_boundary = None, index_info = None, time_end = None):
            time_list = [item[4] for item in instance_list]

            if time_boundary is None:
                time_boundary, index_info = split_func(time_end)

            index_list = []
            for t in time_list:
                pos = np.searchsorted(time_boundary, t, side="right")
                index_list.append(pos - 1)

            batch_list = [[] for _ in range(len(time_boundary))]

            # 遇到超出范围的，直接过滤
            for idx, instance in zip(index_list, instance_list):
                # print(f"cal_func: idx = {idx}. len(time_boundary) = {len(time_boundary)}. instance[2] = {instance[2]}")
                if idx >= len(time_boundary):
                    continue

                batch_list[idx].append(instance[2])
            return batch_list, time_boundary, index_info

        return cal_func

    def completion_by_difference(self, in_list):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        original_array = np.array(in_list)
        valid_indices = np.arange(len(original_array))[~np.less(original_array, 1e-5)]
        interpolated_values = np.interp(np.arange(len(original_array)), \
            valid_indices, original_array[valid_indices])
        return list(interpolated_values)

    def construct_line_chart(self, result_metrics_timed = None):
        """
        构建折线图，展现生成查询质量随时间的变化
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        if result_metrics_timed is None:
            result_metrics_timed = self.result_metrics_timed

        # fig, axes, plot_mapping, name_abbr = self.get_plot_information()
        fig, axes, name_abbr = self.get_plot_information()

        for idx, (metrics_name, item_dict) in enumerate(result_metrics_timed.items()):
            val_dict = item_dict['data']
            position_idx = item_dict['position']
            # selected_idx = plot_mapping[metrics_name]
            ax = axes[position_idx[0]][position_idx[1]]
            # print(f"metrics_name = {metrics_name}. val_dict = {utils.list_round(val_dict['feedback_based_parallel'], 3)}.")
            marker_styles = ['o', 'v', '^', 'p', 'h']

            for style, (method, val_list) in zip(marker_styles, val_dict.items()):
                val_list = self.completion_by_difference(val_list)
                if idx == 0:
                    ax.plot(range(len(val_list)), val_list, marker=style, label=name_abbr[method])
                else:
                    ax.plot(range(len(val_list)), val_list, marker=style)
                ax.set_title(metrics_name)
                ax.xaxis.set_major_locator(MultipleLocator(2))

                ax.set_xlabel("time_epoch")
                ax.set_ylabel('error_value')

        fig.legend(ncol=4, bbox_to_anchor=(0.1, 1),
              loc='lower left')

        plt.tight_layout()
        return fig, axes

    def construct_slope_result(self, result_metrics_timed = None):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        result_dict = {}
        if result_metrics_timed is None:
            result_metrics_timed = self.result_metrics_timed

        _, _, _, name_abbr = self.get_plot_information()
        for idx, (metrics_name, val_dict) in enumerate(result_metrics_timed.items()): 
            local_dict = {}
            for method, val_list in val_dict.items():
                val_list = self.completion_by_difference(val_list)
                local_dict[name_abbr[method]] = self.get_timed_slope(val_list)

            result_dict[metrics_name] = local_dict
        return result_dict
    
    def get_timed_slope(self, val_list):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        data_X, data_y = np.arange(len(val_list)), np.array(val_list)
        model = LinearRegression()
        model.fit(data_X, data_y)
        return model.coef_
    
# %%
class MultiReportConstructor(object):
    """
    针对不同的基数估计器

    Members:
        field1:
        field2:
    """

    def __init__(self, workload, intermediate_dir, result_dir, \
            config_dir = "/home/lianyuan/Research/CE_Evaluator/evaluation/config"):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        self.instance_params = {
            "workload": workload,
            "intermediate_dir": intermediate_dir,
            "result_dir": result_dir,
            "config_dir": config_dir
        }
        self.instance_config_dict = {}
        self.constructor_dict = {}

    def load_config(self, config_in: dict):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        for k, v in config_in['result_dict'].items():
            config_local = deepcopy(config_in)
            del config_local['result_dict']
            config_local['result_list'] = [str(i) for i in v]
            self.instance_config_dict[k] = config_local

        return self.instance_config_dict

    def construct_instance_dict(self, verify_complete: bool = True,
            verify_estimation: bool = False, verify_top: bool = True):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        for k, config_dict in self.instance_config_dict.items():
            local_constructor = ReportConstructor(**self.instance_params)
            local_constructor.construct_instance(config_dict, verify_complete, verify_estimation, verify_top)
            self.constructor_dict[k] = local_constructor

        return self.constructor_dict

    def construct_timed_instance_dict(self, verify_complete: bool = True,
            verify_estimation: bool = False, verify_top: bool = True):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        for k, config_dict in self.instance_config_dict.items():
            local_constructor = ReportConstructor(**self.instance_params)
            # local_constructor.construct_instance(config_dict, verify_complete, verify_estimation, verify_top)
            local_constructor.construct_timed_instance(config_dict, 
                verify_complete, verify_estimation, verify_top)
            self.constructor_dict[k] = local_constructor

        return self.constructor_dict

    def verify_all_p_errors(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        result_dict = {}
        for k, config_dict in self.instance_config_dict.items():
            local_constructor = ReportConstructor(**self.instance_params)
            result_local = local_constructor.evaluate_instance(config_dict)
            # local_constructor.filter_cost_error_cases()
            result_dict[k] = result_local
        return result_dict
    

    def construct_bar_chart_dict(self,):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        for k, v in self.constructor_dict.items():
            ce_method = k
            local_constructor: ReportConstructor = v
            fig, axes = local_constructor.construct_bar_chart()
            fig.savefig(f"{ce_method}_cmp_bar.png", dpi = 1000)


    def construct_line_chart_dict(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        for k, v in self.constructor_dict.items():
            ce_method = k
            local_constructor: ReportConstructor = v
            fig, axes = local_constructor.construct_bar_chart()
            fig.savefig(f"{ce_method}_cmp_line.png", dpi = 1000)    


    def construct_latex_table(self,):
        """
        构造latex表格的对应数据
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        # table_template = """
        #     \setlength\tabcolsep{{1.5pt}} %调列距
        #     \begin{{table*}}[t]
        #         \caption{{A double column table.}}
        #         {tabular_content}
        #         \label{{tab:commands}}
        #     \end{{table*}}
        #     """
        # tabular_template = """
        #     \begin{{tabular}}{align_mode}
        #     {ce_method_line}
        #     {metrics_line}
        #     {result_lines}
        #     \end{{tabular}}
        #     """
        # header_template = "CE Methods & {method_list}"
        # metrics_template = "Metrics & {metrics_list}"
        # result_template = "{method_name} & {result_list}"


    def construct_metrics_str_dict(self,):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        df_dict = {}
        for k, v in self.constructor_dict.items():
            ce_method = k
            local_constructor: ReportConstructor = v
            str_dict = local_constructor.construct_metrics_string()
            df_dict[ce_method] = str_dict

        return df_dict
    

    def cosntruct_latex_table_dict(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        table_repr_dict = {}
        for k, v in self.constructor_dict.items():
            ce_method = k
            local_constructor: ReportConstructor = v
            table_str = local_constructor.construct_latex_table()
            table_repr_dict[ce_method] = table_str

        return table_repr_dict
    

# %%
