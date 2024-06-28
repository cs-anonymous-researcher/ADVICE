#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from os.path import join as p_join
from matplotlib import pyplot as plt
import json, os, time, re, math
import pickle, random
import seaborn as sns

# %%


from grid_manipulation import grid_base, grid_construction

from utility import utils
from data_interaction import feedback
from data_interaction.mv_management import condition_encoding

from utility.utils import objects_signature, \
    general_cache_dump, general_cache_load
from utility.utils import restore_origin_label, result_decouple


# %%

def split_segment(x, n):
    """
    {Description}
    
    Args:
        x: 总长度
        n: 分成过稍短
    Returns:
        res1:
        res2:
    """
    # 生成n-1个随机数
    splits = sorted([random.random() for _ in range(n - 1)])
    # 将第一个分割点设置为0，最后一个分割点设置为x
    splits = [0] + splits + [1]
    # 计算每一段的长度
    segments = [x * (splits[i+1] - splits[i]) for i in range(n)]
    return segments

class GridBuilder(object):
    """
    基于各点结果的拓展程序

    Members:
        field1:
        field2:
    """

    def __init__(self, value_cnt_arr, marginal_value_arr, bins_list):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        self.value_cnt_arr = value_cnt_arr
        self.marginal_value_arr = marginal_value_arr
        self.bins_list = bins_list
        self.global_shape = self.value_cnt_arr.shape

    def factor_split(self, amplification_factor, num, subspace = None):
        """
        {Description}
    
        Args:
            amplification_factor:
            num:
            subspace:
        Returns:
            column_amplification_list:
        """
        amplification_factor_log = np.log(amplification_factor)
        segments = split_segment(amplification_factor_log, num)
        # print("amplification_factor_log = {}.".format(amplification_factor_log))
        # print("segments = {}.".format(segments))
        column_amplification_list = [int(math.exp(s)) for s in segments]
        return column_amplification_list
    
    def get_region_idx_pair_list(self, center, column_amplification_list, global_shape, target, mode):
        """
        根据要求生成符合结果的idx_pair_list
        avaiable mode = ["cardinality", "estimation"]
    
        Args:
            center:
            column_amplification_list:
            global_shape:
            target:
            mode:
        Returns:
            idx_pair_list:
            result_value:
        """
        if mode == "cardinality":
            data_array = self.value_cnt_arr
        elif mode == "esttimation":
            data_array = self.marginal_value_arr

        def filter_by_bound(value_list, bound_list):
            """
            {Description}
            
            Args:
                value_list:
                bound_list:
            Returns:
                filtered_list:
            """
            filtered_list = [min(int(v1), int(v2)) for v1, v2 in zip(value_list, bound_list)]
            return filtered_list

        filter_amplification_list = filter_by_bound(column_amplification_list, list(global_shape))
        # print("filter_amplification_list = {}.".format(filter_amplification_list))

        location_order = np.argsort(filter_amplification_list)
        # print("location_order = {}".format(location_order))

        idx_pair_list = [None for _ in location_order]

        def get_range(center_idx, target_len, total_len):
            target_middle = target_len // 2
            start = max(0, center_idx - target_middle)
            end = min(start + target_len, total_len)
            return start, end

        for col_idx in location_order[:-1]:
            local_start, local_end = get_range(center[col_idx], \
                filter_amplification_list[col_idx], global_shape[col_idx])
            idx_pair_list[col_idx] = (local_start, local_end)

        # result_value = 0
        last_dim = location_order[-1]

        def get_aggregation_array(data_array, idx_pair_list):
            selection_pair_list = []
            axis_list = []
            for dim, idx_pair in enumerate(idx_pair_list):
                if idx_pair is None:
                    selection_pair_list.append(Ellipsis)
                else:
                    selection_pair_list.append(slice(*idx_pair))
                    axis_list.append(dim)

            # print("selection_pair_list = {}".format(selection_pair_list))
            # print("axis_list = {}".format(axis_list))
            # numpy中array的下标都需要转成tuple
            sub_array = data_array[tuple(selection_pair_list)]
            # print("sub_array.shape = {}".format(sub_array.shape))
            axis_tuple = tuple(axis_list)
            res_array = np.sum(sub_array, axis=axis_tuple)
            return res_array

        aggregation_array = get_aggregation_array(data_array, idx_pair_list)

        def step_forward(curr_pos, upper_bound):
            if curr_pos >= upper_bound - 1:
                return False, curr_pos
            else:
                return True, curr_pos + 1

        def step_backward(curr_pos, lower_bound):
            if curr_pos <= lower_bound:
                return False, curr_pos
            else:
                return True, curr_pos - 1

        lower_bound, upper_bound = 0, global_shape[last_dim]
        start, end = center[last_dim], center[last_dim]
        curr_sum = aggregation_array[start] # 初始的值
        # print("start = {}. end = {}. curr_sum = {}.".format(start, end, curr_sum))
        while curr_sum < target:              # 若达到要求，直接退出
            # 更新start步
            flag1, start = step_backward(start, lower_bound)
            if flag1 == True:
                curr_sum += aggregation_array[start]
                # print("start = {}. curr_sum = {}. target = {}.".format(start, curr_sum, target))
                if curr_sum >= target:
                    break

            # 更新end步
            flag2, end = step_forward(end, upper_bound)
            if flag2 == True:
                curr_sum += aggregation_array[end]
                # print("end = {}. curr_sum = {}. target = {}.".format(end, curr_sum, target))
                if curr_sum >= target:
                    break
            if flag1 == False and flag2 == False:
                break
        
        idx_pair_list[last_dim] = (start, end + 1)
        result_value = curr_sum
        return idx_pair_list, result_value


    def generate_random_multi_range(self, card_target = None, est_target = None, mode = "absolute"):
        """
        生成随机的多维range
        available mode = ["global", "absolute"]

        Args:
            card_target:
            est_target:
            mode:
        Returns:
            idx_pair_list:
            result_value:
        """
        random_idx = np.random.randint(low = 0, high = len(self.value_cnt_arr))
        random_center = np.unravel_index(random_idx, shape=self.value_cnt_arr.shape)  # 随机生成center，然后再调用之前的函数
        print("random_idx = {}.".format(random_idx))
        print("random_center = {}.".format(random_center))
        return self.generate_multi_range(center = random_center, card_target = card_target,
                    est_target = est_target, mode = mode)

    def generate_subspace_multi_range(self, center = None, card_target = None, \
            est_target = None, mode = "relative", subspace = []):
        """
        生成子空间下的multi_range
        available mode = ["global", "absolute", "relative"]

        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        pass

    def generate_multi_range(self, center, card_target = None, est_target = None, mode = "relative"):
        """
        {Description}
        available mode = ["relative", "global", "absolute"]

        Args:
            center: 中心位置
            card_target: 目标
            est_target:
            mode:
        Returns:
            idx_pair_list:
            result_value:
        """
        # 一维的index
        if isinstance(center, (int, np.int64)): 
            center = np.unravel_index(center, self.global_shape)
        elif isinstance(center, (tuple, list, np.ndarray)):
            center = tuple(center)
        else:
            raise TypeError("Unsupported center type: {}.".format(type(center)))

        idx_pair_list = []
        # metrics_list = []
        curr_card, curr_est = self.value_cnt_arr[center], self.marginal_value_arr[center]
        if curr_card <= 10:  # 对于curr_card特殊情况的处理，最少也得是10
            curr_card = 10

        amplification_factor = 1.0
        object_dict = {}            # 目标存储字典

        valid_cnt = 0
        if mode == "relative":
            # 相对当前格点的模式
            if card_target is not None:
                amplification_factor = card_target
                card_target = card_target * curr_card
                valid_cnt += 1
            if est_target is not None:
                amplification_factor = est_target
                est_target = est_target * curr_est
                valid_cnt += 1
        elif mode == "absolute":
            # 绝对值模式
            if card_target is not None:
                amplification_factor = card_target / curr_card
                valid_cnt += 1
            if est_target is not None:
                amplification_factor = est_target / curr_est
                valid_cnt += 1
        elif mode == "global":
            # 全局模式
            global_grid_num = len(self.value_cnt_arr)
            global_card_num, global_est_num = int(np.sum(self.value_cnt_arr)), int(np.sum(self.marginal_value_arr))

            if card_target is not None:
                assert card_target<1.0, "card_target = {}".format(card_target)
                amplification_factor = (global_grid_num * card_target)
                card_target = global_card_num * card_target
                valid_cnt += 1
            if est_target is not None:
                assert est_target<1.0, "est_target = {}".format(est_target)
                amplification_factor = (global_grid_num * est_target)
                est_target = global_est_num * est_target
                valid_cnt += 1
        else:
            raise ValueError("Unsupported mode: {}".format(mode))


        if card_target is not None:
            object_dict["target"] = "cardinality"
            object_dict["value"] = card_target
        elif est_target is not None:
            object_dict["target"] = "estimation"
            object_dict["value"] = est_target

        if valid_cnt != 1:
            raise ValueError("error: valid_cnt = {}".format(valid_cnt))
        if amplification_factor <= 1.0:
            amplification_factor = 1.0  # 如果小于1.0，就置成1.0
            # raise ValueError("Illegal amplification_factor: {}".format(amplification_factor))
        
        column_amplification_list = self.factor_split(amplification_factor, num = len(self.global_shape))
        # print("column_amplification_list = {}".format(column_amplification_list))

        if card_target is not None:
            region_mode = "cardinality"
            idx_pair_list, result_value = \
                self.get_region_idx_pair_list(center = center, column_amplification_list = column_amplification_list, \
                    global_shape = self.global_shape, target = card_target, mode = region_mode)
        if est_target is not None:
            region_mode = "estimation"
            idx_pair_list, result_value = \
                self.get_region_idx_pair_list(center = center, column_amplification_list = column_amplification_list, \
                    global_shape = self.global_shape, target = est_target, mode = region_mode)

        # print("idx_pair_list = {}.".format(idx_pair_list))
        # print("result_value = {}.".format(result_value))
        object_dict["actual"] = result_value
        print("object_dict = {}".format(object_dict))
        return idx_pair_list, result_value

    def generate_multi_range_batch(self, center, num, card_target = None, est_target = None, mode="relative"):
        """
        {Description}
        available mode = ["relative", "global", "absolute"]

        Args:
            center:
            num:
            card_target:
            est_target:
            mode:
        Returns:
            idx_pair_list_batch:
            metrics_list:
        """
        return [self.generate_multi_range(center=center, card_target=card_target, \
            est_target=est_target, mode=mode) for _ in range(num)]


class GridAnalyzer(object):
    """
    格点结果的分析程序

    Members:
        field1:
        field2:
    """

    def __init__(self, value_cnt_arr, marginal_value_arr, bins_list, query_meta, \
        params_meta, method = "DeepDB", workload = "job", load_history = True):
        """
        {Description}

        Args:
            value_cnt_arr:
            marginal_value_arr:
            bins_list:
            query_meta: 查询原有的meta信息
            params_meta: 变量化的列meta信息
            method:
            workload:
            load_history:
        """
        self.ratio_arr = self.construct_ratio_arr(\
            value_cnt_arr = value_cnt_arr, marginal_value_arr = marginal_value_arr)
        self.value_cnt_arr = value_cnt_arr
        self.marginal_value_arr = marginal_value_arr
        self.bins_list = bins_list
        self.query_builder = self.construct_builder(existing_meta = \
            query_meta, params_meta = params_meta, workload=workload)
        self.result_fetcher = feedback.ResultFetcher(call_type=method, workload=workload)
        self.post_processor = self.construct_processor(\
            value_cnt_arr, marginal_value_arr, bins_list)       # 结果处理器
        
        self.global_val_pair_list_batch = []
        self.global_result_pair_list = []

        self.grid_builder = GridBuilder(value_cnt_arr = value_cnt_arr, \
            marginal_value_arr = marginal_value_arr, bins_list = bins_list)

        self.load_history = load_history
        if load_history == True:
            self.load_historical_result()
        self.function_register()

    def function_register(self, ):
        """
        注册grid_builder中的函数
        
        Args:
            None
        Returns:
            None
        """
        self.generate_multi_range = self.grid_builder.generate_multi_range
        self.generate_multi_range_batch = self.grid_builder.generate_multi_range_batch
        self.generate_random_multi_range = self.grid_builder.generate_random_multi_range
        self.generate_subspace_multi_range = self.grid_builder.generate_subspace_multi_range



    def get_repr_str(self,):
        """
        获得表示字符串，用于保存结果的识别
        
        Args:
            None
        Returns:
            repr_str:
        """
        default_meta = self.query_builder.get_default_meta()
        return condition_encoding(*default_meta)

    def construct_processor(self, value_cnt_arr, marginal_value_arr, bins_list):
        """
        {Description}
    
        Args:
            value_cnt_arr:
            marginal_value_arr:
            bins_list:
        Returns:
            processor:
        """
        return grid_base.PostProcessor(value_cnt_arr, marginal_value_arr, bins_list)

    def construct_builder(self, existing_meta, params_meta, workload):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        query_builder = grid_base.QueryBuilder(existing_meta, workload=workload)
        query_builder.set_multi_conditions(\
            column_info_list = params_meta) # 设置多个条件
        return query_builder

    def grid_summary(self,):
        """
        打印当前grid的信息
        
        Args:
            None
        Returns:
            total_cnt:
            grid_shape:
        """
        total_cnt = np.sum(self.value_cnt_arr)
        grid_shape = self.value_cnt_arr.shape
        # print("total_cnt = {}. grid_shape = {}.".format(total_cnt, grid_shape))
        return total_cnt, grid_shape

    def instance_test(self, idx_pair_list):
        """
        针对单个实例的测试
    
        Args:
            idx_pair_list:
        Returns:
            query_text: 文本
            cardinality: 基数
            val_pair_list: value pair构成的列表
        """
        val_pair_list, label = self.post_processor.\
            select_region_by_index(index_pair_list = idx_pair_list)
        query_text = self.query_builder.generate_query(val_pair_list = val_pair_list)
        return query_text, label, val_pair_list

    def do_evaluation(self, param_list):
        """
        {Description}
        
        Args:
            param_list:
        Returns:
            result:
        """
        return self.do_evaluation_on_grids(param_list)

    def do_evaluation_on_grids(self, param_list):
        """
        根据所选的idx_list，评测基数估计器的性能
        
        Args:
            param_list:
        Returns:
            val_pair_list_batch: 
            label_list: 
            test_query_list: 
            result:
        """
        val_pair_list_batch, label_list = \
            self.post_processor.select_grids_by_1d_index(index_list = param_list)
        test_query_list = self.query_builder.generate_batch_queries(val_pair_list_batch)
        result, pair_list = self.result_fetcher.get_result(query_list = test_query_list, \
            label_list = label_list, with_origin = True)
        
        self.update_result(val_pair_list_batch, pair_list)
        return val_pair_list_batch, label_list, test_query_list, result
    
    def evaluation_on_regions(self, idx_pair_list_batch):
        """
        在一批region上进行评测
        
        Args:
            idx_pair_list_batch:
        Returns:
            val_pair_list_batch: 
            label_list: 
            test_query_list: 
            result:
        """
        val_pair_list_batch, label_list = self.post_processor.\
            select_regions_by_index_batch(index_pair_list_batch = idx_pair_list_batch)
        test_query_list = self.query_builder.generate_batch_queries(val_pair_list_batch = val_pair_list_batch)
        result, pair_list = self.result_fetcher.get_result(query_list = test_query_list, \
            label_list = label_list, with_origin = True)
        
        self.update_result(val_pair_list_batch, pair_list)
        return val_pair_list_batch, label_list, test_query_list, result

    def evaluation_on_one_region(self, idx_pair_list):
        """
        在单个region上进行评测
        
        Args:
            idx_pair_list:
        Returns:
            val_pair_list: 
            label: 
            query_text: 
            q_error:
        """
        val_pair_list, label = self.post_processor.\
            select_region_by_index(index_pair_list = idx_pair_list)
        query_text = self.query_builder.generate_query(val_pair_list = val_pair_list)
        q_error, pair = self.result_fetcher.get_one_result(query_text, label, with_origin=True)
        self.update_one_result(val_pair_list, pair)    # 单例结果的更新
        return val_pair_list, label, query_text, q_error

    def construct_ratio_arr(self, value_cnt_arr, marginal_value_arr):
        """
        {Description}
    
        Args:
            value_cnt_arr:
            marginal_value_arr:
        Returns:
            ratio_arr:
        """
        ratio_arr = value_cnt_arr / (1e-9 + marginal_value_arr)
        return ratio_arr

    def construct_log_bins(self, bin_num, data_vec):
        """
        构造log scale的bins
        
        Args:
            bin_num:
            data_vec:
        Returns:
            bin_list:
        """
        min_val, max_val = np.min(data_vec), np.max(data_vec)
        left_bound, right_bound = np.log(min_val), np.log(max_val)
        idx_list = np.linspace(left_bound, right_bound, bin_num + 1)
        bin_list = np.exp(idx_list)

        return bin_list

    def constuct_uniform_bins(self, bin_num, data_vec):
        """
        构建均匀分布的bins
        
        Args:
            bin_num:
            data_vec:
        Returns:
            bin_list:
        """
        left_bound, right_bound = np.min(data_vec), np.max(data_vec)
        bin_list = np._linspace_dispatcher(left_bound, right_bound, bin_num + 1)
        return bin_list

    def plot_ratio_distribution(self, sample_config = None, bin_num = 40):
        """
        绘制柱状图，表示ratio的分布情况

        Args:
            sample_config:
        Returns:
            fig:
            ax:
        """
        if sample_config is None:
            # 呈现所有结果
            ratio_vec = np.ravel(self.ratio_arr)        # array转成一维vector
            ratio_vec = ratio_vec[ratio_vec != 0]       # 把其中的0给去掉
            fig, ax = plt.subplots()
            bins = self.construct_log_bins(bin_num, data_vec = ratio_vec)   # 构造log-scale的bins
            ax.hist(ratio_vec, log = True, bins = bins)
            ax.set_xscale("log")
            ax.set_xlabel("ratio")
            ax.set_ylabel("count")
            return fig, ax
        else:
            num, method = sample_config['num'], sample_config['method']
            fig, ax = plt.subplots()
            ratio_vec = self.weight_sample(num)
            ax.hist(ratio_vec, log = True, bins = 40)
            return fig, ax

    def plot_eval_scatter(self, sample_config = None):
        """
        绘制估计结果的散点图，横轴是ratio_arr的结果，纵轴是q_error

        Args:
            sample_config:
        Returns:
            res_fig:
        """
        pass
    
    def sample_in_bin(self, bin_elems, num):
        """
        在bin中进行随机采样
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        if num >= len(bin_elems):   # 采样数目大于全局数目，直接退出
            return bin_elems
        else:
            return np.random.choice(bin_elems, size = num)
    
    def weight_sample(self, num, bins_num = 40, mode = "log"):
        """
        根据结果的分布进行采样，策略是先划分出很多个bins，
        然后在bins中进行均匀采样
    
        Args:
            num:
            bins_num:
            mode:
        Returns:
            result_list:
        """
        # index和value会同时产生变化
        ratio_vec = np.ravel(self.ratio_arr)
        ratio_idx = np.arange(ratio_vec.size)

        nonzero_idx = ratio_vec >= 1e-9
        ratio_vec = ratio_vec[nonzero_idx]
        ratio_idx = ratio_idx[nonzero_idx]

        bins = self.constuct_uniform_bins(bin_num = bins_num, data_vec = ratio_vec)
        digital_res = np.digitize(ratio_vec, bins)
        
        bin_res = [[] for _ in range(bins_num)]
        for bin_idx, origin_idx in zip():
            bin_res[bin_idx].append(origin_idx)

        bin_sample_num = np.ceil(num / bins_num)

        res_idx_list = []
        for i in bin_res:
            res_idx_list.extend(self.sample_in_bin(i, bin_sample_num))

        return [np.unravel(i, self.ratio_arr.shape) for i in res_idx_list]
    
    def uniform_sample(self, num, minimum_ratio = 1e-3):
        """
        对于所有样本均匀采样
        
        Args:
            num:
        Returns:
            result_list:
        """
        ratio_arr = self.ratio_arr
        total_idx_list = np.ravel_multi_index(np.where(ratio_arr > 1e-3), ratio_arr.shape)    # 选择特定范围的数据

        if len(total_idx_list) > num:
            sample_res = np.random.choice(total_idx_list, size = num, replace = False)
        else:
            sample_res = total_idx_list

        return sample_res
    
    def update_one_result(self, val_pair_list, result_pair):
        """
        更新单个的结果
    
        Args:
            val_pair_list:
            result_pair:
        Returns:
            global_val_pair_list_batch:
            global_result_pair_list:
        """
        self.global_val_pair_list_batch.append(val_pair_list)
        self.global_result_pair_list.append(result_pair)
        if self.load_history == True:
            self.flush_result()
        return self.global_val_pair_list_batch, self.global_result_pair_list


    def update_result(self, val_pair_list_batch, result_pair_list):
        """
        添加单次实验的结果
    
        Args:
            val_pair_list_batch:
            result_pair_list:
        Returns:
            global_val_pair_list_batch:
            global_result_pair_list:
        """
        self.global_val_pair_list_batch.extend(val_pair_list_batch)
        self.global_result_pair_list.extend(result_pair_list)
        if self.load_history == True:
            self.flush_result()
        return self.global_val_pair_list_batch, self.global_result_pair_list

    def load_historical_result(self, ):
        """
        加载历史中的结果
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        data_path = "../tmp/grid_analysis/{}.pkl".format(self.get_repr_str())
        res = utils.load_pickle(data_path)
        if res is not None:
            self.global_val_pair_list_batch, self.global_result_pair_list = res
        else:
            self.global_val_pair_list_batch, self.global_result_pair_list = [], []
        return self.global_val_pair_list_batch, self.global_result_pair_list

    def flush_result(self, ):
        """
        支持结果的持久化，提升实验的效率
        
        Args:
            None
        Returns:
            res1:
            res2:
        """
        data_path = "../tmp/grid_analysis/{}.pkl".format(self.get_repr_str())
        utils.dump_pickle(res_obj = (self.global_val_pair_list_batch, \
            self.global_result_pair_list), data_path = data_path)
        return self.global_val_pair_list_batch, self.global_result_pair_list


    def select_target_test_instance(self, num = 1, min_true_card = 100):
        """
        选择满足目标的测试实例
        
        Args:
            num:
            min_true_card:
        Returns:
            result_list:
        """
        result_list = []
        for _ in range(num):
            idx_pair_list, result_value = self.generate_random_multi_range(card_target = min_true_card)
            query_text, cardinality, val_pair_list = self.instance_test(idx_pair_list)
            result_list.append((query_text, cardinality))

        return result_list

    def select_target_result_instance(self, num = 1, mode = "default", min_true_card = 100, \
            obj_func = None, valid_func = None):
        """
        从结果集中选择满足目标的结果实例
        available modes = ["default", "over_estimation", "under_estimation", "q_error"]
        
        Args:
            num:
            mode:
            min_true_card:
            obj_func:
            valid_func:
        Returns:
            res_num: default 1
            query_meta:
            query_result:
        """
        valid_list = []

        if obj_func is None:
            if mode == "default" or mode == "q_error":
                obj_func = q_error_top_k
            elif mode == "over_estimation":
                obj_func = over_estimation_top_k
            elif mode == "under_estimation":
                obj_func = under_estimation_top_k

        if valid_func is None:
            valid_func = filter_by_card(in_card = min_true_card)

        def list_idx_selection(in_list, idx_list):
            return [in_list[idx] for idx in idx_list]

        local_val_pair_list_batch, local_result_pair_list = [], []

        for val_pair_list, result_pair in zip(self.global_val_pair_list_batch, self.global_result_pair_list):
            # print("val_pair_list = {}".format(val_pair_list))
            # print("result_pair = {}".format(result_pair))
            if valid_func(*result_pair) == True:    # 有效性判断
                local_val_pair_list_batch.append(val_pair_list)
                local_result_pair_list.append(result_pair)

        # print("local_val_pair_list_batch = {}.".format(local_val_pair_list_batch))
        # print("local_result_pair_list = {}.".format(local_result_pair_list))

        index_list = obj_func(local_result_pair_list, num)  # 最终结果的筛选
        out_val_pair_list_batch = list_idx_selection(local_val_pair_list_batch, index_list)
        out_result_pair_list = list_idx_selection(local_result_pair_list, index_list)

        if num == 1:
            res_num = 1
            query_meta = self.build_query_meta(out_val_pair_list_batch[0])
            return res_num, query_meta, out_result_pair_list[0]
        else:
            assert len(out_val_pair_list_batch) == len(out_result_pair_list)
            res_num = len(out_val_pair_list_batch)
            query_meta_batch = self.build_query_meta_batch(out_val_pair_list_batch)
            return res_num, query_meta_batch, out_result_pair_list


    def build_query_meta_batch(self, val_pair_list_batch):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        return [self.build_query_meta(val_pair_list) for \
            val_pair_list in val_pair_list_batch]

    def build_query_meta(self, val_pair_list):
        """
        组装查询的元信息
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        # return self.post_processor.build_query_meta(val_pair_list)
        return self.query_builder.generate_meta(val_pair_list = val_pair_list)

def q_error_top_k(result_pair_list, num):
    """
    {Description}
    
    Args:
        result_pair_list:
    Returns:
        index_list:
    """
    # print("result_pair_list = {}.".format(result_pair_list))
    # print("num = {}.".format(num))

    q_error_list = [max(i / j, j / i) for i, j in result_pair_list]
    return np.argsort(q_error_list)[::-1][:num]


def over_estimation_top_k(result_pair_list, num):
    """
    {Description}
    
    Args:
        result_pair_list: (真实基数，估计基数)组成的list
    Returns:
        index_list:
    """
    over_estimation_list = [j / i for i, j in result_pair_list]
    return np.argsort(over_estimation_list)[::-1][:num]


def under_estimation_top_k(result_pair_list, num):
    """
    {Description}
    
    Args:
        result_pair_list: (真实基数，估计基数)组成的list
    Returns:
        index_list:
    """
    under_estimation_list = [i / j for i, j in result_pair_list]
    return np.argsort(under_estimation_list)[::-1][:num]


def filter_by_card(in_card):
    def res_func(true_card, estimation_card):
        if true_card >= in_card:
            return True
        else:
            return False
    return res_func

