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
import types
# %%
import sys
import torch
import botorch
import gpytorch
from utility import utils
from botorch.models import SingleTaskGP
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.acquisition import qUpperConfidenceBound, qKnowledgeGradient, qExpectedImprovement
from botorch.optim import optimize_acqf
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from botorch.utils.transforms import standardize

# %%

def find_closest_element_index(sorted_list, target):
    left = 0
    right = len(sorted_list) - 1

    # print(f"find_closest: target = {target}. len(sorted_list) = {len(sorted_list)}. sorted_list = {sorted_list}.")
    while left <= right:
        mid = (left + right) // 2
        # print(f"find_closest: left = {left}. right = {right}. mid = {mid}.")
        if sorted_list[mid] == target:
            return mid

        if sorted_list[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    # 如果循环结束仍然没有找到精确匹配，此时left和right会指向最接近元素的两个位置
    # 计算它们之间的距离并返回最接近元素的位置
    if left == len(sorted_list):
        left -= 1

    if abs(sorted_list[left] - target) < abs(sorted_list[right] - target):
        return left
    else:
        return right

# %%

def locate_point_quantile(cumsum_arr, val_arr):
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    cumsum_arr = cumsum_arr.astype('float64')
    val_arr = val_arr.astype('float64')

    val_arr /= 2.0
    total_sum = cumsum_arr[-1]
    point_arr = cumsum_arr[:-1] + val_arr
    quantile_arr = point_arr / total_sum
    # print(f"locate_point_quantile: arr = {quantile_arr}")
    return quantile_arr

# %%

class BayesSelector(object):
    """
    {Description}
    实例特征表示: 包含两部分的特征，一是位置特征[p0, p1, ..., pn]，二是范围特征[r0, r1, ..., rn]
    位置特征表示该谓词中心的位置，范围特征表示该位置的大小

    Members:
        field1:
        field2:
    """

    def __init__(self, workload: str, mode: str, target_func: types.FunctionType, \
            column_info: dict, column_order: list, batch_num = 5):
        """
        {Description}

        Args:
            workload:
            mode:
            target_func:
            column_info: {
                ("table_name", "column_name"): {
                    "bins_list": [],
                    "marginal_list": [],
                    "reverse_dict": {}
                },

            }
            batch_num: 
        """
        self.workload, self.mode = workload, mode
        self.target_func = target_func
        self.column_info = column_info
        self.batch_num = batch_num

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 测试下CPU能不能快一点
        # self.device = torch.device("cpu")

        # self.column_order = list(sorted(column_info.keys()))
        self.column_order = column_order
        self.feature_num = len(column_order) * 2

        self.construct_marginal_cumsum()
        assert len(self.column_order) == len(self.column_info), \
            f"BayesSelector: column_order = {self.column_order}. column_info = {self.column_info.keys()}."
            # f"BayesSelector: len(column_order) = {len(self.column_order)}. len(column_info) = {len(self.column_info)}."

        # 测试初始化的正确性
        feature_vector_list, _, metrics_list, _, _ = self.init_samples(num=50, init_mode="feature_vector")
        # feature_vector_list, _, metrics_list, _, _ = self.init_samples(num=50, init_mode="range_dict")

        self.X_scaled = np.array(feature_vector_list)
        self.metrics_list = metrics_list
        # print(f"BayesSelector.__init__: metrics_list = {metrics_list}.")
        # sys.exit("Exit on BayesSelector.__init__")

        # 确定每个变量的边界
        self.bounds_tensor = torch.tensor([[0.0] * self.feature_num, 
            [1.0] * self.feature_num], device=self.device, dtype=torch.double)

    def construct_marginal_cumsum(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        key_list = list(self.column_info.keys())
        for k in key_list:
            v = self.column_info[k]
            marginal_list = v['marginal_list']
            # bins_list = v['bins_list']

            # print(f"construct_marginal_cumsum: k = {k}. len(marginal_list) = {len(marginal_list)}. marginal_list = {marginal_list}. ")
            # print(f"construct_marginal_cumsum: k = {k}. len(bins_list) = {len(bins_list)}. bins_list = {bins_list}.")
            # print()

            marginal_cumsum = np.cumsum(marginal_list)
            # total_sum = np.sum(marginal_list)
            marginal_cumsum = np.insert(marginal_cumsum, 0, 0)
            self.column_info[k]['marginal_cumsum'] = marginal_cumsum / marginal_cumsum[-1]
            self.column_info[k]['point_quantile'] = \
                locate_point_quantile(marginal_cumsum, marginal_list)
        
        return self.column_info


    def init_samples(self, num, init_mode = "range_dict"):
        """
        {Description}
    
        Args:
            num:
            init_mode:
        Returns:
            feature_vector_list: 
            range_dict_list: 
            metrics_list: 
            true_card_list: 
            est_card_list:
        """
        assert init_mode in ("range_dict", "feature_vector")
        feature_vector_list, range_dict_list = [], []
        for idx in range(num):
            if init_mode == "range_dict":
                # 先生成range_dict，然后构造feature_vector
                range_dict = self.generate_random_predicates()
                feature_vector = self.range2feature(range_dict)

            elif init_mode == "feature_vector":
                # 先生成feature_vector，然后构造range_dict
                feature_vector = self.generate_random_values()
                range_dict = self.feature2range(feature_vector)
            
            feature_vector_list.append(feature_vector)
            range_dict_list.append(range_dict)

        metrics_list, true_card_list, est_card_list = \
            self.target_func(range_dict_list)
        
        return feature_vector_list, range_dict_list, metrics_list, true_card_list, est_card_list
    


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
    

    def generate_random_predicates(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        range_dict = {}
        for k, v in self.column_info.items():
            bins_local = v['bins_list']
            start_idx, end_idx = self.random_condition(len(bins_local))
            start_val, end_val = utils.predicate_transform(\
                bins_local, start_idx, end_idx)
            range_dict[k] = start_val, end_val

        return range_dict
    
    def generate_random_values(self,):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        return np.random.random(size=self.feature_num)

    def feature2range(self, feature_vector):
        """
        {Description}

        Args:
            feature_list:
            arg2:
        Returns:
            range_out:
            return2:
        """
        # 判断维数是否一致
        column_num = len(self.column_info)
        assert len(feature_vector) == column_num * 2

        center_list = []
        range_dict = {}

        # 确定center的位置
        for val, column in zip(feature_vector[:column_num], self.column_order):
            center_list.append(self.val2idx(val, column))

        # 确定start和end的位置
        for idx, (val, column) in enumerate(zip(feature_vector[column_num:], self.column_order)):
            bins_list = self.column_info[column]['bins_list']
            start_idx, end_idx = self.val2range(center_list[idx], val, column)
            range_dict[column] = utils.predicate_transform(bins_list, start_idx, end_idx)

        return range_dict

    def range2feature(self, range_dict):
        """
        {Description}

        Args:
            range_dict:
            arg2:
        Returns:
            feature_vector:
            return2:
        """
        feature_list = []

        for col in self.column_order:
            start_val, end_val = range_dict[col]
            reverse_dict = self.column_info[col]['reverse_dict']
            marginal_cumsum = self.column_info[col]['marginal_cumsum']
            start_idx, end_idx = utils.predicate_location(\
                reverse_dict, start_val, end_val)
            
            center_idx = (start_idx + end_idx) // 2
            feature_list.append(self.idx2val(center_idx, col))

        for col in self.column_order:
            start_val, end_val = range_dict[col]
            reverse_dict = self.column_info[col]['reverse_dict']
            marginal_cumsum = self.column_info[col]['marginal_cumsum']
            start_idx, end_idx = utils.predicate_location(\
                reverse_dict, start_val, end_val)
            
            start_quantile, end_quantile = \
                marginal_cumsum[start_idx], marginal_cumsum[end_idx]
            delta_val = end_quantile - start_quantile
            feature_list.append(delta_val)

        
        return np.array(feature_list)
    
    def val2idx(self, in_value, column):
        """
        将输入的值转换成输出的索引
        
        Args:
            in_value:
            arg2:
        Returns:
            out_index:
            res2:
        """
        bins_list = self.column_info[column]['bins_list']
        slot_num = len(bins_list) - 1
        out_idx = int(np.floor(in_value * slot_num))
        # print(f"val2idx: in_value = {in_value: .2f}. slot_num = {slot_num}. column = {column}. out_idx = {out_idx}")
        return out_idx
    
    def idx2val(self, in_idx, column):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        bins_list = self.column_info[column]['bins_list']
        # slot_num = len(bins_list) - 1
        slot_num = len(bins_list) - 1
        # print(f"idx2val: slot_num = {slot_num}. in_idx = {in_idx}.")
        return (1.0 * in_idx / slot_num) + (0.5 / slot_num)

    def val2range(self, center_idx, in_value, column):
        """
        {Description}
        
        Args:
            center_idx: 
            in_value:
            column:
        Returns:
            start_idx:
            end_idx:
        """
        point_arr = self.column_info[column]['point_quantile']
        ratio_arr = self.column_info[column]['marginal_cumsum']

        try:
            center_val = point_arr[center_idx]
        except IndexError:
            center_val = point_arr[-1]
            
        in_value /= 2.0
        start_val, end_val = center_val - in_value, center_val + in_value

        start_idx = find_closest_element_index(ratio_arr, target=start_val)
        end_idx = find_closest_element_index(ratio_arr, target=end_val)

        return start_idx, end_idx


    def transform_metric(self, Y_input):
        """
        {Description}
    
        Args:
            Y_input:
            arg2:
        Returns:
            Y_transformed:
            return2:
        """
        # 暂时不进行处理
        # Y_input[Y_input < 1e-4] = 1e-4
        # Y_transformed = np.log(Y_input)
        Y_transformed = Y_input
        # print(f"transform_metric: Y_input = {Y_input[:10].squeeze()}. Y_transformed = {Y_transformed[:10].squeeze()}.")
        return Y_transformed

    def step(self, iter_num = 1, batch_num = 5):
        """
        执行优化的步骤，每次都新建一个GP模型，并且优化超参数
        
        Args:
            iter_num:
            batch_num:
        Returns:
            range_dict_total:
            feature_vector_total:
            metrics_total:
            true_card_total:
            est_card_total:
        """
        range_dict_total, feature_vector_total, metrics_total = [], [], []
        true_card_total, est_card_total = [], []

        for _ in range(iter_num):
            Y = np.array(self.metrics_list).reshape(-1, 1)
            Y_transformed = self.transform_metric(Y)

            train_X, train_y = torch.tensor(self.X_scaled, \
                device=self.device), torch.tensor(Y_transformed, device=self.device)     # 构造训练和标签数据的tensor
            train_y = standardize(train_y)

            # print(f"step: train_y = {train_y}")
            # 调用
            model = SingleTaskGP(train_X, train_y)                      # 新建SingleTask
            mll = ExactMarginalLogLikelihood(model.likelihood, model)   # 确定likehood的概率
            fit_gpytorch_model(mll)                                     # 拟合模型
            
            # 采集函数(考虑多种Acqusition Function)

            qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
            qEI = qExpectedImprovement(
                model=model,
                # best_f=0.2,
                best_f=train_y.max(),
                sampler = qmc_sampler
            )

            # qUCB = qUpperConfidenceBound(
            #     model, 0.1, qmc_sampler
            # )

            # NUM_FANTASIES = 128
            # qKG = qKnowledgeGradient(model, num_fantasies=NUM_FANTASIES)

            candidates, acq_value = optimize_acqf(
                acq_function = qEI,    
                # acq_function = qKG,
                bounds = self.bounds_tensor,
                q = batch_num, 
                num_restarts = 10, 
                raw_samples = 128,
                options = {
                    "batch_limit": 5,
                    "maxiter": 50
                }
            )
            new_x = candidates.detach()
            # print(f"new_x = {new_x}")
            feature_array = new_x.cpu().numpy()

            # 测试结果的shape
            # print(f"step: new_x.shape = {new_x.shape}. feature_array.shape = {feature_array.shape}")    

            range_dict_local = []
            for feature_vector in feature_array:
                range_dict = self.feature2range(feature_vector)
                # print(f"step: range_dict = {range_dict}")
                range_dict_local.append(range_dict)

            # 通过目标函数获得指标
            metrics_local, true_card_local, est_card_local = self.target_func(range_dict_local)   

            # 更新原有的训练数据
            self.X_scaled = np.vstack((self.X_scaled, feature_array))
            self.metrics_list.extend(metrics_local)
            # sys.exit("line 4XX: step iteration. eval new_x")

            # 更新本次探索的结果
            feature_vector_total.extend(feature_array.tolist())
            metrics_total.extend(metrics_local), range_dict_total.extend(range_dict_local)
            true_card_total.extend(true_card_local), est_card_total.extend(est_card_local)

        # print(f"metrics_total = {metrics_total}")
        return range_dict_total, feature_vector_total, metrics_total, true_card_total, est_card_total
    
    def explore_query(self, sample_num):
        """
        {Description}
    
        Args:
            sample_num:
            arg2:
        Returns:
            range_dict: 
            true_card: 
            est_card:
        """
        batch_num = self.batch_num
        iter_num = int(np.ceil(sample_num / self.batch_num))

        range_dict_total, feature_vector_total, metrics_total, \
            true_card_total, est_card_total = self.step(iter_num, batch_num)

        return self.select_best_instance(range_dict_total, feature_vector_total, \
                metrics_total, true_card_total, est_card_total)


    def select_best_instance(self, range_dict_list, feature_vector_list, metrics_list, true_card_list, est_card_list): 
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            range_dict: 
            true_card: 
            est_card:
        """
        instance_list = list(zip(range_dict_list, feature_vector_list, \
                                 metrics_list, true_card_list, est_card_list))
        instance_list.sort(key=lambda a: a[2], reverse=True)

        range_dict, feature_vector, _, true_card, est_card = instance_list[0]
        print(f"select_best_instance: feature_vector = {feature_vector}.")
        return range_dict, true_card, est_card
    
# %%
