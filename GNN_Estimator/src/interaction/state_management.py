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
import shutil

# %%
from utilities import graph_data_process, utils, global_config
from models import gcn_model, dropout_bayes_model, gat_model, deep_ensemble_model
from data import query_dataset, dataloader

import torch
# import torch.nn as nn
from data.dataloader import get_train_test_loader
# from training.optimizer import get_optimizer, get_lr_scheduler
from training import gnn_train, deep_ensemble_train, dropout_bayes_train
from data import data_transform
from torch_geometric.data import Data, Batch

# %%
import warnings
from sklearn.metrics import r2_score
from utilities.global_config import *
from multiprocessing import Process, Lock
from utilities import global_config
from utilities.global_config import hyper_params_dict

# 重新训练模型的进程锁
retrain_lock = Lock()

# %%
command_template = f"{global_config.python_path} execution_unit.py --model_type {{model_type}} --out_path {{out_path}} "\
                   "--option {option} --signature {signature} --workload {workload}"

# %%

class StateManager(object):
    """
    {Description}
    20231216: 需要实现模型训练的异步化

    Members:
        field1:
        field2:
    """

    def __init__(self, workload, method, signature, model_type, \
            model_init_num = 30, model_update_num = 5, auto_update = False):
        """
        {Description}

        Args:
            workload: 数据集
            method: 基数估计方法
            signature: 结果路径签名
            model_type: 可选的模型类型("gat", "gcn", "dropout_bayes", "deep_ensemble")
            model_init_num: 
            model_update_num:
            auto_update:
        """
        model_list = ["gat", "gcn", "dropout_bayes", "deep_ensemble"]
        assert model_type in model_list, f"StateManager: available mode_list = {model_list}"

        self.workload, self.method, self.signature = \
            workload, method, signature

        self.model_type, self.auto_update = model_type, auto_update
        self.train_dataset = None

        # 初始化状态下将模型设置为空
        self.GNN_model, self.previous_length, self.model_id = None, None, -1

        self.instance_list, self.meta_list, self.selection_list  = [], [], []     # 查询实例的列表
        self.query_encoder = data_transform.QueryEncoder(workload)

        # data_root = "/home/jinly/GNN_Estimator/online_data/"
        data_root = global_config.online_data_dir

        self.model_meta_path = p_join(data_root, \
            self.workload, self.signature, "model", "model_meta.json")
        # self.model_meta_dict = {}

        # 创建路径
        # os.makedirs(self.dir_path, exist_ok=True)
        self.model_init_num, self.model_update_num = model_init_num, model_update_num
        self.root = data_root
        self.sample_num = 1
        self.create_directory()


    def set_sample_num(self, new_num: int):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        self.sample_num = new_num

    def eval_model_update(self,):
        """
        判断模型是否需要更新
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """

        if (self.previous_length is None and len(self.instance_list) > self.model_init_num) or \
            (self.previous_length is not None and len(self.instance_list) > self.previous_length + self.model_update_num):
            print(f"eval_model_update: len(instance_list) = {len(self.instance_list)}. " \
                f"selection_num = {self.get_selected_number()}. previous_len = {self.previous_length}. model_id = {self.model_id}.")
            
            if retrain_lock.acquire(block=False) == True:
                # 尝试获取锁，如果获取成功
                self.previous_length = len(self.instance_list)
                model_proc = Process(target = self.model_retrain, args = (retrain_lock,))
                model_proc.start()
                print("eval_model_update: acquire = True.")
            else:
                print("eval_model_update: acquire = False.")
            return True
        else:
            return False
    

    def infer_max_id(self, model_meta_dict):
        """
        {Description}
    
        Args:
            model_meta_dict:
            arg2:
        Returns:
            id:
            return2:
        """
        if len(model_meta_dict) != 0:
            max_id = max([int(id) for id in model_meta_dict.keys()])
        else:
            max_id = -1

        return max_id


    def model_retrain(self, lock: Lock):
        """
        重新训练模型
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        # 制作新的数据集
        self.make_train_dataset(update_member = True, with_selection = True)

        # 确定新模型的ID
        model_meta_dict: dict = utils.load_json(self.model_meta_path)
        max_id = self.infer_max_id(model_meta_dict) + 1

        print(f"model_retrain: max_id = {max_id}. len(train_dataset) = {len(self.train_dataset)}.")

        model_path = p_join(self.root, self.workload, self.signature, "model", f"model_{max_id}.pth")
        self.model_train(self.model_type, out_path=model_path)
        # 导入新模型的元信息
        model_meta_dict[str(max_id)] = model_path
        utils.dump_json(model_meta_dict, self.model_meta_path)

        # if lock.locked() == True:
        try:
            lock.release()
        except ValueError as e:
            return
        
    def model_reload(self,):
        """
        重新加载模型，选择model_id最大的进行加载
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        model_meta_dict: dict = utils.load_json(self.model_meta_path)
        max_id = self.infer_max_id(model_meta_dict)

        if max_id != self.model_id:
            print(f"model_reload: max_id = {max_id}. model_id = {self.model_id}. 加载GNN模型")

            self.model_id = max_id      # 更新当前model_id
            try:
                model_path = model_meta_dict[str(max_id)]
                self.GNN_model = self.load_model(self.model_type, model_path)
            except KeyError as e:
                print(f"model_reload: meet KeyError. model_meta_dict = {model_meta_dict}.")
                raise e
        # else:
        #     print("model_reload: ")
        return self.GNN_model
    
    def clean_on_signature(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        data_dir = p_join(self.root, self.workload, self.signature)

        print(f"clean_on_signature: data_dir = {data_dir}.")
        # print(f"clean_on_signature: model_dir = {model_dir}.")

        # self.clean_directory(data_dir)
        # self.clean_directory(model_dir)

    # def clean_directory(self, dir_path):
    #     """
    #     清理目录下的所有资源
    
    #     Args:
    #         arg1:
    #         arg2:
    #     Returns:
    #         return1:
    #         return2:
    #     """
    #     shutil.rmtree(dir_path)

    def create_directory(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        # 创建数据的目录
        dir_path = p_join(self.root, self.workload, self.signature, "raw")
        os.makedirs(dir_path, exist_ok=True)
        self.dir_path = dir_path

        # 创建模型的目录
        model_dir = p_join(self.root, self.workload, self.signature, "model")
        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir


    def clean_directory(self,):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        raw_dir = p_join(self.root, self.workload, self.signature, "raw")
        model_dir = p_join(self.root, self.workload, self.signature, "model")
        processed_dir = p_join(self.root, self.workload, self.signature, "processed")

        if os.path.isdir(raw_dir) == True:
            shutil.rmtree(raw_dir)

        if os.path.isdir(model_dir) == True:
            shutil.rmtree(model_dir)

        if os.path.isdir(processed_dir) == True:
            shutil.rmtree(processed_dir)
        
        self.create_directory()

    def get_description(self,):
        """
        获得当前的状态
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        print(f"get_description: workload = {self.workload}. method = {self.method}. signature = {self.signature}.")
        return self.workload, self.method, self.signature

    def get_selected_number(self,):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        return np.sum(self.selection_list)

    def meta_contain(self, meta_1, meta_2):
        """
        {Description}
    
        Args:
            meta_1:
            meta_2:
        Returns:
            flag:
            return2:
        """
        schema_list_1, filter_list_1 = meta_1
        schema_list_2, filter_list_2 = meta_2

        schema_flag = set(schema_list_1).issubset(schema_list_2)
        filter_flag = set(filter_list_1).issubset(filter_list_2)

        return schema_flag and filter_flag

    def update_selection_status(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        for idx, selection in enumerate(self.selection_list):
            if selection == False:
                continue
            
            meta_1 = self.meta_list[idx]
            for meta_2, sel_2 in zip(self.meta_list, self.selection_list):
                if sel_2 == False:
                    continue
                if self.meta_contain(meta_1, meta_2) == True:
                    self.selection_list[idx] = False

        return self.selection_list

    def make_train_dataset(self, update_member = False, with_selection = False):
        """
        制作训练数据集
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        file_content = f"{self.workload}_0.pkl"

        # 没有文件，执行保存操作
        f_path = p_join(self.dir_path, file_content)
        if os.path.isfile(f_path) == False:
            print("make_train_dataset: 没有找到文件，执行数据保存")
            self.save_data(with_selection)

        train_dataset = query_dataset.QueryGraphDataset(root=self.root, \
            workload=self.workload, file_content=file_content, \
            signature=self.signature, clean_data=True)
        
        if update_member:
            self.train_dataset = train_dataset
        return train_dataset
    
    def add_train_data(self, meta_list, card_dict_list):
        """
        {Description}
        
        Args:
            meta_list: 
            card_dict_list:
        Returns:
            instance_list:
            res2:
        """
        result_list = graph_data_process.process_instance_list(\
            self.workload, zip(meta_list, card_dict_list))
        self.instance_list.extend(result_list)

        # 更新样本选择的信息
        self.meta_list.extend(meta_list)
        self.selection_list.extend([True for _ in meta_list])

        # 自动模型更新
        if self.auto_update == True:
            self.eval_model_update()

        return self.instance_list

    def save_data(self, with_selection = False):
        """
        保存数据
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        out_path = p_join(self.dir_path, f"{self.workload}_0.pkl")

        if with_selection == False:
            # 导出所有数据
            utils.dump_pickle(self.instance_list, out_path)
        else:
            # 导出部分数据
            instance_sublist = []
            for instance, selection in zip(self.instance_list, self.selection_list):
                if selection == True:
                    instance_sublist.append(instance)

            utils.dump_pickle(instance_sublist, out_path)

    def encode_single_instance(self, idx, attr_dict, graph_dict):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        encoder = self.query_encoder
        feature_vector, edge_index = None, None   #
        src_list, dst_list = [], []
        for src_idx, dst_idx in graph_dict['edge_set']:
            src_list.append(src_idx), dst_list.append(dst_idx)

        edge_index=torch.stack([torch.tensor(src_list,dtype=torch.long), \
                                torch.tensor(dst_list,dtype=torch.long)],dim=0)
        try:
            card_list = [v['true_card'] for v in attr_dict.values()]
        except KeyError as e:
            print(f"encode_single_instance: meet KeyError. v = {list(attr_dict.values())[0]}.")
            raise e
        
        encoder.load_card_info(idx, card_list)

        local_data = encoder.encode_graph(attr_dict = attr_dict, \
            mask_num = None, edge_index=edge_index)
        return local_data


    def make_test_instance(self, query_meta, card_dict):
        """
        构造测试实例
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        self.query_encoder.reset()     # 重置encoder
        # 提取信息
        attr_dict, graph_dict = graph_data_process.\
            process_instance(self.workload, query_meta, card_dict)
        return self.encode_single_instance(0, attr_dict, graph_dict)


    def make_test_dataset(self, meta_list, card_dict_list):
        """
        构造测试数据集

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        result_list = graph_data_process.process_instance_list(\
            self.workload, zip(meta_list, card_dict_list))
        self.query_encoder.reset()     # 重置encoder
        data_list = []
        for idx, (attr_dict, graph_dict) in enumerate(result_list):
            # print(f"make_test_dataset: idx = {idx}. attr_dict = {attr_dict}.", flush=True)
            local_data = self.encode_single_instance(idx, attr_dict, graph_dict)
            data_list.append(local_data)

        return data_list
        

    def create_model(self, model_type):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        model_type = model_type.lower()
        print(f"StateManager.create_model: model_type = {model_type}.")
        self.model_type = model_type

        # if model_type == "gcn":
        #     GNN_model = gcn_model.GCNModel(workload=self.workload, hidden_channels_1=200, embedding_dims=50)
        # elif model_type == "gat":
        #     GNN_model = gat_model.GATModel(workload=self.workload, hidden_channels_1=200, embedding_dims=50)
        # elif model_type == "dropout_bayes":
        #     GNN_model = dropout_bayes_model.GNNModel(workload=self.workload, hidden_size=100, \
        #         embedding_dims=50, layer_type="gat", dropout_rate=0.2, decay=1e-6)
        # elif model_type == "deep_ensemble":
        #     # GNN_model = deep_ensemble_model.GaussianGNN(hidden_channels=100, \
        #     #     embedding_dims=50)
        #     GNN_model = deep_ensemble_model.create_model_list(100, 50, 10)
        # else:
        #     raise ValueError(f"launch_train_with_eval: Unsupported model({model_type})")

        if model_type == "gcn":
            GNN_model = gcn_model.GCNModel(workload=self.workload, **hyper_params_dict["GCN"])
        elif model_type == "gat":
            GNN_model = gat_model.GATModel(workload=self.workload, **hyper_params_dict["GAT"])
        elif model_type == "dropout_bayes":
            GNN_model = dropout_bayes_model.GNNModel(workload=self.workload, **hyper_params_dict["dropout_bayes"])
        elif model_type == "deep_ensemble":
            # GNN_model = deep_ensemble_model.GaussianGNN(hidden_channels=100, \
            #     embedding_dims=50)
            GNN_model = deep_ensemble_model.create_model_list(**hyper_params_dict["deep_ensemble"])
        else:
            raise ValueError(f"launch_train_with_eval: Unsupported model({model_type})")

        return GNN_model

    def model_train(self, model_type: str, out_path = "model.pth"):
        """
        训练图神经网络模型，根据model_type的不同调用不同的训练函数
    
        Args:
            model_type: 模型的类型
            out_path: 结果的保存路径
        Returns:
            return1:
            return2:
        """
        # file_dir = "/home/jinly/GNN_Estimator/src"
        file_dir = global_config.source_code_dir
        curr_dir = os.getcwd()

        os.chdir(file_dir)
        command_params = {
            "model_type": model_type,
            "out_path": out_path,
            "option": "train",
            "signature": self.signature,
            "workload": self.workload
        }
        
        local_command = command_template.format(**command_params) + \
            f" >> {p_join(global_config.output_dir, 'unit_out.txt')}"
        
        print(f"model_train: local_command = {local_command}.")
        os.system(local_command)
        os.chdir(curr_dir)

    def load_model(self, model_type, model_path):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            GNN_model:
            return2:
        """
        print(f"StateManager.load_model: model_type = {model_type}. model_path = {model_path}.")
        if model_type.lower() in ("gcn", "gat", "gin"):
            #
            GNN_model = self.create_model(model_type)
            state_dict = torch.load(model_path)
            GNN_model.load_state_dict(state_dict)
        elif model_type.lower() == "dropout_bayes":
            # 
            GNN_model = self.create_model(model_type)
            state_dict = torch.load(model_path)

            params_path = model_path.replace(".pth", "_params.pkl")
            GNN_model.load_state_dict(state_dict)
            GNN_model.load_params(params_path)
        elif model_type.lower() == "deep_ensemble":
            # 创建模型
            GNN_model = self.create_model(model_type)
            deep_ensemble_train.ensemble_model_load(GNN_model, model_path)
            # raise NotImplementedError("model_type.lower() == 'deep_ensemble'")
        return GNN_model
    
    def infer_on_instance(self, query_meta, card_dict, reload = True):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            flag:
            subquery_res:
            single_table_res:
        """
        flag, out_res = self.infer_on_list([query_meta,], [card_dict,], reload)
        
        if flag == False:
            return flag, {}, {}
        else:
            subquery_res, single_table_res = out_res[0]
            return flag, subquery_res, single_table_res
        
    def infer_on_instance_with_uncertainty(self, query_meta, card_dict, reload = True):
        """
        专门用于dropout_bayesian，给出分布的结果而不是sample后的结果，用来对比不同sample的效果
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        meta_list, card_dict_list = [query_meta, ], [card_dict, ]

        if reload == True:
            self.model_reload()     # 尝试重新加载模型
        if self.GNN_model is None:
            # 如果
            return False, {}, {}
        
        data_list = self.make_test_dataset(meta_list, card_dict_list)
        num_list, idx_map_list = self.build_result_index(data_list)
        data_batch = Batch.from_data_list(data_list)

        data_num = self.GNN_model.training_num
        out_mean, out_std = dropout_bayes_train.inference_with_uncertainty(\
            self.GNN_model, data_batch, data_num=data_num)
        mask = data_batch.test_mask
        selected_mean, selected_std = out_mean[mask], out_std[mask]
        # print(f"infer_on_list: mean_shape = {selected_mean.shape}. std_shape = {selected_std.shape}.")

        # out_sample = dropout_bayes_train.sample_on_prob(selected_mean, selected_std, sample_num)
        out_params = dropout_bayes_train.pack_prob(selected_mean, selected_std)
        result_list = self.split_result(out_params, num_list)
        # print(f"infer_on_instance_with_uncertainty: num_list = {num_list}. out_params = {out_params}. result_list = {result_list}.")
        out_dict_list = []
        
        for graph_idx, (node_idx_map, local_params) in enumerate(zip(idx_map_list, result_list)):
            # 
            # assert len(local_params) == 2, f"infer_on_instance_with_uncertainty: len(local_params) = {len(local_params)}."
            out_dict = self.query_encoder.restore_origin_distributions(\
                graph_idx, node_idx_map, local_params)
            
            # 转化成subquery的格式，不使用op_func
            subquery_out, single_table_out = self.query_encoder.\
                convert_to_subquery_format(graph_idx, out_dict, op_func=None)
            out_dict_list.append((subquery_out, single_table_out))

        # return True, out_dict_list
        subquery_res, single_table_res = out_dict_list[0]
        return True, subquery_res, single_table_res

    def infer_on_list(self, meta_list, card_dict_list, reload = True):
        """
        {Description}
        
        Args:
            meta_list:
            card_dict_list:
        Returns:
            flag: 推理是否成功
            out_dict_list: 神经网络的推理结果
        """
        model_type, sample_num = self.model_type, self.sample_num

        if reload == True:
            self.model_reload()     # 尝试重新加载模型
        if self.GNN_model is None:
            # 如果
            return False, []

        data_list = self.make_test_dataset(meta_list, card_dict_list)
        num_list, idx_map_list = self.build_result_index(data_list)
        data_batch = Batch.from_data_list(data_list)

        # print(f"infer_on_list(type): data_list = {type(data_list)}. "\
        #       f"data_batch = {type(data_batch)}. data_list[0] = {type(data_list[0])}.")
        # print(f"infer_on_list: num_graphs = {data_batch.num_graphs}. num_list = {num_list}.")

        # 根据模型类型执行不同的推理步骤
        if model_type in ("gcn", "gat"):
            out = self.GNN_model(data_batch.x, data_batch.edge_index)
            # print(f"infer_on_list: shape of out = {out.shape}. test_mask = {data_batch.test_mask}")
            result_list = self.split_result(out[data_batch.test_mask], num_list)
            out_dict_list = []
            
            for graph_idx, (node_idx_map, nn_out) in enumerate(zip(idx_map_list, result_list)):
                out_dict = self.query_encoder.restore_origin_values(\
                    graph_idx, node_idx_map, nn_out)
                subquery_out, single_table_out = self.query_encoder.\
                    convert_to_subquery_format(graph_idx, out_dict, op_func=int)
                # out_dict_list.append(out_dict)
                out_dict_list.append((subquery_out, single_table_out))
        elif model_type == "dropout_bayes":
            data_num = self.GNN_model.training_num
            out_mean, out_std = dropout_bayes_train.inference_with_uncertainty(\
                self.GNN_model, data_batch, data_num=data_num)
            mask = data_batch.test_mask
            selected_mean, selected_std = out_mean[mask], out_std[mask]
            # print(f"infer_on_list: mean_shape = {selected_mean.shape}. std_shape = {selected_std.shape}.")

            out_sample = dropout_bayes_train.sample_on_prob(selected_mean, selected_std, sample_num)
            result_list = self.split_result(out_sample, num_list)
            out_dict_list = []
            
            for graph_idx, (node_idx_map, local_sample) in enumerate(zip(idx_map_list, result_list)):
                out_dict = self.query_encoder.restore_origin_values(\
                    graph_idx, node_idx_map, local_sample)
                subquery_out, single_table_out = self.query_encoder.\
                    convert_to_subquery_format(graph_idx, out_dict, op_func=int)
                # out_dict_list.append(out_dict)
                out_dict_list.append((subquery_out, single_table_out))
            
        elif model_type == "deep_ensemble":
            raise NotImplementedError("dropout_bayes has not implemented")

        return True, out_dict_list
    
    def build_result_index(self, data_list):
        """
        {Description}
        
        Args:
            data_list:
            arg2:
        Returns:
            num_list: 求解变量个数列表
            idx_map_list: 索引映射列表
        """
        num_list, idx_map_list = [], []

        for data in data_list:
            test_mask: torch.Tensor = data.test_mask
            num = int(test_mask.sum())
            num_list.append(num)
            idx_all = torch.where(test_mask==True)[0].tolist()
            local_map = {idx: val for idx, val in enumerate(idx_all)}
            idx_map_list.append(local_map)

        return num_list, idx_map_list

    def split_result(self, out_tensor, num_list):
        """
        划分神经网络的预测结果
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        cumsum_list = [0,]
        val = 0
        for i in num_list:
            val += i
            cumsum_list.append(val)

        start_idx_list = cumsum_list[:-1]
        end_idx_list = cumsum_list[1:]

        if isinstance(out_tensor, torch.Tensor):
            out_list = out_tensor.detach().tolist()
        elif isinstance(out_tensor, list):
            out_list = out_tensor
        elif isinstance(out_tensor, np.ndarray):
            out_list = list(out_tensor)
        else:
            raise TypeError(f"split_result type(out_tensor) = {type(out_tensor)}.")

        # print(f"split_result: out_list = {out_list}.")
        result_list = [out_list[s: e] for s, e \
            in zip(start_idx_list, end_idx_list)]
        return result_list
    

# %%
