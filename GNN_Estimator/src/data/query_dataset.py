#!/usr/bin/env python
# -*- coding: utf-8 -*-


# %%

from typing import Optional, Callable, List, Tuple, Union
from torch_geometric.data import InMemoryDataset, Data, download_url
import numpy as np
import pandas as pd
from os.path import join as p_join
from matplotlib import pyplot as plt
import json, os, time, re, math
import pickle
from copy import deepcopy

from torch_geometric.data.data import BaseData
from torch_geometric.data.dataset import Dataset, IndexType
from data import data_transform
from utilities import utils, global_config
import torch
from collections import Counter
import shutil

# %%

def read_query_graph_data(raw_dir, raw_file_names, \
    encoder:data_transform.QueryEncoder, augumenter: data_transform.DataAugumenter):
    """
    读取查询图数据
    
    Args:
        raw_dir:
        raw_file_names:
    Returns:
        res1:
        res2:
    """

    data, res_list = [], []

    for raw_file in raw_file_names:
        try:
            raw_path = p_join(raw_dir, raw_file)
            local_list = utils.load_pickle(raw_path)
            res_list.extend(local_list)
        except TypeError as e:
            print(f"read_query_graph_data: meet TypeError. raw_path = {raw_path}")
            raise e
        
    augumented_res_list = []
    for attr_dict, graph_dict in res_list:
        augumenter.load_instance(attr_dict, graph_dict)
        instance_list = augumenter.augument_by_subgraph(clip_level = 1)
        augumented_res_list.extend(instance_list)
        augumented_res_list.append((attr_dict, graph_dict))
    
    print(f"read_query_graph_data: len(res_list) = {len(res_list)}. "\
          f"len(augumented_list) = {len(augumented_res_list)}.")

    for global_idx, (attr_dict, graph_dict) in enumerate(augumented_res_list):
        edge_index = None   #

        src_list, dst_list = [], []
        # print(f"read_query_graph_data: graph_dict = {graph_dict}")
        for src_idx, dst_idx in graph_dict['edge_set']:
            src_list.append(src_idx), dst_list.append(dst_idx)

        edge_index=torch.stack([torch.tensor(src_list,dtype=torch.long), \
                                torch.tensor(dst_list,dtype=torch.long)],dim=0)

        card_list = [v['true_card'] for v in attr_dict.values()]
        encoder.load_card_info(global_idx, card_list)    # 加载所有的基数

        local_data = encoder.encode_graph(attr_dict = \
            attr_dict, mask_num = 1, edge_index=edge_index)
        # local_data.edge_index = edge_index
        data.append(local_data)

    # return augumented_res_list
    return data

# %%

class QueryGraphDataset(InMemoryDataset):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    url = 'https://github.com/Julyuan/QueryData/raw/master/data'
    # def __init__(self, root = "/home/jinly/GNN_Estimator/data", workload = "stats", file_content="stats_0.pkl", \
    #         meta_info_dir: str = "/home/jinly/GNN_Estimator/config", transform: Optional[Callable] = None, \
    #         pre_transform: Optional[Callable] = None, mask_option = None, signature = "0", clean_data = True):
    def __init__(self, root = global_config.data_root, workload = "stats", file_content="stats_0.pkl", \
            meta_info_dir: str = global_config.meta_info_dir, transform: Optional[Callable] = None, \
            pre_transform: Optional[Callable] = None, mask_option = None, signature = "0", clean_data = True):
        """
        {Description}

        Args:
            root:
            workload:
            file_content:
            meta_info_dir:
            transform:
            pre_transform:
        """
        if clean_data == True:
            # 删除路径
            processed_dir = p_join(root, workload, signature, "processed")
            if os.path.isdir(processed_dir) == True:
                shutil.rmtree(processed_dir)
                os.makedirs(processed_dir)

        self.root, self.workload, self.file_content, self.signature = \
            root, workload, file_content, signature
        # self.processed_dir = p_join(self.root, self.workload, "processed")
        # print(f"QueryGraphDataset.__init__: processed_dir = {self.processed_dir}")
        # print(f"QueryGraphDataset.__init__: raw_file_names = {self.raw_file_names}")

        self.encoder = data_transform.QueryEncoder(workload=workload, meta_info_dir=meta_info_dir)
        self.augumenter = data_transform.DataAugumenter()

        super().__init__(root, transform, pre_transform)
        # self.graph_data_list = self.load_data(file_content)
        self.data, self.slices = torch.load(self.processed_paths[0])
        data = self.get(0)
        self.data, self.slices = self.collate([data])
        self.data_integrity_verification()

    @property
    def raw_dir(self) -> str:
        if self.signature is None:
            return p_join(self.root, self.workload, "raw")
        else:
            return p_join(self.root, self.workload, self.signature, "raw")
    
    @property
    def processed_dir(self) -> str:
        if self.signature is None:
            return p_join(self.root, self.workload, "processed")
        else:
            return p_join(self.root, self.workload, self.signature, "processed")
    # @property
    # def raw_paths(self) -> List[str]:
    #     files = self.raw_file_names
    #     print(f"raw_paths: files = {files}")
    #     return [p_join(self.raw_dir, f) for f in files]

    @property
    def raw_file_names(self) -> str | List[str] | Tuple:
        file_content = self.file_content
        file_list = []
        if file_content == "all":
            file_list = [f_name for f_name in os.listdir(\
                self.raw_dir) if f_name.endswith(".pkl")]
        elif isinstance(file_content, str):
            file_list = [file_content,]
        elif isinstance(file_content, list):
            file_list = file_content

        print(f"raw_file_names: file_list = {file_list}")
        return file_list

    @property
    def processed_file_names(self) -> str | List[str] | Tuple:
        return 'data.pt'

    # def download(self):
    #     for name in self.raw_file_names:
    #         download_url(f'{self.url}/{name}', self.raw_dir)

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: IndexType | int) -> Dataset | BaseData:
        # return super().__getitem__(idx)
        return self.data[idx]

    def process(self):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        data = read_query_graph_data(self.raw_dir, self.raw_file_names, self.encoder, self.augumenter)
        data = data if self.pre_transform is None else self.pre_transform(data)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])        
        return data
    
    def get_node_feature_number(self,):
        """
        获得节点特征的维数
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        return self.data[0].x.shape[1]

    def data_integrity_verification(self,):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        dimension_list = []

        for item in self.data:
            dimension_list.append(item.x.shape[1])

        assert(len(Counter(dimension_list)) == 1)
# %%