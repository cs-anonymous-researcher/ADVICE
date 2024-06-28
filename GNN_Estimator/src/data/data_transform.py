#!/usr/bin/env python
# -*- coding: utf-8 -*-


# %%
import torch
import torch.nn as nn
from os.path import join as p_join
import networkx as nx

# %%
from utilities import utils, global_config
import psycopg2 as pg
import numpy as np
from collections import defaultdict
from copy import copy, deepcopy
from matplotlib import pyplot as plt
from torch_geometric.data import Data

# %%

def filter_none(val_list):
    return [val for val in val_list if val is not None]

from functools import partial

def func1(card_max, card_min, a):
    return (a - card_min) / (card_max - card_min)

def func2(card_max, card_min, a):
    return card_min + a * (card_max - card_min)
    
# %%

class QueryEncoder(object):
    """
    查询信息的编码器

    Members:
        field1:
        field2:
    """

    def __init__(self, workload, meta_info_dir = global_config.meta_info_dir):
        """
        构造函数

        Args:
            arg1:
            arg2:
        """
        self.workload, self.graph_idx = workload, None
        self.est_card_dict, self.idx2key_dict = defaultdict(dict), defaultdict(dict)

        self.alias_mapping = utils.workload_alias_option[workload]
        self.meta_path = p_join(meta_info_dir, f"{self.workload}.json")
        self.alias_mapping = utils.workload_alias_option[workload]
        self.alias_reverse = {}
        self.card_list_dict = defaultdict(list)

        for k, v in self.alias_mapping.items():
            self.alias_reverse[v] = k

        self.load_meta()

        # 设置默认的成员函数
        self.normalize, self.abnormalize = lambda a: a, lambda b: b

    def reset(self,):
        """
        重置card_list_dict
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        self.card_list_dict = defaultdict(list)

    def load_meta(self,):
        """
        加载元数据信息

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        self.meta_dict = utils.load_json(self.meta_path)
        self.construct_position_dict(meta_dict=self.meta_dict)
        self.construct_minmax_dict(meta_dict=self.meta_dict)
        return self.meta_dict

    def update_meta(self,):
        """
        更新加载元数据信息
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        utils.dump_json(self.meta_dict, self.meta_path)

    
    def construct_position_dict(self, meta_dict: dict = None):
        """
        构造特征到embedding位置的字典
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        table_pos_dict, column_pos_dict = {}, {}
        tbl_cnt, col_cnt = 0, 0

        try:
            for table, column_list in meta_dict["column_info"].items():
                table_pos_dict[table] = tbl_cnt
                tbl_cnt += 1
                # alias = self.alias_mapping[table]
                for column in column_list:
                    # column_pos_dict[(alias, column)] = col_cnt
                    column_pos_dict[(table, column)] = col_cnt, col_cnt + 1
                    col_cnt += 2
        except KeyError as e:
            print(f"construct_position_dict: meet KeyError. meta_dict = {meta_dict.keys()}.")
            raise e
        
        self.table_pos_dict, self.column_pos_dict = table_pos_dict, column_pos_dict
        return table_pos_dict, column_pos_dict
    

    def load_card_info(self, idx, card_list):
        """
        加载图中的基数信息
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        # for k, v in self.meta_dict['column_info'].items():
        #     pass
        self.card_list_dict[idx].extend(card_list)
        self.set_current_idx(idx)

    def set_current_idx(self, idx):
        """
        设置当前的实例索引
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        card_local = self.card_list_dict[idx]
        card_local = filter_none(card_local)
        try:
            card_min, card_max = min(card_local), max(card_local)
        except TypeError as e:
            print(f"set_current_idx: meet TypeError. card_local = {card_local}.")
            raise e
         
        # self.normalize = lambda a: (a - card_min) / (card_max - card_min)
        # self.abnormalize = lambda a: card_min + a * (card_max - card_min)
        self.normalize = partial(func1, card_max, card_min)
        self.abnormalize = partial(func2, card_max, card_min)

        self.graph_idx = idx        # 设置当前的处理图ID

        return self.normalize, self.abnormalize

    def get_minmax_values(self, table, column_list):
        """
        获得一个表各列的最大/最小值
        
        Args:
            table:
            column_list:
        Returns:
            res1:
            res2:
        """
        minmax_query_template = "SELECT {column_content} FROM {table_name};"
        value_pair_list, field_list = [], []
        
        for column in column_list:
            field_list.append(f"MIN({column}) as min_{column}")
            field_list.append(f"MAX({column}) as max_{column}")

        query_text = minmax_query_template.format(column_content = \
                        ",\n".join(field_list), table_name = table)
        
        db_conn = pg.connect(**global_config.workload_conn_option[self.workload])
        with db_conn.cursor() as cursor:
            cursor.execute(query_text)
            result = cursor.fetchall()[0]
        
        for idx in range(0, len(result), 2):
            value_pair_list.append((result[idx], result[idx + 1]))
        # print(f"get_minmax_values: result = {result}. value_pair_list = {value_pair_list}.")

        return value_pair_list


    def construct_minmax_dict(self, meta_dict):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        minmax_dict = meta_dict['minmax_info']

        for table, column_list in meta_dict['column_info'].items():
            if table not in minmax_dict:
                minmax_dict[table] = {}

            column_missing = []
            for column in column_list:
                if column not in minmax_dict:
                    column_missing.append(column)
            
            value_pair_list = self.get_minmax_values(table, column_missing)

            # print("construct_minmax_dict: table = {}. column_missing = {}.".\
            #       format(table, column_missing))
            # print("construct_minmax_dict: table = {}. value_pair_list = {}.".\
            #       format(table, value_pair_list))
            
            for column, value_pair in zip(column_missing, value_pair_list):
                minmax_dict[table][column] = value_pair
            
        self.meta_dict['minmax_info'] = minmax_dict
        self.update_meta()      # 更新meta的信息
        return minmax_dict

    def table_encode(self, schema_list:list, table_pos_dict: dict):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        res_vector = [None for _ in range(len(table_pos_dict))]
        for table, pos in table_pos_dict.items():
            if table in schema_list:
                res_vector[pos] = 1
            else:
                res_vector[pos] = 0

        return res_vector

    def apply_transform(self, min_val, max_val, start_value, end_value):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        def value_trans(in_val, min_val, max_val):
            return (in_val - min_val) / (max_val - min_val)
        
        start_res = value_trans(start_value, min_val, max_val)
        end_res = value_trans(end_value, min_val, max_val)

        return start_res, end_res


    def column_encode(self, filter_list, column_pos_dict):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        res_vector = [None for _ in range(2 * len(column_pos_dict))]
        filter_local = {}
        for item in filter_list:
            start_val, end_val = item[2], item[3]
            alias, column_name = item[0], item[1]
            table_name = self.alias_reverse[alias]
            filter_local[(table_name, column_name)] = start_val, end_val

        minmax_dict = self.meta_dict['minmax_info']

        for table_name, column_dict in minmax_dict.items():
            for column_name, (min_val, max_val) in column_dict.items():
                pos1, pos2 = self.column_pos_dict[(table_name, column_name)]
                if (table_name, column_name) in filter_local:
                    start_val, end_val = filter_local[(table_name, column_name)]
                    res_vector[pos1], res_vector[pos2] = \
                        self.apply_transform(min_val, max_val, start_val, end_val)
                else:
                    res_vector[pos1], res_vector[pos2] = 0.0, 0.0

        return res_vector
    
    def cardinality_encode(self, estimation_card, true_card, mask: bool = False):
        """
        {Description}
        
        Args:
            estimation_card:
            true_card:
        Returns:
            card_vector:
            label:
        """
        # print(f"cardinality_encode: true_card = {true_card}.")
        est_card_normalized = self.normalize(estimation_card)   # 
        if true_card is not None:
            if true_card <= 1e-5 or estimation_card <= 1e-5:
                label = np.log((true_card + 1.0) / (estimation_card + 1.0))
            else:
                label = np.log(true_card / estimation_card)
            # print(f"cardinality_encode: true_card = {true_card}. est_card = {estimation_card}. "\
            #       f"ratio = {true_card / estimation_card:.3f}. label = {label:.3f}")
        else:
            label = 0.0
            
        if mask == False:
            state_indicator = 0
            return [est_card_normalized, state_indicator, label], label
        else:
            state_indicator = 1
            return [est_card_normalized, state_indicator, 0], label
    

    # def infer_true_card(self, est_card_normalized, label):
    def infer_true_card(self, graph_idx, node_idx, label):
        """
        使用denormalize，推断真实的基数
        
        Args:
            graph_idx:
            label:
        Returns:
            true_card_origin:
            res2:
        """
        # 
        try:
            est_card = self.est_card_dict[graph_idx][node_idx]
        except Exception as e:
            print(f"infer_true_card: meet KeyError. self.est_card_dict = {self.est_card_dict}."\
                  f"graph_idx = {graph_idx}. node_idx = {node_idx}.")
            raise e
        
        ratio = np.exp(label)
        try:
            true_card_origin = est_card * ratio
        except TypeError as e:
            print(f"infer_true_card: meet TypeError. est_card = {est_card}. ratio = {ratio: .3f}.")
            raise e

        # print(f"infer_true_card: ratio = {ratio: .2f}. est_card = {est_card: .2f}. true_card = {true_card_origin: .2f}.")
        return true_card_origin
    

    def restore_origin_values(self, graph_idx, node_idx_map, nn_out):
        """
        {Description}
    
        Args:
            graph_idx: 当前数据点的ID
            node_idx_map: 
            nn_out: 神经网络的输出结果
        Returns:
            out_dict: 结果字典，key是node_id， value是基数信息
            return2:
        """
        out_dict = {}
        self.set_current_idx(graph_idx)

        for idx, val in enumerate(nn_out):
            node_idx = node_idx_map[idx]
            origin_true_card = self.infer_true_card(graph_idx, node_idx, val)
            out_dict[node_idx] = origin_true_card

        return out_dict
    
    def restore_origin_distributions(self, graph_idx, node_idx_map, nn_out):
        """
        恢复到之前的分布
    
        Args:
            graph_idx: 当前数据点的ID
            node_idx_map: 
            nn_out: 神经网络的输出结果
        Returns:
            out_dict: 结果字典，key是node_id， value是基数信息
            return2:
        """
        out_dict = {}
        self.set_current_idx(graph_idx)

        for idx, vals in enumerate(nn_out):
            assert len(vals) == 2, f"restore_origin_distributions: len(vals) = {len(vals)}." 
            dist_mean, dist_std = vals
            node_idx = node_idx_map[idx]
            origin_est_card  = self.est_card_dict[graph_idx][node_idx]
            # 记录分布的相关信息，由三个参数组成
            out_dict[node_idx] = origin_est_card, dist_mean, dist_std

        return out_dict


    def convert_to_subquery_format(self, graph_idx, value_dict, op_func = None):
        """
        {Description}
        
        Args:
            graph_idx:
            value_dict:
        Returns:
            subquery_res:
            single_table_res:
        """
        local_dict = self.idx2key_dict[graph_idx]
        # print(f"convert_to_subquery_format: graph_idx = {graph_idx}. local_dict = {local_dict}.", flush=True)
        subquery_res, single_table_res = {}, {}

        for k, v in value_dict.items():
            assert isinstance(local_dict[k], tuple)
            # print(type(v))
            if op_func is not None:
                if isinstance(v, (float, np.float64)):
                    v = op_func(v)
                elif isinstance(v, (list, tuple, np.ndarray)):
                    v = [op_func(item) for item in v]
                else:
                    print(f"convert_to_subquery_format: v.shape = {v.shape}.")
                    raise TypeError(f"convert_to_subquery_format: type(v) = {type(v)}")

            if len(local_dict[k]) == 1:
                # 属于single_table的情况
                single_table_res[local_dict[k][0]] = v
            else:
                # 属于subquery的情况
                subquery_res[local_dict[k]] = v

        return subquery_res, single_table_res
    

    def encode_single_query(self, query_meta, estimation_card, \
            true_card, mask: bool = False, node_idx = None, mode = "both"):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            feature_vector:
            label:
        """
        try:
            schema_list, filter_list = query_meta
        except Exception as e:
            print(f"func_name: meet Error. .")
            raise e

        assert mode in ("both", "query-only", "card-only")
        query_repr_vector = self.table_encode(schema_list, self.table_pos_dict) + \
                            self.column_encode(filter_list, self.column_pos_dict)

        card_vector, label = self.cardinality_encode(estimation_card, true_card, mask)
        # print(f"encode_single_query: card_vector = {card_vector}")
        if node_idx is not None:
            self.est_card_dict[self.graph_idx][node_idx] = estimation_card

        if mode == "both":
            return query_repr_vector + card_vector, label
        elif mode == "query-only":
            return query_repr_vector, label
        else:
            return card_vector, label
        # return card_vector, label

    def encode_graph(self, attr_dict, mask_num = 1, edge_index = None) -> Data:
        """
        针对整个query_graph进行特征编码
        
        Args:
            attr_dict: 
            mask_num:
            edge_index: 图上edge的相关信息
        Returns:
            data_obj:
        """
        if mask_num is not None and mask_num >= 1:
            # 随机添加mask用于训练模型
            train_mask, test_mask, selected_idx = self.add_node_mask_randomly(attr_dict, mask_num)
        else:
            # 根据输入基数的缺失添加mask，用于测试
            train_mask, test_mask, selected_idx = self.add_node_mask_along_card(attr_dict)

        feature_vector_list = []
        label_list = []
        for idx in sorted(attr_dict.keys()):
            local_dict = attr_dict[idx]
            if idx not in selected_idx:
                # 节点结果已知的情况
                feature_vector, label = self.encode_single_query(
                    query_meta=local_dict['meta'], estimation_card=local_dict['est_card'], \
                    true_card=local_dict['true_card'], mask = False, node_idx=idx
                )
            else:
                # 节点结果未知的情况
                feature_vector, label = self.encode_single_query(
                    query_meta=local_dict['meta'], estimation_card=local_dict['est_card'], \
                    true_card=local_dict['true_card'], mask = True, node_idx=idx
                )
            # print(f"encode_graph: len(feature_vector) = {len(feature_vector)}")
            feature_vector_list.append(feature_vector)
            label_list.append(label)

        for node_idx in attr_dict.keys():
            try:
                self.idx2key_dict[self.graph_idx][node_idx] = attr_dict[node_idx]['alias_tuple']
            except KeyError as e:
                print(f"meet KeyError: {attr_dict[node_idx].keys()}.")
                raise e
        # local_data = Data(x = torch.Tensor(feature_vector_list, \
        #             dtype=torch.float32), edge_index = edge_index)
        # print(f"encode_graph: label_list = {label_list}")


        local_data = Data(x = torch.Tensor(feature_vector_list), y = torch.Tensor(label_list))
        local_data.train_mask = train_mask
        local_data.test_mask = test_mask

        # 加入edge_label
        if edge_index is not None:
            # print(f"encode_graph: edge_index = {edge_index}.")
            local_data.edge_index = edge_index
            local_data.edge_label = local_data.y[local_data.edge_index[0]] - \
                local_data.y[local_data.edge_index[1]]
            
            # 

            local_data.edge_train_mask = torch.isin(local_data.edge_index[1], torch.tensor(selected_idx))
            local_data.edge_test_mask = torch.isin(local_data.edge_index[1], torch.tensor(selected_idx))
        # else:
        #     print(f"encode_graph: edge_index = {edge_index}.")

        return local_data
    
    def add_node_mask_along_card(self, attr_dict):
        """
        {Description}
        
        Args:
            attr_dict:
            arg2:
        Returns:
            train_mask: 
            test_mask: 
            selected_idx:
        """
        num_nodes = len(attr_dict)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)   # 不作为训练数据
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        selected_idx = []
        # mask_cnt = 0
        # 获取相关的alias_tuple以及对应的index
        for idx, v in attr_dict.items():
            # print(f"add_node_mask_along_card: idx = {idx}. v = {v}")
            if v['true_card'] is None:
                selected_idx.append(idx)
        
        assert len(selected_idx) > 0
        # train_mask[selected_idx] = True
        test_mask[selected_idx] = True
        # print(f"add_node_mask_along_card: selected_idx = {selected_idx}")
        return train_mask, test_mask, selected_idx
    
    def add_node_mask_randomly(self, attr_dict, num = 1):
        """
        {Description}
    
        Args:
            attr_dict: 属性字典
            num: 添加mask的数目
        Returns:
            train_mask: 
            test_mask: 
            selected_idx:
        """
        num_nodes = len(attr_dict)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        selected_idx = []
        candidate_dict = defaultdict(set)

        idx_tuple_list = []
        for k, v in attr_dict.items():
            idx_tuple_list.append((k, v['alias_tuple']))

        idx_tuple_list.sort(key=lambda a: len(a[1]), reverse=True)
        max_idx, max_alias_tuple = idx_tuple_list[0]

        selected_alias = np.random.choice(max_alias_tuple)   # 选择的表别名

        # 获取相关的alias_tuple以及对应的index
        for idx, v in attr_dict.items():
            if selected_alias in v['alias_tuple']:
                candidate_dict[len(v['alias_tuple'])].add(idx)

        # 根据target_num由上到下选择idx
        left_num = num
        for alias_num in sorted(candidate_dict.keys(), reverse=True):
            if left_num <= len(candidate_dict[alias_num]):
                # print(f"add_node_mask: alias_num = {alias_num}. candidate_dict[alias_num] = {candidate_dict[alias_num]}.")
                selected_local = np.random.choice(list(candidate_dict[alias_num]), left_num)
                selected_idx.extend(selected_local)
                break
            else:
                selected_local = candidate_dict[alias_num]
                left_num -= selected_local
                selected_idx.extend(selected_local)

        # selected_idx = [max_idx, ]

        train_mask[selected_idx] = True
        test_mask[selected_idx] = True

        return train_mask, test_mask, selected_idx
    

# %%

class DataAugumenter(object):
    """
    数据增强的实例类，用以获得更高质量的训练数据

    Members:
        field1:
        field2:
    """

    def __init__(self, ):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        self.attr_dict, self.graph_dict = {}, {}
        self.graph_global = None


    def load_instance(self, attr_dict, graph_dict):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        graph_global = nx.Graph()
        for start_idx, end_idx in graph_dict['edge_set']:
            graph_global.add_edge(start_idx, end_idx)

        self.graph_global = graph_global
        self.attr_dict, self.graph_dict = \
            attr_dict, graph_dict

    def get_subquery_nodes(self, top_repr):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        sub_idx_list = []

        for idx, attr_local in self.attr_dict.items():
            if set(attr_local['alias_tuple']).issubset(set(top_repr)) == True:
                sub_idx_list.append(idx)

        return sub_idx_list


    def graph2dict(self, node_idx_list: list, in_graph: nx.Graph):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        node_idx_list.sort()    # subgraph idx进行排序
        mapping_local = {}
        local_attr_dict, local_graph_dict = {}, {"edge_set": set(), "node_dict": defaultdict(list)}

        for local_idx, node_idx in enumerate(node_idx_list):
            mapping_local[node_idx] = local_idx
            local_attr_dict[local_idx] = deepcopy(self.attr_dict[node_idx])     # 复制整一个节点属性

        for start_idx, end_idx in in_graph.edges():
            start_local, end_local = mapping_local[start_idx], mapping_local[end_idx]

            local_graph_dict['edge_set'].add((start_local, end_local))
            local_graph_dict['node_dict'][start_local].append(end_local)

        return local_attr_dict, local_graph_dict


    def augument_by_masking(self, masking_num, out_num):
        """
        通过masking的机制来进行数据的增强
        
        Args:
            masking_num: 
            out_num:
        Returns:
            res1:
            res2:
        """
        graph_current, attr_current = deepcopy(self.graph_dict), deepcopy(self.attr_dict)
        instance_list = []

        return instance_list
    

    def augument_by_subgraph(self, clip_level = 1):
        """
        对于当前的图数据进行增强
    
        Args:
            clip_level:
            arg2:
        Returns:
            instance_list:
            return2:
        """
        alias_tuple_list = [item['alias_tuple'] for item in self.attr_dict.values()]

        top_tuple = max(alias_tuple_list, key=lambda a: len(a))
        # top_candidates = [at for at in top_tuple if len(top_tuple) - len(at) <= \
        #                   clip_level and len(top_tuple) > len(at)]
        top_candidates = [at for at in alias_tuple_list if len(top_tuple) - len(at) <= \
                          clip_level and len(top_tuple) > len(at)]
        
        # print(f"augument_by_subgraph: top_candidates = {top_candidates}")
        instance_list = []
        for candidate in top_candidates:
            sub_idx_list = self.get_subquery_nodes(candidate)
            sub_graph = self.graph_global.subgraph(nodes=sub_idx_list)

            local_attr_dict, local_graph_dict = self.graph2dict(sub_idx_list, sub_graph)
            instance_list.append((local_attr_dict, local_graph_dict))
        return instance_list
    
    def node_location(self, alias_tuple_list, mode = "left-align"):
        """
        确定每个node在图上的具体位置
        
        Args:
            alias_tuple_list:
            mode: [left-align|centering]
        Returns:
            pos_dict:
        """
        level_at_dict = defaultdict(list)
        pos_dict = {}
        for alias_tuple in alias_tuple_list:
            level = len(alias_tuple)
            level_at_dict[level].append(alias_tuple)

        if mode == "left-align":
            for level, local_at_list in level_at_dict.items():
                for idx, at in enumerate(local_at_list):
                    pos_dict[at] = (idx, level)
        elif mode == "centering":
            max_width = max()

        return pos_dict
    
    def graph_visualize(self,):
        """
        图结果的可视化
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        def tuple_label(in_tuple, max_per_line = 3):
            out_list = []
            for idx, item in enumerate(in_tuple):
                if idx % max_per_line == 0 and idx > 0:
                    out_list.append(f"\n{item}")
                else:
                    out_list.append(f"{item}")

            return ",".join(out_list)

        alias_tuple_list = [v['alias_tuple'] for v in self.attr_dict.values()]
        pos_dict = self.node_location(alias_tuple_list=alias_tuple_list)
        pos_input = {}

        for k, v in self.attr_dict.items():
            pos_input[k] = pos_dict[v['alias_tuple']]
        
        # nx.draw(self.graph_global, pos_input)
        attr_dict = self.attr_dict
        labels = {
            # k: v for k, v in [(idx, str(",".join(\
            #     attr_dict[idx]['alias_tuple']))) for idx in attr_dict.keys()]
            k: v for k, v in [(idx, tuple_label(attr_dict[idx]['alias_tuple'])) for idx in attr_dict.keys()]
        }

        print(f"graph_visualize: labels = {labels}")
        
        # nx.draw_networkx_labels(self.graph_global, pos=pos_input, labels=labels)
        style_dict = {
            "node_size": 600
        }
        nx.draw(self.graph_global, pos=pos_input, labels=labels, \
                with_labels = True, **style_dict)
        
        # nx.draw_networkx_labels(self.graph_global, labels=labels)
        plt.show()

# %%

