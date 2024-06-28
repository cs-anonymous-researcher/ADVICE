#!/usr/bin/env python
# -*- coding: utf-8 -*-

# %%
from matplotlib import pyplot as plt
from torch_geometric.data import data
import graphviz

# %%
def display_single_graph(graph_data):
    """
    {Description}

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    out_graph = graphviz.Digraph()
    # out_graph.node()
    # out_graph.node()
    # out_graph.edge()

    return out_graph

# %%

def visualize_all_labels(dataset):
    """
    将所有label可视化出来

    Args:
        dataset:
        arg2:
    Returns:
        label_list:
        return2:
    """
    label_list = []
    for data in dataset:
        data: data.Data = data
        label_list.extend(data.y[data.train_mask].tolist())

    plt.hist(label_list, bins=20)
    plt.savefig("hist.png", dpi=1000)
    return label_list

# %%
