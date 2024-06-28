#!/usr/bin/env python
# -*- coding: utf-8 -*-


# %%
import argparse
from models import gcn_model, dropout_bayes_model, gat_model, deep_ensemble_model
from training import gnn_train, deep_ensemble_train, dropout_bayes_train
from data import query_dataset, dataloader
from utilities.global_config import data_split_config, hyper_params_dict
from utilities import utils, global_config

# %%


def create_model(model_type, workload = None):
    """
    {Description}

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    # hard-code设置超参数
    # if model_type == "gcn":
    #     GNN_model = gcn_model.GCNModel(workload=workload, hidden_channels_1=200, embedding_dims=50)
    # elif model_type == "gat":
    #     GNN_model = gat_model.GATModel(hidden_channels_1=200, embedding_dims=50)
    # elif model_type == "dropout_bayes":
    #     GNN_model = dropout_bayes_model.GNNModel(workload=workload, hidden_size=200, \
    #         embedding_dims=50, layer_type="gat", dropout_rate=0.2, decay=1e-6)
    # elif model_type == "deep_ensemble":
    #     GNN_model = deep_ensemble_model.create_model_list(100, 50, 10)
    # else:
    #     raise ValueError(f"launch_train_with_eval: Unsupported model({model_type})")
    
    if model_type == "gcn":
        GNN_model = gcn_model.GCNModel(workload=workload, **hyper_params_dict["GCN"])
    elif model_type == "gat":
        GNN_model = gat_model.GATModel(workload=workload, **hyper_params_dict["GAT"])
    elif model_type == "dropout_bayes":
        GNN_model = dropout_bayes_model.GNNModel(workload=workload, **hyper_params_dict["dropout_bayes"])
    elif model_type == "deep_ensemble":
        GNN_model = deep_ensemble_model.create_model_list(**hyper_params_dict["deep_ensemble"])
    else:
        raise ValueError(f"launch_train_with_eval: Unsupported model({model_type})")
    
    return GNN_model


def model_train(model_type, workload, signature, out_path):
    """
    {Description}

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    # root = "/home/jinly/GNN_Estimator/online_data/"
    root = global_config.online_data_dir

    file_content = f"{workload}_0.pkl"
    # 创建dataset
    train_dataset = query_dataset.QueryGraphDataset(root=root, \
            workload=workload, file_content=file_content, \
            signature=signature, clean_data=True)
    GNN_model = create_model(model_type, workload)

    train_loader, test_loader = dataloader.get_train_test_loader(train_dataset, data_split_config)

    if model_type in ("gcn", "gat"):
        gnn_train.model_train(GNN_model, train_loader, test_loader, out_path)
    elif model_type == "deep_ensemble":
        deep_ensemble_train.ensemble_model_train(GNN_model, train_loader, test_loader, out_path)
    elif model_type == "dropout_bayes":
        data_num = len(train_dataset) * data_split_config['train_ratio']
        print(f"model_train: dropout_bayes. data_num = {data_num}.")
        GNN_model.set_training_num(data_num)
        dropout_bayes_train.dropout_model_train(GNN_model, train_loader, test_loader, out_path)
    return None


# %%

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", "-M", default="gcn")
    parser.add_argument("--option", "-O", default="train")
    parser.add_argument("--signature", "-S")
    parser.add_argument("--workload", "-W")
    parser.add_argument("--out_path", "-P", default="")

    in_args = parser.parse_args()
    print(f"exection_unit: in_args = {in_args}.")
    option = in_args.option
    if option == "train":
        model_train(in_args.model_type, in_args.workload, \
                    in_args.signature, in_args.out_path)


# %%
