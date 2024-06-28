#!/usr/bin/env python
# -*- coding: utf-8 -*-

# %%

import flask
from interaction import state_management
from flask import Flask, request, abort
import time
from utilities import utils
import multiprocessing as mp

app = Flask(__name__)

# %%
global_manager: state_management.StateManager = None

# %%

# 允许访问的IP地址
ip_allow_list = [
    "127.0.0.1"          # localhost
]

@app.before_request
def block_method():
    ip = request.environ.get("REMOTE_ADDR")
    if ip not in ip_allow_list:
        abort(403)


# %%
def process_input_params(query_meta, card_dict):
    """
    {Description}
    
    Args:
        query_meta:
        card_dict:
    Returns:
        _out-meta:
        card_out:
    """
    out_meta = eval(query_meta)

    subquery_true, single_table_true, subquery_est, \
        single_table_est = utils.extract_card_info(card_dict)
    
    subquery_true = utils.dict_apply(subquery_true, eval, mode="key")
    subquery_est = utils.dict_apply(subquery_est, eval, mode="key")

    out_card_dict = utils.pack_card_info(subquery_true, \
        single_table_true, subquery_est, single_table_est)
    
    return out_meta, out_card_dict

def construct_output_params(subquery_out, single_table_out):
    """
    {Description}

    Args:
        subquery_out:
        single_table_out:
    Returns:
        return1:
        return2:
    """
    subquery_new = utils.dict_apply(subquery_out, str, mode="key")
    single_table_new = utils.dict_apply(single_table_out, str, mode="key")

    return subquery_new, single_table_new



@app.route("/start_new_task", methods = ["POST"])
def start_new_task():
    """
    开始新的任务

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    global global_manager
    global pair_manager

    config_dict = request.args
    method = config_dict['method']
    workload = config_dict['workload']
    model_type = config_dict['model_type']

    signature = time.strftime("%Y%m%d%H%M%S", time.localtime())
    # 创建global_manager
    if global_manager is None:
        global_manager = state_management.StateManager(\
            workload, method, signature, model_type, auto_update = True)

    if model_type == "dropout_bayes" or model_type == "deep_ensemble":
        global_manager.set_sample_num(int(config_dict['sample_num']))

    reply_dict = {
        "signature":signature    
    }

    return reply_dict


@app.route("/finish_task", methods = ["POST"])
def finish_task():
    """
    结束任务

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    global global_manager
    global_manager = None

    reply_dict = {
        "status": "success"    
    }
    return reply_dict


@app.route("/save_instance", methods = ["POST"])
def save_instance():
    """
    保存实例
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    config_dict = request.args
    data_dict = request.json

    query_meta = data_dict['query_meta']
    card_dict = data_dict['card_dict']

    # 处理还原query_meta
    query_meta, card_dict = process_input_params(query_meta, card_dict)

    global_manager.add_train_data([query_meta,], [card_dict,])
    reply_dict = {
        "status": "success"
    }
    return reply_dict


@app.route("/cardinality_inference", methods = ["POST"])
def cardinality_inference():
    """
    基数推断
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    data_dict = request.json

    query_meta = data_dict['query_meta']    # 查询元信息
    card_dict = data_dict['card_dict']      # 基数字典
    # 
    query_meta, card_dict = process_input_params(query_meta, card_dict)

    # 推断结果
    flag, subquery_res, single_table_res = \
        global_manager.infer_on_instance(query_meta, card_dict)
    
    print(f"cardinality_inference: subquery_res = {subquery_res}. "\
          f"single_table_res = {single_table_res}.")
    # 
    subquery_res, single_table_res = construct_output_params(subquery_res, single_table_res)
    reply_dict = {
        "flag": flag,
        "result": (subquery_res, single_table_res)
    }
    return reply_dict


if __name__ == "__main__":
    # try:
    #     torch.multiprocessing.set_start_method('spawn', force=True)
    # except RuntimeError:
    #     pass
    try:
        # app.run(host='0.0.0.0', port=30007, debug=True)     # 设置监听的端口
        app.run(host='0.0.0.0', port=30007)     # 设置监听的端口
    except Exception as e:
        print(f"server.main: meet Error. e = {e}.")
        import sys
        sys.exit(1)
# %%
