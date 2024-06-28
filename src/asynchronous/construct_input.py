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

from data_interaction import postgres_connector, mv_management
from query import query_construction
from utility import utils, common_config
import socket
import psycopg2 as pg
import psutil
# %%

def info_adjust(in_dict: dict, key_mapping = {"db_name": "database"}):
    """
    {Description}

    Args:
        arg1:
        arg2:
    Returns:
        return1:
        return2:
    """
    out_dict = {}

    for k, v in in_dict.items():
        if k in key_mapping.keys():
            out_dict[key_mapping[k]] = v
        else:
            out_dict[k] = v

    return out_dict

def conn_dict(workload):
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    return info_adjust(postgres_connector.workload_option[workload])

# %%

def get_query_signature(query_meta):
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    return mv_management.condition_encoding(query_meta[0], query_meta[1])


def get_card_hint_query(query_text, card_dict = None):
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    pass


def get_output_path(query_signature):
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    # return "/home/lianyuan/Research/CE_Evaluator/intermediate/async/{}.txt".format(query_signature)
    async_dir = "/home/lianyuan/Research/CE_Evaluator/intermediate/async"
    return p_join(async_dir, f"{common_config.async_prefix}/{query_signature}.txt")

def get_proc_fpath(query_signature):
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    # return "/home/lianyuan/Research/CE_Evaluator/intermediate/async/{}_proc.json".format(query_signature)
    async_dir = "/home/lianyuan/Research/CE_Evaluator/intermediate/async"
    return p_join(async_dir, f"{common_config.async_prefix}/{query_signature}_proc.json")

def output_path_from_meta(query_meta):
    return get_output_path(get_query_signature(query_meta=query_meta))

def get_batch_connections(workload, conn_num):
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    conn_list = []
    pid_list = []

    for _ in range(conn_num):
        conn_info = info_adjust(postgres_connector.workload_option[workload])
        db_conn = pg.connect(**conn_info)
        local_pid = get_conn_pid(db_conn=db_conn)
        conn_list.append(db_conn)
        pid_list.append(local_pid)

    return conn_list, pid_list


def get_conn_pid(db_conn):
    """
    获得当前进程的pid
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    # with db_conn.cursor() as cursor:
    #     cursor.execute("select pg_backend_pid();")
    #     process_id = cursor.fetchall()[0][0]
    process_id = db_conn.get_backend_pid()
    return process_id

import multiprocessing
file_lock = multiprocessing.Lock()  # 结果文件的锁
proc_lock = multiprocessing.Lock()  # proc映射的锁

workload_conn = {
    "stats": {
        "user": "lianyuan",
        "host": "localhost",
        "password": "",
        "database": "stats", 
        "port": "6432"
    },
    "job": {
        "user": "lianyuan",
        "host": "localhost",
        "password": "",
        "database": "imdbload", 
        "port": "6432"
    },
    "release": {
        "user": "lianyuan",
        "host": "localhost",
        "password": "",
        "database": "release_eight_tables", 
        "port": "6432"
    },
    "dsb": {
        "user": "lianyuan",
        "host": "localhost",
        "password": "",
        # "database": "dsb_2g", 
        "database": "dsb_5g", 
        "port": "6432"
    }
}

def proc_multi_queries_under_timeout(workload: str, query_list: list, key_list: list, out_path: list, timeout = 600000):
    """
    进程的函数，用于处理多个query，对于
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    # 
    func_start_time = time.time()
    out_proc = out_path.replace(".txt", "_proc.json")
    db_conn = pg.connect(**workload_conn[workload], options = f"-c statement_timeout={timeout}")

    # 显示当前的时间
    curr_time_str = time.strftime("%H:%M:%S", time.localtime())
    # print("proc_function: len(query_list) = {}. out_path = {}. key_list = {}. curr_time = {}".\
    #       format(len(query_list), out_path, key_list, curr_time_str))
    pid = db_conn.get_backend_pid()

    if os.path.isfile(out_proc) == False:
        utils.dump_json({}, out_proc)   # 如果文件不存在的话，初始化为空字典

    for idx, (key, query_text) in enumerate(zip(key_list, query_list)):
        ts = time.time()
        try:
            with db_conn.cursor() as cursor:
                with proc_lock:
                    # print("proc_function: update proc. key = {}.".format(key))
                    proc_dict = utils.load_json(out_proc)
                    proc_dict[str(key)] = pid
                    utils.dump_json(proc_dict, out_proc)
                    # print("proc_function: update finish. key = {}.".format(key))
                cursor.execute(query_text)
                result = cursor.fetchall()
        except (pg.errors.QueryCanceled, pg.errors.InternalError_) as e1:
            # 查询超时，所有剩余的查询写入无效值(-1)
            te = time.time()
            with file_lock:
            # 不用json存了，用text file存提高效率
                with open(out_path, "a+") as f_out:
                    for (key, query_text) in zip(key_list[idx:], query_list[idx:]):
                        f_out.write("{}#{}#{}#{}\n".format(key, query_text, -1, te - ts))
            return
        except pg.OperationalError as e2:
            # 处理查询
            # print("proc_multi_queries: error first! pid = {}. idx = {}. total_len = {}. e = {}. query_text = {}.".
            #       format(pid, idx, len(key_list), str(e2).replace("\n", " "), query_text))

            te = time.time()
            with file_lock:
            # 不用json存了，用text file存提高效率
                with open(out_path, "a+") as f_out:
                    for (key, query_text) in zip(key_list[idx:], query_list[idx:]):
                        f_out.write("{}#{}#{}#{}\n".format(key, query_text, -1, te - ts))
            return


        te = time.time()

        with file_lock:
            if timeout is not None and os.path.isfile(out_path) == True:
                f_content = ""
                with open(out_path, "r") as f_in:
                    f_content = f_in.read()
                if "#-1#" in f_content:
                    # 已经由其他进程写入错误的结果，直接退出
                    print("proc_multi_queries_under_card_dict: detect error in other processes. exit directly")
                    with open(out_path, "a+") as f_out:
                        for (key, query_text) in zip(key_list[idx:], query_list[idx:]):
                            f_out.write("{}#{}#{}#{}\n".format(key, query_text, -1, te - ts))
                    return

            # 不用json存了，用text file存提高效率
            with open(out_path, "a+") as f_out:
                f_out.write("{}#{}#{}#{}\n".format(key, query_text, result[0][0], te - ts))

    if pid != -1:
        # 记录当前pid对应的cpu_time
        proc = psutil.Process(pid)
        cpu_time = proc.cpu_times()
        cpu_total = cpu_time.user + cpu_time.system

        with proc_lock:
            proc_dict = utils.load_json(out_proc)
            if 'proc_time' not in proc_dict.keys():
                proc_dict['proc_time'] = {}
            proc_dict['proc_time'][pid] = cpu_total
            utils.dump_json(proc_dict, out_proc)
            # print("pid = {}. cpu_total = {}.".format(pid, cpu_total))
    else:
        pass
    
    # 关闭数据库连接
    db_conn.close()
    func_end_time = time.time()

    # 打印函数执行的信息
    with proc_lock:
        # utils.trace("proc_multi_queries: out_path = {}. start_time = {}. end_time = {}.".\
        #             format(out_path, func_start_time, func_end_time))
        with open("./temp.txt", "a+") as f:
            f.write("trace: proc_multi_queries: out_path = {}. start_time = {}. end_time = {}.\n".\
                     format(out_path, func_start_time, func_end_time))
        

def proc_multi_queries(workload: str, query_list, key_list, out_path):
    """
    一个进程的函数，用于处理多个query
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    # 
    func_start_time = time.time()

    out_proc = out_path.replace(".txt", "_proc.json")
    # print("proc_multi_queries: out_path = {}. out_proc = {}.".format(out_path, out_proc))

    db_conn = pg.connect(**workload_conn[workload])

    # 显示当前的时间
    curr_time_str = time.strftime("%H:%M:%S", time.localtime())
    # print("proc_function: len(query_list) = {}. out_path = {}. key_list = {}. curr_time = {}".\
    #       format(len(query_list), out_path, key_list, curr_time_str))
    pid = db_conn.get_backend_pid()

    if os.path.isfile(out_proc) == False:
        utils.dump_json({}, out_proc)   # 如果文件不存在的话，初始化为空字典


    for idx, (key, query_text) in enumerate(zip(key_list, query_list)):
        ts = time.time()
        try:
            with db_conn.cursor() as cursor:
                with proc_lock:
                    # print("proc_function: update proc. key = {}.".format(key))
                    proc_dict = utils.load_json(out_proc)
                    proc_dict[str(key)] = pid
                    utils.dump_json(proc_dict, out_proc)
                    # print("proc_function: update finish. key = {}.".format(key))
                cursor.execute(query_text)
                result = cursor.fetchall()
        except pg.OperationalError as e1:
            # print("proc_multi_queries: error first! pid = {}. idx = {}. total_len = {}. e = {}. query_text = {}.".
            #       format(pid, idx, len(key_list), str(e1).replace("\n", " "), query_text))
            return
            # # 重新构造DB连接
            # time.sleep(1)
            # db_conn = pg.connect(**workload_conn[workload])
            # pid = db_conn.get_backend_pid()     # 修改pid的信息
            # try:

            #     with db_conn.cursor() as cursor:
            #         with proc_lock:
            #             # print("proc_function: update proc. key = {}.".format(key))
            #             proc_dict = utils.load_json(out_proc)
            #             proc_dict[str(key)] = pid
            #             utils.dump_json(proc_dict, out_proc)
            #             # print("proc_function: update finish. key = {}.".format(key))

            #         cursor.execute(query_text)
            #         result = cursor.fetchall()
            # except pg.OperationalError as e2:
            #     print("proc_multi_queries: error again! pid = {}. idx = {}. total_len = {}. e = {}. query_text = {}.".
            #           format(pid, idx, len(key_list), e2, query_text))
                
            #     # print("proc_multi_queries: error again! e = {}. query_text = {}.".format(e2, query_text))
            #     return

        te = time.time()

        with file_lock:
            # 不用json存了，用text file存提高效率
            with open(out_path, "a+") as f_out:
                f_out.write("{}#{}#{}#{}\n".format(key, query_text, result[0][0], te - ts))

    if pid != -1:
        # 记录当前pid对应的cpu_time
        proc = psutil.Process(pid)
        cpu_time = proc.cpu_times()
        cpu_total = cpu_time.user + cpu_time.system

        with proc_lock:
            proc_dict = utils.load_json(out_proc)
            if 'proc_time' not in proc_dict.keys():
                proc_dict['proc_time'] = {}
            proc_dict['proc_time'][pid] = cpu_total
            utils.dump_json(proc_dict, out_proc)
            # print("pid = {}. cpu_total = {}.".format(pid, cpu_total))
    else:
        pass
    
    # 关闭数据库连接
    db_conn.close()
    func_end_time = time.time()

    # print(f"proc_multi_queries: process finish. pid = {pid}. out_path = {out_path}.")

    # proc_signature = os.path.basename(out_path).replace(".txt", "")
    # 打印函数执行的信息
    with proc_lock:
        # utils.trace("proc_multi_queries: out_path = {}. start_time = {}. end_time = {}.".\
        #             format(out_path, func_start_time, func_end_time))
        with open("./temp.txt", "a+") as f:
            f.write("trace: proc_multi_queries: out_path = {}. start_time = {}. end_time = {}.\n".\
                     format(out_path, func_start_time, func_end_time))
        

def proc_multi_queries_under_card_dict(workload: str, query_list, key_list, card_dict_list, out_path, timeout = None):
    """
    {Description}

    Args:
        workload:
        query_list:
        key_list:
        card_dict_list:
        out_path:
        timeout:
    Returns:
        return1:
        return2:
    """
    # 
    func_start_time = time.time()

    out_proc = out_path.replace(".txt", "_proc.json")
    if timeout is None:
        db_conn = pg.connect(**workload_conn[workload])
    else:
        db_conn = pg.connect(**workload_conn[workload], options = f"-c statement_timeout={timeout}")


    # 显示当前的时间
    curr_time_str = time.strftime("%H:%M:%S", time.localtime())

    hint_query_list = []
    alias_mapping = query_construction.abbr_option[workload]
    hint_adder = postgres_connector.HintAdder(\
        alias2table=utils.dict_reverse(alias_mapping))

    for query, card_dict in zip(query_list, card_dict_list):
        try:
            subquery_dict, single_table_dict = \
                card_dict['subquery'], card_dict['single_table']
            rows_list = [(k, v) for k, v in subquery_dict.items()]
            schema_list = [(k, v) for k, v in single_table_dict.items()]

            hint_str = hint_adder.generate_cardinalities_hint_str(    
                rows_list = rows_list,
                schema_list = schema_list)
            hint_query_text = f"{hint_str}\n{query}"
            # print(f"proc_multi_queries_under_card_dict: hint_query_text = {hint_query_text}.")
        except KeyError as e:
            # print(f"proc_multi_queries_under_card_dict: meet KeyError. card_dict = {card_dict}.")
            # raise e
            hint_query_text = query
            
        hint_query_list.append(hint_query_text)

    pid = db_conn.get_backend_pid()
    origin_query_list = query_list

    if os.path.isfile(out_proc) == False:
        utils.dump_json({}, out_proc)   # 如果文件不存在的话，初始化为空字典

    for idx, (key, query_origin, query_text) in enumerate(zip(key_list, origin_query_list, hint_query_list)):
        ts = time.time()
        try:
            with db_conn.cursor() as cursor:
                with proc_lock:
                    # print("proc_function: update proc. key = {}.".format(key))
                    proc_dict = utils.load_json(out_proc)
                    proc_dict[str(key)] = pid
                    utils.dump_json(proc_dict, out_proc)
                    # print("proc_function: update finish. key = {}.".format(key))
                cursor.execute(query_text)
                result = cursor.fetchall()
        # except pg.OperationalError as e1:
        #     print("proc_multi_queries: error first! pid = {}. idx = {}. total_len = {}. e = {}. query_text = {}.".
        #           format(pid, idx, len(key_list), str(e1).replace("\n", " "), query_origin))
        #     return
        except (pg.errors.QueryCanceled, pg.errors.InternalError_) as e1:
            # 查询超时，所有剩余的查询写入无效值(-1)
            te = time.time()
            with file_lock:
            # 不用json存了，用text file存提高效率
                with open(out_path, "a+") as f_out:
                    for (key, query_text) in zip(key_list[idx:], query_list[idx:]):
                        f_out.write("{}#{}#{}#{}\n".format(key, query_text, -1, te - ts))
            return
        except pg.OperationalError as e2:
            # 处理查询
            # print("proc_multi_queries: error first! pid = {}. idx = {}. total_len = {}. e = {}. query_text = {}.".
            #       format(pid, idx, len(key_list), str(e2).replace("\n", " "), query_text))

            te = time.time()
            with file_lock:
            # 不用json存了，用text file存提高效率
                with open(out_path, "a+") as f_out:
                    for (key, query_text) in zip(key_list[idx:], query_list[idx:]):
                        f_out.write("{}#{}#{}#{}\n".format(key, query_text, -1, te - ts))
            return

        te = time.time()

        # 正常写入结果
        with file_lock:
            if timeout is not None and os.path.isfile(out_path) == True:
                f_content = ""
                with open(out_path, "r") as f_in:
                    f_content = f_in.read()
                if "#-1#" in f_content:
                    # 已经由其他进程写入错误的结果，直接退出
                    print("proc_multi_queries_under_card_dict: detect error in other processes. exit directly")
                    with open(out_path, "a+") as f_out:
                        for (key, query_text) in zip(key_list[idx:], query_list[idx:]):
                            f_out.write("{}#{}#{}#{}\n".format(key, query_text, -1, te - ts))
                    return
                
            # 不用json存了，用text file存提高效率
            with open(out_path, "a+") as f_out:
                f_out.write("{}#{}#{}#{}\n".format(key, query_origin, result[0][0], te - ts))

    if pid != -1:
        # 记录当前pid对应的cpu_time
        proc = psutil.Process(pid)
        cpu_time = proc.cpu_times()
        cpu_total = cpu_time.user + cpu_time.system

        with proc_lock:
            proc_dict = utils.load_json(out_proc)
            if 'proc_time' not in proc_dict.keys():
                proc_dict['proc_time'] = {}
            proc_dict['proc_time'][pid] = cpu_total
            utils.dump_json(proc_dict, out_proc)
            # print("pid = {}. cpu_total = {}.".format(pid, cpu_total))
    else:
        pass
    
    # 关闭数据库连接
    db_conn.close()
    func_end_time = time.time()

    # print(f"proc_multi_queries: process finish. pid = {pid}. out_path = {out_path}.")
    # 打印函数执行的信息
    with proc_lock:
        # utils.trace("proc_multi_queries: out_path = {}. start_time = {}. end_time = {}.".\
        #             format(out_path, func_start_time, func_end_time))
        with open("./temp.txt", "a+") as f:
            f.write("trace: proc_multi_queries: out_path = {}. start_time = {}. end_time = {}.\n".\
                     format(out_path, func_start_time, func_end_time))


def proc_function(workload: str, query_text: str, key: str, out_path: str):
    """
    一个进程的函数
    
    Args:
        arg1:
        arg2:
    Returns:
        res1:
        res2:
    """
    # 
    out_proc = out_path.replace(".txt", "_proc.json")

    db_conn = pg.connect(**workload_conn[workload])
    # print("proc_function: db_conn = {}.".format(db_conn.closed))
    # time.sleep(np.random.rand())
    ts = time.time()
    # 随机睡眠一段时间，防止一起向DB发请求

    # while True:
    try:
        with db_conn.cursor() as cursor:
            pid = db_conn.get_backend_pid()
            with proc_lock:
                # print("proc_function: update proc. key = {}.".format(key))
                proc_dict = utils.load_json(out_proc)
                proc_dict[str(key)] = pid
                utils.dump_json(proc_dict, out_proc)
                # print("proc_function: update finish. key = {}.".format(key))

            cursor.execute(query_text)
            result = cursor.fetchall()
    except pg.OperationalError:
        print("error! proc_function: query_text = {}.".format(query_text))
        return

    te = time.time()

    with file_lock:
        # 不用json存了，用text file存提高效率
        with open(out_path, "a+") as f_out:
            f_out.write("{}#{}#{}#{}\n".format(key, query_text, result[0][0], te - ts))

        # res_dict = {
        #     "query": query_text,
        #     "cardinality": result[0][0],
        #     "execution_time": te - ts
        # }
        # utils.dump_json(res_dict, out_path)

# %%
def query_cost_estimation(workload, query_list, card_dict_list):
    """
    具有部分card_dict下查询执行代价估计
    
    Args:
        arg1:
        arg2:
    Returns:
        hint_query_list: 
        cost_list:
    """
    card_hint_adder = postgres_connector.HintAdder({})
    hint_template = """/*+ {COMMENT} */ EXPLAIN (FORMAT JSON) {QUERY}"""

    hint_query_list, cost_list = [], []
    db_conn = pg.connect(**conn_dict(workload=workload))

    with db_conn.cursor() as cursor:
        for query, card_dict in zip(query_list, card_dict_list):
            try:
                # print("query_cost_estimation: card_dict = {}.".format(card_dict))
                subquery_list = list(card_dict["subquery"].items())
                single_table_list = list(card_dict["single_table"].items())
                hinted_str = card_hint_adder.generate_cardinalities_hint_str(\
                    rows_list=subquery_list, schema_list=single_table_list, with_wrapper=False)
                
                hint_query = hint_template.format(COMMENT=hinted_str, QUERY=query)
                hint_query_list.append(hint_query)

                cursor.execute(hint_query)
                result = cursor.fetchall()
                cost_list.append(result[0][0][0]['Plan']['Total Cost'])
            except KeyError:
                # 没有hint的情况
                hint_query = "EXPLAIN (FORMAT JSON) " + query
                hint_query_list.append(hint_query)
                cursor.execute(hint_query)
                result = cursor.fetchall()
                cost_list.append(result[0][0][0]['Plan']['Total Cost'])
                
    return hint_query_list, cost_list


def multi_query_scheduling(query_list, cost_list, proc_num, debug = True):
    """
    {Description}
    
    Args:
        arg1:
        arg2:
    Returns:
        proc_index_list: 
        proc_query_list: 
        proc_cost_list:
    """
    # print("cost_list = {}.".format(cost_list))
    sorted_idx = np.argsort(cost_list)[::-1]

    proc_index_list = [[] for _ in range(proc_num)]
    proc_query_list = [[] for _ in range(proc_num)]
    proc_cost_list = [0.0 for _ in range(proc_num)]

    def find_proc_idx():
        return np.argmin(proc_cost_list)

    # print("proc_cost_list = {}.".format(proc_cost_list))

    for idx in sorted_idx:
        idx = int(idx)
        slot = find_proc_idx()
        # print("multi_query_scheduling: idx = {}. slot = {}.".format(idx, slot))
        proc_index_list[slot].append(idx)
        proc_query_list[slot].append(query_list[idx])
        proc_cost_list[slot] += cost_list[idx]

    # print("proc_index_list = {}.".format(proc_index_list))
    # print("proc_cost_list = {}.".format(proc_cost_list))

    # 对于结果进行反转，保证cost小的可以先被执行，短时间尽可能执行多的query
    for i in range(proc_num):
        proc_index_list[i].reverse()
        proc_query_list[i].reverse()

    return proc_index_list, proc_query_list, proc_cost_list

# %%
