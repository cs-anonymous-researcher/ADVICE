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
from baseline.utility import base
import socket
import requests
from plan import node_query, node_extension
from query import query_construction
from grid_manipulation import grid_preprocess
from data_interaction import mv_management

# %%

class RLBasedPlanSearcher(base.BasePlanSearcher):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, schema_total, workload, table_num_limit, time_limit, ce_type):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        print("RLBasedPlanSearcher.__init__: workload = {}.".format(workload))
        self.workload = workload
        self.table_num_limit = table_num_limit
        self.schema_total = schema_total

        self.current_query_instance = None       # 当前的查询对象
        self.current_query_meta = None           # 当前的元信息
        
        # 历史结果的保存
        self.historical_query_instance = []
        self.historical_query_meta = []
        self.time_list = []
        super(RLBasedPlanSearcher, self).__init__(workload=workload, ce_type=ce_type, time_limit=time_limit)

    def get_column_value_bounds(self, table_name, column_name):
        """
        {Description}
        
        Args:
            table_name:
            column_name:
        Returns:
            lower_bound:
            upper_bound:
        """
        lower_bound = int(self.data_manager.load_table(tbl_name=table_name)[column_name].min())
        upper_bound = int(self.data_manager.load_table(tbl_name=table_name)[column_name].max())

        return lower_bound, upper_bound
    
    def meta_match_evaluation(self, in_meta):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        return mv_management.meta_comparison(in_meta, self.current_query_meta)

    def launch_search_process(self, total_time, with_start_time=False):
        """
        {Description}

        Args:
            total_time:
        Returns:
            query_list: 
            meta_list: 
            result_list: 
            card_dict_list:
        """
        start_time = time.time()

        # 搜索的配置
        search_config = {
            "workload": self.workload,
            "table_num_limit": self.table_num_limit,
            "total_time": total_time,
            "schema_total": self.schema_total
        }
        ip_address = "localhost"
        port = "30007"
        reply = requests.post(url="http://{}:{}/launch_experiment".format(ip_address, port), json=search_config, timeout=60)
        # reply = requests.post(url="http://{}:{}/launch_pesudo".format(ip_address, port), json=search_config)    # 先进行流程的测试
        
        # 建立socket，接收GPU Server发出的请求，json为格式
        server_socket = socket.socket()
        server_address = ('0.0.0.0', 30006)
        server_socket.bind(server_address)

        # 监听连接
        server_socket.listen(1)

        # client_desired = "127.0.0.1"
        client_desired = ["101.6.96.160", "127.0.0.1", "0.0.0.0", "localhost"]
        print("Launch socket_server. Begin to accept request")

        while True:
            connection, client_address = server_socket.accept()
            print("client address = {}.".format(client_address))
            # if client_address[0] == client_desired:
            if client_address[0] in client_desired:
                print(f"launch_search_process: meet unexpected address. client_address = {client_address[0]}.")
                break
            else:
                print("地址出错，继续尝试连接")
                # server_socket.close()
                server_socket.listen(1)

        try:
            while True:
                data_str = connection.recv(8192)
                print("data_str = {}.".format(data_str))
                try:
                    datagram = json.loads(data_str)
                except json.decoder.JSONDecodeError:
                    break

                cmd_option = datagram['option']
                print("launch_search_process: receive datagram. command option is {}.".format(cmd_option))

                reply_dict = {}
                if cmd_option == "initialize":
                    # 初始化
                    self.init_query()
                    cost1, cost2 = 0, 0
                    print("launch_search_process: initialize explore query")
                elif cmd_option == "add_table":
                    # 添加新的table
                    table_name = datagram["table"]

                    # 修改schema_list
                    self.current_query_meta[0].append(table_name)
                    cost1, cost2 = self.update_query_instance(self.current_query_meta)
                    print(f"launch_search_process: add table: {table_name}")
                elif cmd_option == "add_column":
                    # 添加新的column
                    table_name = datagram["table"]
                    column_name = datagram["column"]
                    table_alias = self.data_manager.tbl_abbr[table_name]

                    # 修改元信息
                    lower_bound, upper_bound = self.get_column_value_bounds(table_name, column_name)

                    # filter_list中添加新的项
                    self.current_query_meta[1].append((table_alias, column_name, lower_bound, upper_bound))
                    
                    # 修改QueryInstance
                    cost1, cost2 = self.update_query_instance(self.current_query_meta)
                    print(f"launch_search_process: add column: ({table_name}, {column_name})")

                elif cmd_option == "add_predicate":
                    table_name = datagram["table"]
                    table_alias = self.data_manager.tbl_abbr[table_name]

                    column_name = datagram["column"]
                    operator = datagram["operator"]
                    value = datagram["value"]
                    
                    for idx, item in enumerate(self.current_query_meta[1]):
                        if item[0] == table_alias and item[1] == column_name:
                            # 获得filter_list的匹配
                            if operator == "lower_bound":
                                item_new = (item[0], item[1], value, item[3])
                                print("add predicate: ({}.{} >= {})".format(table_alias, column_name, value))
                            elif operator == "upper_bound":
                                item_new = (item[0], item[1], item[2], value)
                                print("add predicate: ({}.{} <= {})".format(table_alias, column_name, value))

                            self.current_query_meta[1][idx] = item_new  # 完成替换
                            break
                    cost1, cost2 = self.update_query_instance(self.current_query_meta)
                elif cmd_option == "terminate":
                    # 终止单个查询的探索
                    meta_info = eval(datagram['meta'])
                    assert True == self.meta_match_evaluation(meta_info), \
                        f"launch_search_process: meta mismatch in terminate. meta1 = {meta_info}. meta2 = {self.current_query_meta}."
                    self.historical_query_instance.append(deepcopy(self.current_query_instance))
                    immutable_meta = tuple(self.current_query_meta[0]), tuple(self.current_query_meta[1])
                    self.historical_query_meta.append(immutable_meta)
                elif cmd_option == "ping":
                    reply_dict = {
                        "status": "success",
                        "info": "connection establish",
                    }
                    connection.send(json.dumps(reply_dict).encode())
                    continue
                elif cmd_option == "exit":
                    # 退出探索过程
                    print("整个搜索过程结束了")
                    reply_dict = {
                        "status": "success"
                    }
                    print("data_type = {}.".format(type(json.dumps(reply_dict).encode())))
                    connection.send(json.dumps(reply_dict).encode())
                    time.sleep(1)
                    break
                else:
                    raise ValueError("cmd_option = {}.".format(cmd_option))
                reply_dict = {
                    "status": "success",
                    "cost1": cost1,
                    "cost2": cost2
                }
                # 每次处理完消息都回复
                connection.send(json.dumps(reply_dict).encode())
        finally:
            connection.close()

        end_time = time.time()
        # 探索结束，关闭连接
        server_socket.close()   

        if with_start_time == True:
            time_info = self.time_list, start_time, end_time
            return self.query_list, self.meta_list, self.result_list, self.card_dict_list, time_info
        else:
            return self.query_list, self.meta_list, self.result_list, self.card_dict_list

    def get_search_result(self,):
        """
        获得搜索的结果

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        pass

    def init_query(self,):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        self.current_query_meta = ([], [])      # 空的meta信息
        # self.current_query_instance = node_query.get_query_instance(\
        #     workload=self.workload, query_meta=self.current_query_meta, )
        self.current_query_instance = None
        return self.current_query_meta, self.current_query_instance

    def update_query_instance(self, new_query_meta):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        if self.current_query_instance is None:
            # if len(new_query_meta[0]) > 1 or len(new_query_meta[1]) >= 1:
            if len(new_query_meta[0]) > 1:
                external_query = {
                    "data_manager": self.data_manager,
                    "mv_manager": self.mv_manager,
                    "ce_handler": self.ce_handler,
                    "query_ctrl": self.query_ctrl,
                    "multi_builder": self.multi_builder
                }
                self.current_query_instance: node_query.QueryInstance = node_query.get_query_instance(
                    workload=self.workload, query_meta=mv_management.meta_copy(new_query_meta), 
                    external_dict=external_query, ce_handler=self.ce_handler)
                return 0, 0
            else:
                # 单表无condition的query，不进行构造
                return 0, 0
        else:
            #
            self.time_list.append(time.time())
            cost1, cost2 = self.current_query_instance.modify_by_query_meta(new_query_meta, self.time_limit)

            # 添加历史结果
            query_text = self.current_query_instance.query_text
            query_meta = (tuple(new_query_meta[0]), tuple(new_query_meta[1]))
            result = cost2 / (cost1 + 2.0), cost2, cost1

            instance = self.current_query_instance
            card_dict = {
                "true": {
                    "subquery": instance.true_card_dict,
                    "single_table": instance.true_single_table
                },
                "estimation": {
                    "subquery": instance.estimation_card_dict,
                    "single_table": instance.estimation_single_table
                }
            }

            self.add_exploration_instance(query_text=query_text, \
                query_meta=query_meta, result=result, card_dict=deepcopy(card_dict))

            # 返回cost
            return cost1, cost2
