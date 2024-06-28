#!/usr/bin/env python
# -*- coding: utf-8 -*-


# %%
import requests
from base import BaseCaller
from utility.common_config import cal_timeout

workload_option = {
    "job": {
        "ip_address": "166.111.121.55",
        "port": 20012,
        "workload": "job"
    },
    "stats": {
        "ip_address": "166.111.121.55",
        "port": 20012,
        "workload": "stats",
    },
    "release": {
        "ip_address": "172.6.31.12",
        "port": 30005,
        "workload": "release",
    }
}


def get_Oracle_caller_instance(workload = "job"):
    """
    {Description}
    
    Args:
        workload:
    Returns:
        res1:
        res2:
    """
    return OracleCaller(**workload_option[workload])


class OracleCaller(BaseCaller):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, ip_address = "101.6.96.160", port = "30005", workload = "job"):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        self.workload = workload
        self.ip_address = ip_address
        self.port = port

    def initialization(self,):
        """
        尝试连接Oracle数据库服务器，失败的话直接报错
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        params_dict = {
            "workload": self.workload,
            "method": "Oracle"
        }
        res = requests.get(url="http://{}:{}/start".format(self.ip_address, \
                            self.port), params=params_dict, timeout=5)
        return res.text


    def get_estimation_in_batch(self, query_list, label_list):
        """
        {Description}

        Args:
            query_list:
            label_list:
        Returns:
            card_list:
        """
        res = requests.post(url="http://{}:{}/cardinality_batch?method=Oracle".format(self.ip_address, self.port), \
                json={"sql_list": query_list, "label_list": [1000 for _ in query_list], "workload": self.workload}, \
                timeout=cal_timeout(len(query_list)))
        # print("res1 = {}.".format(res1))
        # res_dict = res.json()
        # print(f"OracleCaller.get_estimation_in_batch: res_dict = {res.json()}")
        
        return res.json()['card_list']

    def get_estimation(self, sql_text, label):
        """
        {Description}

        Args:
            sql_text:
            label:
        Returns:
            card:
        """
        request_url = "http://{}:{}/cardinality?method=Oracle".format(self.ip_address, self.port)
        res = requests.post(url=request_url, json={"query": sql_text, "workload": self.workload}, timeout=20)

        return res.json()["cardinality"]
    
# %%
