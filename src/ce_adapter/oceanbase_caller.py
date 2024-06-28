#!/usr/bin/env python
# -*- coding: utf-8 -*-


# %%
import requests
from base import BaseCaller
from utility.common_config import cal_timeout
import time
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
    "dsb": {
        "ip_address": "166.111.121.55",
        "port": 20012,
        "workload": "dsb",
    }
}

def get_Oceanbase_caller_instance(workload = "job"):
    """
    {Description}
    
    Args:
        workload:
    Returns:
        res1:
        res2:
    """
    return OceanbaseCaller(**workload_option[workload])


class OceanbaseCaller(BaseCaller):
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
        尝试连接Oceanbase数据库服务器，失败的话直接报错
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        params_dict = {
            "workload": self.workload,
            "method": "Oceanbase"
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
        # res = requests.post(url="http://{}:{}/cardinality_batch?method=Oceanbase".format(self.ip_address, self.port), \
        #         json={"sql_list": query_list, "label_list": [1000 for _ in query_list], "workload": self.workload}, timeout=cal_timeout(len(query_list)))

        # 设置请求失败重传
        max_try_times, flag = 5, False

        url = "http://{}:{}/cardinality_advance?method=Oceanbase".format(self.ip_address, self.port)
        for idx in range(max_try_times):
            try:
                res = requests.post(url=url, json={"sql_list": query_list, \
                    "label_list": [1000 for _ in query_list], "workload": self.workload}, \
                    timeout=cal_timeout(len(query_list)))
                flag = True
                break
            except Exception as e:
                curr_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
                print(f"OceanbaseCaller.get_estimation_in_batch: network error. info = {e}. idx = {idx}. curr_time = {curr_time}.")
                time.sleep(3)
                continue

        if flag == False:
            curr_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
            err_str = f"OceanbaseCaller.get_estimation_in_batch: network error. url = {url}. "\
                f"len(sql_list) = {len(query_list)}. curr_time = {curr_time}."
            raise ValueError(err_str)
        
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
        request_url = "http://{}:{}/cardinality?method=Oceanbase".format(self.ip_address, self.port)
        res = requests.post(url=request_url, json={"query": sql_text, "workload": self.workload}, timeout=20)

        return res.json()["cardinality"]
    
# %%
