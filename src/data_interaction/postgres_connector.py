#!/usr/bin/env python
# -*- coding: utf-8 -*-

# %%

import numpy as np
import pandas as pd
from os.path import join as p_join
from matplotlib import pyplot as plt
import json, os, time, re, math
import pickle
from multiprocessing import Pool
import psycopg2 as pg
from functools import partial

# %% Workload DB Config

JOB_config = {
    "user": "lianyuan",
    "host": "localhost",
    "password": "",
    "db_name": "imdbload", 
    "port": "6432"
    # "port": "5432"
}

STATS_config = {
    "user": "lianyuan",
    "host": "localhost",
    "password": "",
    "db_name": "stats", 
    "port": "6432"
    # "port": "5432"
}

RELEASE_config = {
    "user": "lianyuan",
    "host": "localhost",
    "password": "",
    "db_name": "release_eight_tables", 
    "port": "6432"
}

DSB_config = {
    "user": "lianyuan",
    "host": "localhost",
    "password": "",
    # "db_name": "dsb_2g",    
    "db_name": "dsb_5g",    # 目前采用5g的数据
    "port": "6432",
    # "port": "5432"
}
  
workload_option = {
    "job": JOB_config,
    "stats": STATS_config,
    "release": RELEASE_config,
    "dsb": DSB_config
}

def connector_instance(workload = "job", extra_info = {}):
    if workload in workload_option.keys():
        db_config = workload_option[workload]
        db_config.update(extra_info)
        if len(extra_info) > 0:
            print(f"connector_instance: workload = {workload}. extra_info = {extra_info}.")
        return Connector(**db_config)
    else:
        raise ValueError("Unsupported connector_instance: {}".format(workload))
        return None


def parse_all_subquery_cardinality(plan_dict, kw = "Plan Rows"):
    """
    获得估计过程中所有子查询的基数。
    
    Args:
        arg1:
        arg2:
    Returns:
        subquery_res:
        single_table_res:
    """

    global_result = {}          # subquery的结果
    single_table_result = {}    # 单表的结果

    def construct_key(table_set):
        # 构造字典的key，需要进行排序
        sorted_list = sorted(list(table_set))
        return tuple(sorted_list)

    def local_func(plan_dict):
        """
        递归处理子查询的基数
        
        Args:
            plan_dict:
        Returns:
            table_set:
        """
        node_type = plan_dict["Node Type"]
        # 判断节点的类型
        if node_type in ["Hash Join", "Merge Join", "Nested Loop"]:
            # 如果遇到的是Join类型的节点，进行分裂处理
            current_cardinality = plan_dict[kw]
            left_plan_dict, right_plan_dict = plan_dict["Plans"]     # 获得左计划和右计划

            left_set = local_func(left_plan_dict)       # 连接左边的数据处理
            right_set = local_func(right_plan_dict)     # 连接右边的数据处理
            total_set = left_set | right_set            # 合并结果

            global_result[construct_key(total_set)] = current_cardinality
            return total_set
        else:
            if node_type in ["Index Scan", "Seq Scan", "Bitmap Heap Scan", "Index Only Scan"]:
                # 如果是scan类型的节点
                total_set = set([plan_dict["Alias"],])
                if node_type == "Seq Scan":     
                    # 针对某些单表扫描，获取其具体的基数
                    # global_result[construct_key(total_set)] = plan_dict["Plan Rows"]
                    single_table_result[plan_dict["Alias"]] = plan_dict[kw]
                return total_set
            else:
                # 如果是其他类型的节点，目前不处理，直接探索下一层（考虑到Aggregate的情况）
                if "Plans" in plan_dict and len(plan_dict["Plans"]) == 1:
                    return local_func(plan_dict["Plans"][0])   
                else:
                    return set([])

    local_func(plan_dict)
    return global_result, single_table_result


# %%

# 和pg_hint_plan相关的代码

disable_parallel = "SET max_parallel_workers_per_gather = 0;"   # 禁用并行化的操作
enable_parallel = "SET max_parallel_workers_per_gather = 10;"   # 启用并行化的操作

load_pg_hint = "LOAD 'pg_hint_plan';"                           #
disable_reorder = "SET join_collapse_limit = 1;"                # 
enable_reorder = "SET join_collapse_limit = 20;"                # 禁止
disable_geqo = "SET geqo = 0;"                                  # 禁用遗传算法处理大的Join
disable_bitmapscan = "SET enable_bitmapscan = off;"             # 禁用bitmap扫描

enable_print_subquery = "SET pg_hint_plan.enable_print_subqueries = true;"      # 启动打印subquery的步骤
disable_print_subquery = "SET pg_hint_plan.enable_print_subqueries = false;"    # 关闭打印subquery的步骤

global_setting_dict = {
    "disable_parallel": disable_parallel,
    "disable_geqo": disable_geqo,
    "disable_bitmapscan": disable_bitmapscan
}


dummy_hint = \
"""/*+ 
    File(/123)
*/\n"""

foreign_key_cmd = """SELECT conrelid::regclass AS table_name, 
       conname AS foreign_key, 
       pg_get_constraintdef(oid) 
FROM   pg_constraint 
WHERE  contype = 'f' 
AND    connamespace = 'public'::regnamespace   
ORDER  BY conrelid::regclass::text, contype DESC;"""

column_cnt_cmd = """SELECT {column_name}, COUNT(*) FROM {table_name} WHERE {column_name} IS NOT NULL
GROUP BY {column_name} ORDER BY {column_name};"""

column_type_cmd = 

num_process = 40

# %%

def execute_query_list(q_list, user = "lianyuan", host = "localhost", \
    password = "", db_name = "imdbload", port = "6432", timeout = 10000):
    # 这里timeout以毫秒为单位，默认限制是10秒
    # 处理一批的查询
    result_list = []
    config = {
        "user": user,
        "host": host,
        "password": password,
        "database": db_name,
        "port": port
    }
    if timeout is None or timeout <= 0:
        # timeout是None或者是小于零非法的
        conn = pg.connect(**config)
    else:
        conn = pg.connect(**config, options = "-c statement_timeout={}".format(timeout))

    with conn.cursor() as cursor:
        # 为了防止出很意外的结果，这里直接把结果nestloop distable掉(可能也考虑用pandas算)
        cursor.execute("set enable_nestloop=off;")
        for q in q_list:
            try:
                cursor.execute(q)
                result = cursor.fetchall()
                result_list.append(result)
            except pg.errors.QueryCanceled as e:
                conn.rollback()
                result_list.append(None)    # 无效的话结果是None
            except pg.errors.InternalError_ as e:
                conn.rollback()
                result_list.append(None)    # 无效的话结果是None
    conn.close()    # 关闭释放连接
    return result_list


class Connector(object):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, user, host, password, db_name, port):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        config = {
            "user": user,
            "host": host,
            "password": password,
            "database": db_name,
            "port": port
        }

        self.db_info = config
        self.config = config

        # 
        max_try_times = 3
        while True:
            try:
                self.conn = pg.connect(**config)
                break
            except pg.OperationalError as e:
                time.sleep(1)
                max_try_times -= 1
                if max_try_times <= 0:
                    raise e

        self.cursor = self.conn.cursor()
        # self.cursor.execute(disable_parallel) 
        self.cursor.execute(disable_geqo)
        self.cursor.execute(load_pg_hint)

        self.hint_adder = HintAdder(alias2table = {})

    def __del__(self,):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        if self.conn is not None:
            # 关闭连接
            self.conn.close()

    def __setstate__(self, data):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        self.__dict__.update(data)
        self.conn = pg.connect(**self.db_info)
        self.cursor = self.conn.cursor()
        # self.cursor.execute(disable_parallel) 
        self.cursor.execute(disable_geqo)
        self.cursor.execute(load_pg_hint)

    def __getstate__(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        # # 保存原始的值
        # tmp_conn = self.conn
        # tmp_cursor = self.cursor

        # # 将变量置成空
        # self.conn = None
        # self.cursor = None
        variable_dict = {}
        for k, v in vars(self).items():  # 深复制创建变量字典，但还是感觉代价太大了
            variable_dict[k] = v
        
        variable_dict['conn'] = None
        variable_dict['cursor'] = None
        # variable_dict = vars(self)              # 深复制创建变量字典，但还是感觉代价太大了

        # # 还原原始的值
        # self.conn = tmp_conn
        # self.cursor = tmp_cursor
        return variable_dict

    def __del__(self,):
        """
        析构函数，关闭连接
    
        Args:
            None
        Returns:
            None
        """
        try:
            self.conn.close()   
        except AttributeError:
            # 如果没有该属性，就无事发生
            pass


    def get_plan(self, sql_text, verbose = False, hint = "", mode = "json"):
        '''
        获得查询计划
        '''
        if mode == "json":
            prefix = hint + "EXPLAIN (FORMAT JSON) "
        elif mode == "text":
            prefix = hint + "EXPLAIN "
        else:
            raise ValueError("Unexpected mode value: {}".format(mode))

        # if verbose is True:
        # print("get_plan: sql_text = {}".format(sql_text))

        self.cursor_eval()
        self.cursor.execute(prefix + sql_text)      # 直接执行，没有走cache
        record = self.cursor.fetchall()

        if mode == "json":
            return record[0][0][0]['Plan']
        elif mode == "text":
            return record

    def get_plan_under_single_process(self, sql_text, verbose = False, hint = "", mode = "json"):
        """
        单进程的查询计划获取
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        self.cursor_eval()
        self.cursor.execute(disable_parallel)
        result = self.get_plan(sql_text, verbose, hint, mode)
        self.cursor.execute(enable_parallel)
        return result

    def disable_subquery(self,):
        """
        {Description}
        
        Args:
            None
        Returns:
            None
        """
        self.cursor_eval()
        self.cursor.execute(disable_print_subquery)

    def eval_subqueries(self, sql_text):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        self.enable_subquery()
        # print("启动subquery")
        self.get_plan(sql_text, hint=dummy_hint)
        self.disable_subquery()
        # print("关闭subquery")

    def get_cardinalities(self, sql_list, timeout = None):
        """
        获得一批查询中的真实基数

        TODO: 添加多进程优化(已完成)
        
        Args:
            sql_list:
            timeout: 查询执行超时的限制
        Returns:
            result_list:
        """
        # print("get_cardinalities: sql_list = {}".format(sql_list))
        result_list = self.parallel_query_execution(sql_list, \
                    post_process="cardinality", timeout=timeout)
        # print("get_cardinalities: result_list = {}".format(result_list))
        return result_list
    

    def construct_hint_sql_list(self, sql_list, card_dict_list):
        """
        {Description}
    
        Args:
            sql_list:
            card_dict_list:
        Returns:
            result_list:
            return2:
        """
        result_list = []
        for sql_text, card_dict in zip(sql_list, card_dict_list):
            subquery_true, single_table_true = \
                card_dict['subquery'], card_dict['single_table']
            
            hint_sql_text = self.inject_cardinalities_sql(\
                sql_text, subquery_true, single_table_true, head="")
            result_list.append(hint_sql_text)
        
        return result_list


    def get_cardinalities_with_hint(self, sql_list, hint_list = None, timeout = None):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        if hint_list is not None:
            card_dict_list = hint_list
            hint_sql_list = self.construct_hint_sql_list(sql_list, card_dict_list)
        else:
            hint_sql_list = sql_list

        result_list = self.parallel_query_execution(hint_sql_list, \
                    post_process="cardinality", timeout=timeout)
        return result_list

    def parse_all_subquery_cardinality(self, plan_dict, kw = "Plan Rows"):
        """
        获得估计过程中所有子查询的基数。
        
        Args:
            plan_dict:
            kw:
        Returns:
            global_result:
            single_table_result:
        """
        return parse_all_subquery_cardinality(plan_dict=plan_dict, kw=kw)


    def execute_sql_under_single_process(self, sql_text):
        """
        单进程模式下执行sql，并且不使用cache
        
        Args:
            sql_text:
        Returns:
            result:
        """
        self.cursor_eval()
        self.cursor.execute(disable_parallel)
        self.cursor.execute(sql_text)
        record = self.cursor.fetchall()
        self.cursor.execute(enable_parallel)
        return record

    def execute_single_sql(self, sql_text, verbose = False, use_cache = True):
        """
        执行单条sql，最后连接数据库的函数，利用cache加快执行速度

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        if verbose is True:
            print("sql_text = {}".format(sql_text))

        self.cursor_eval()
        self.cursor.execute(sql_text)
        record = self.cursor.fetchall()

        return record

    def execute_with_error_handle(self, cursor:pg.extensions.cursor, sql_text):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        try:
            res = cursor.execute(sql_text)  # 尝试获得结果
        except Exception:
            cursor.rollback()               # 回滚
            res = None
        return res

    def cursor_eval(self, info = None):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        if self.conn.closed != 0:
            self.conn = pg.connect(**self.config)

        try:
            with self.conn.cursor() as cursor:
                cursor.execute("SELECT 1;")
                cursor.fetchall()
        except pg.OperationalError as e:
            self.conn = pg.connect(**self.config)

        if self.cursor.closed:
            if info is not None:
                print(info)
            self.cursor = self.conn.cursor()
            # self.cursor.execute(disable_parallel)      
            self.cursor.execute(disable_geqo)

    def enable_subquery(self,):
        """
        启动打印子查询
        
        Args:
            None
        Returns:
            None
        """
        self.cursor_eval()
        self.cursor.execute(enable_print_subquery)


    def get_card_estimation(self, query_text):
        """
        获得内部优化器估计的基数
        
        Args:
            query_text:
        Returns:
            estimated_card:
        """
        if 'COUNT(*)' in query_text:
            query_text = query_text.replace("COUNT(*)", "*")
        elif 'count(*)' in query_text:
            query_text = query_text.replace("count(*)", "*")

        prefix = "EXPLAIN (FORMAT JSON) "
        result = self.execute_single_sql(prefix + query_text)
        plan_top_dict = result[0][0][0]['Plan']
        estimated_card = plan_top_dict["Plan Rows"]

        return estimated_card


    def get_query_under_complete_hint(self, sql_text, subquery_dict, single_table_dict,\
        join_ops, leading_hint, scan_ops, head = "EXPLAIN (FORMAT JSON)"):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        def dict2list(in_dict):
            return [(k, v) for k, v in in_dict.items()]
        
        # 从字典到列表
        join_ops = dict2list(join_ops)  
        scan_ops = dict2list(scan_ops)

        config_dict = {
            "subquery_dict": subquery_dict,
            "single_table_dict": single_table_dict,
            "join_ops": join_ops,
            "leading_hint": leading_hint,
            "scan_ops": scan_ops
        }

        hint_str = self.hint_adder.generate_complete_hint_str(config_dict = config_dict)
        hint_sql_template = "{HINT_STRING}\n" + head + " {SQL_STRING}"
        hint_sql_text = hint_sql_template.format(HINT_STRING = hint_str, SQL_STRING = sql_text)

        return hint_sql_text


    def execute_sql_under_complete_hint(self, sql_text, subquery_dict, single_table_dict,\
        join_ops, leading_hint, scan_ops, head = "EXPLAIN (FORMAT JSON)"):
        """
        在具备完整hint的条件下执行sql
        
        Args:
            sql_text:
            subquery_dict:
            single_table_dict:
            join_ops:
            leading_hint:
            scan_ops:
            head:
        Returns:
            result: sql执行的结果
        """
        hint_sql_text = self.get_query_under_complete_hint(sql_text, \
            subquery_dict, single_table_dict, join_ops, leading_hint, scan_ops, head)
        # print("hint_sql_text = {}.".format(hint_sql_text))
        return self.execute_sql_under_single_process(hint_sql_text)     # 返回执行的结果

    def inject_cardinalities_sql(self, sql_text, \
        subquery_dict, single_table_dict, head = "EXPLAIN (FORMAT JSON)"):
        """
        {Description}
        
        Args:
            sql_text:
            subquery_dict:
            single_table_dict:
        Returns:
            hint_sql_text:
        """
        
        rows_list = [(k, v) for k, v in subquery_dict.items()]
        schema_list = [(k, v) for k, v in single_table_dict.items()]
        hint_str = self.hint_adder.generate_cardinalities_hint_str(rows_list, schema_list)
        hint_sql_template = "{HINT_STRING}\n" + head + " {SQL_STRING}"
        hint_sql_text = hint_sql_template.format(HINT_STRING = hint_str, SQL_STRING = sql_text)
        return hint_sql_text

    def post_processor(self, result_list, mode = ""):
        """
        对于结果获取后的处理
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        if mode == "cardinality":
            out_list = []
            for a in result_list:
                if a is not None:
                    out_list.append(a[0][0]) 
                else:
                    out_list.append(None)   # 直接添加空元素
            return out_list
        else:
            return result_list

    def query_list_packing(self, query_list, target_num = 100):
        """
        {Description}
        
        Args:
            query_list:
            target_num:
        Returns:
            query_list_batch:
        """
        query_list_batch = []
        if len(query_list) <= target_num:
            for i in query_list:
                query_list_batch.append([i,])
        else:
            if target_num > 1:
                idx_list = np.linspace(0, len(query_list), target_num, dtype=int)
                for start, end in zip(idx_list[:-1], idx_list[1:]):
                    # print("query_list_packing: start = {}. end = {}.".format(start, end))
                    query_list_batch.append(query_list[start:end])
            else:
                query_list_batch = [query_list, ]
        return query_list_batch


    def parallel_query_execution(self, sql_list, target_num = 10, post_process = None, timeout = None):
        """
        并行的查询执行
        
        Args:
            sql_list:
            post_process: 后续处理
            timeout: 查询时间限制
        Returns:
            res1:
            res2:
        """
        ts = time.time()
        query_list_batch = self.query_list_packing(sql_list, target_num=target_num)
        
        local_func = partial(execute_query_list, user = self.db_info['user'], \
            host = self.db_info['host'], password = self.db_info['password'], \
            db_name = self.db_info['database'], port = self.db_info['port'], timeout = timeout)

        with Pool(num_process) as pool:
            result_list_batch = pool.map(local_func, query_list_batch)

        from functools import reduce
        result_list = reduce(lambda a, b: a+b, result_list_batch, [])
        result_list = self.post_processor(result_list, mode=post_process)

        te = time.time()
        # print(f"parallel_query_execution: delta_time = {te - ts: .2f}. len(sql_list) = {len(sql_list)}. len(query_list_batch) = {len(query_list_batch)}")
        return result_list

# %%

class HintAdder(object):
    """
    Hint添加的相关内容

    Members:
        field1:
        field2:
    """

    def __init__(self, alias2table):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        self.table2alias = {}
        for k, v in alias2table.items():
            self.table2alias[v] = k

    def generate_complete_hint_str(self, config_dict):
        """
        {Description}
        "subquery_dict": subquery_dict,
        "single_table_dict": single_table_dict,
        "join_ops": join_ops,
        "leading_hint": leading_hint,
        "scan_ops": scan_ops

        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """

        PG_HINT_CMNT_TMP = '''/*+ {COMMENT} */'''

        hint_str_list = []

        # 处理物理算子相关
        func_dict = {
            "leading_hint": self.generate_leading_hint_str,
            "join_ops": self.generate_join_operator_hint_str,
            "scan_ops": self.generate_scan_operator_hint_str 
        }

        for k, func in func_dict.items():
            if config_dict[k] is not None:
                # 非空代表有这个hint，添加该项
                hint_str_list.append(func(config_dict[k]))
            else:
                pass

        # 处理子查询基数相关
        if config_dict['subquery_dict'] is not None and config_dict['single_table_dict'] is not None:
            rows_list = [(k, v) for k, v in config_dict['subquery_dict'].items()]
            schema_list = [(k, v) for k, v in config_dict['single_table_dict'].items()]
            hint_str_list.append(self.generate_cardinalities_hint_str(\
                rows_list, schema_list, with_wrapper=False))
                        
        return PG_HINT_CMNT_TMP.format(COMMENT = "\n".join(hint_str_list))

    def generate_leading_hint_str(self, join_order):
        """
        {Description}
        
        Args:
            join_order:
        Returns:
            hint_str:
        """
        PG_HINT_LEADING_TMP = "Leading({JOIN_ORDER})"
        if "Leading" not in join_order:
            return PG_HINT_LEADING_TMP.format(JOIN_ORDER = join_order)
        else:
            return join_order
            
    def generate_join_operator_hint_str(self, hint_list):
        """
        {Description}
        
        Args:
            hint_list:
        Returns:
            hint_str:
        """
        def remove_space(join_kw):
            # 删掉join keyword中
            kw_mapping = {
                "NestedLoop": "NestLoop"
            }
            join_kw = join_kw.replace(" ", "")
            if join_kw in kw_mapping.keys():
                return kw_mapping[join_kw]
            else:
                return join_kw

        PG_HINT_JOIN_TMP = "{JOIN_TYPE}({TABLES}) "
        hint_str_list = []
        # print("join_ops_hint: {}".format(hint_list))
        for tables, join_type in hint_list:
            hint_str_list.append(PG_HINT_JOIN_TMP.format(\
                JOIN_TYPE = remove_space(join_type), TABLES = tables))

        return "\n".join(hint_str_list)

    def generate_scan_operator_hint_str(self, hint_list):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        def remove_space(scan_kw):
            # 别名映射
            kw_mapping = {
                "BitmapHeapScan": "BitmapScan",
            }
            # 删掉join keyword中空格
            scan_kw = scan_kw.replace(" ", "")
            if scan_kw in kw_mapping.keys():
                return kw_mapping[scan_kw]
            else:
                return scan_kw

        PG_HINT_SCAN_TMP = "{SCAN_TYPE}({TABLE}) "
        hint_str_list = []
        for table, scan_type in hint_list:
            hint_str_list.append(PG_HINT_SCAN_TMP.\
                format(SCAN_TYPE = remove_space(scan_type), TABLE = table))
        return "\n".join(hint_str_list)
    

    def card_standardization(self, kv_list):
        """
        将基数估计的结果规范化，确保注入的基数是整数
    
        Args:
            kv_list:
            arg2:
        Returns:
            out_list:
            return2:
        """
        out_list = []
        for name, card in kv_list:
            try:
                card = int(card)
            except TypeError as e:
                print(f"card_standardization: meet TypeError. kv_list = {kv_list}.")
                raise e
            
            card = card if card > 1 else 1
            out_list.append((name, card))
        return out_list
    
    def generate_cardinalities_hint_str(self, rows_list, schema_list, with_wrapper = True, db = "Postgres"):
        """
        {Description}
        
        Args:
            rows_list:
            schema_list:
            db: 
        Returns:
            res1:
            res2:
        """
        rows_list = self.card_standardization(rows_list)
        schema_list = self.card_standardization(schema_list)

        def rows_line(table_set, cardinality):
            """
            构造一行row_hint
            
            Args:
                table_set:
                cardinality:
            Returns:
                line:
            """
            hint_template = "Rows({tables} #{rows})"
            table_str = " ".join(table_set)         # 构造字符串
            return hint_template.format(tables = table_str, rows = cardinality)

        def schema_line(table, cardinality):
            """
            构造一行row_hint
            
            Args:
                table_set:
                cardinality:
            Returns:
                line:
            """
            hint_template = "Schema({tables} #{rows})"
            table_str = str(table)         # 构造字符串
            return hint_template.format(tables = table_str, rows = cardinality)

        # print("generate_cardinalities_hint_str: rows_list = {}".format(rows_list))
        # print("generate_cardinalities_hint_str: schema_list = {}".format(schema_list))

        str_list = []

        # if len(rows_list[0]) > 0:
        for table_set, cardinality in rows_list:
            str_list.append(rows_line(table_set, cardinality))

        # if len(schema_list[0]) > 0:
        for table, cardinality in schema_list:
            str_list.append(schema_line(table, cardinality))

        if with_wrapper == True:
            PG_HINT_CMNT_TMP = '''/*+ {COMMENT} */'''
            return PG_HINT_CMNT_TMP.format(COMMENT="\n".join(str_list))
        else:
            return "\n".join(str_list)

# %%


def get_path_cardinalities(sql_text:str, workload:str, order_spec: list = None, timeout = None):
    """
    获得指定路径的所有基数
    
    Args:
        sql_text:
        workload:
        order_spec:
        timeout: 数据库连接的超时
    Returns:
        global_result: 
        single_table_result:
    """ 
    sql_text = sql_text.replace("COUNT(*)", "*")    # 关键字不一定需要替代
    prefix = "EXPLAIN (FORMAT JSON, ANALYZE)"       # 查询的前缀

    if sql_text.startswith(prefix) == False:
        sql_text = prefix + sql_text

    if order_spec is not None:
        pass
    
    if timeout is not None:
        # 设置超时
        local_conn = pg.connect(**workload_option[workload], \
                                options = "-c statement_timeout={}".format(timeout))
    else:
        local_conn = pg.connect(**workload_option[workload])

    try:
        with local_conn.cursor() as local_cursor:
            result = local_cursor.execute(sql_text)     # 获得查询计划
    except Exception:
        pass
    # 关闭数据库连接
    local_conn.close()

    global_result, single_table_result = parse_all_subquery_cardinality(\
        plan_dict=result[0][0][0]['Plan'], kw="Actual Rows")

    return global_result, single_table_result

# %%

def batch_path_cardinalities(sql_list, workload, order_list):
    """
    批量的路径基数获取
    
    Args:
        sql_list:
        workload:
        order_list:
    Returns:
        result_list:
    """
    num_process = 20
    def get_path_cardinalities_wrapper(args):
        return get_path_cardinalities(*args)
    # 
    args_list = []
    for item1, item2 in zip(sql_list, order_list):
        args_list.append((item1, workload, item2))

    with Pool(num_process) as pool:
        result_list = pool.map(get_path_cardinalities_wrapper, args_list)
    return result_list

# %%
