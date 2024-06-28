#!/usr/bin/env python
# -*- coding: utf-8 -*-


# %%

from asynchronous import construct_input, process_output
from utility import utils, workload_spec
import psutil
import time
import psycopg2 as pg

class WorkloadInspector(object):
    """
    async模式下，工作负载的监控器

    Members:
        field1:
        field2:
    """

    def __init__(self, workload: str, db_conn: pg.extensions.connection):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        self.workload = workload
        self.db_conn = db_conn

        self.reset_monitor_state()

    def reset_monitor_state(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        self.pid_mapping = {}
        self.card_dict_mapping = {}
        self.extension_info = {}

    def reset_connection(self,):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        if self.db_conn.closed == True:
            self.db_conn = pg.connect(**construct_input.conn_dict(self.workload))

    def load_card_info(self, signature, subquery_dict, single_table_dict):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        # print("load_card_info: subquery_dict = {}. single_table_dict = {}.".format(subquery_dict, single_table_dict))
        local_info = {
            "subquery_num": len(subquery_dict),
            "single_table_num": len(single_table_dict),
            "complete": False
        }

        self.extension_info[signature] = local_info

    def remove_task(self, signature):
        """
        移除特定的任务
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        pass

    def get_all_unfinished_tasks(self, ):
        """
        获得所有未完成的任务
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        result_list = []
        for k, v in self.extension_info.items():
            if v['complete'] == False:
                result_list.append(k)
        return result_list

    # @utils.timing_decorator
    def get_proc_mapping(self, signature):
        """
        {Description}

        Args:
            signature:
            arg2:
        Returns:
            return1:
            return2:
        """
        #
        proc_fpath = construct_input.get_proc_fpath(signature)
        subquery_mapping, single_table_mapping = \
            process_output.get_async_proc_mapping(proc_fpath)
        self.pid_mapping[signature] = subquery_mapping, single_table_mapping

        return subquery_mapping, single_table_mapping
    

    def get_finished_proc(self, signature):
        """
        获得已经执行完成的进程
    
        Args:
            signature:
            arg2:
        Returns:
            proc_list:
            return2:
        """
        return list(self.get_proc_cpu_time(signature).keys())

    # @utils.timing_decorator
    def get_proc_cpu_time(self, signature):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        proc_fpath = construct_input.get_proc_fpath(signature)
        return process_output.get_async_proc_cpu_time(proc_fpath)

    # @utils.timing_decorator
    def get_time_dict(self, signature):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        # proc_fpath = construct_input.get_proc_fpath(signature)
        proc_fpath = construct_input.get_output_path(signature)
        # return process_output.get_async_proc_cpu_time(proc_fpath)
        return process_output.get_async_duration(out_path=proc_fpath)
    
    # @utils.timing_decorator
    def get_card_dict(self, signature):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        if self.extension_info[signature]['complete'] == True:
            return self.card_dict_mapping[signature]
        
        output_path = construct_input.get_output_path(signature)

        try:
            subquery_out, single_table_out = \
                process_output.get_async_cardinalities(output_path)
        except FileNotFoundError as f_error:
            print("WorkloadInspector.get_card_dict: File({}) not found".format(f_error.filename))
            subquery_out, single_table_out = {}, {}

        self.card_dict_mapping[signature] = subquery_out, single_table_out

        # 处理complete的情况
        # print("get_card_dict: len(subquery_out) = {}. subquery_num = {}. len(single_table_out) = {}. single_table_num = {}.".\
        #       format(len(subquery_out), self.extension_info[signature]["subquery_num"], len(single_table_out), \
        #              self.extension_info[signature]["single_table_num"]))
        
        if len(subquery_out) == self.extension_info[signature]["subquery_num"] and \
            len(single_table_out) == self.extension_info[signature]["single_table_num"]:
            self.extension_info[signature]['complete'] = True           # 标记为已完成，之后就不应该读取了
            # 信息读入
            self.card_dict_mapping[signature] = subquery_out, single_table_out
        
        return subquery_out, single_table_out

    # @utils.timing_decorator
    def get_workload_state(self, ):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        # start_time = time.time()

        unfinished_list = self.get_all_unfinished_tasks()
        result_dict = {}

        for signature in unfinished_list:
            proc_cpu_dict, local_cpu_time, finished_num, running_num = self.get_cpu_elapsed_time(signature)
            # print("get_workload_state: finished_num = {}. running_num = {}.".format(finished_num, running_num))
            
            subquery_curr, single_table_curr = self.get_card_dict(signature=signature)
            subquery_time, single_table_time = self.get_time_dict(signature=signature)

            result_dict[signature] = {
                "cpu_time_dict": proc_cpu_dict,
                "cpu_time_total": local_cpu_time,
                "finished_num": finished_num,
                "running_num": running_num,
                "subquery_dict": subquery_curr,
                "single_table_dict": single_table_curr,
                "subquery_time": subquery_time,
                "single_table_time": single_table_time
            }

        # end_time = time.time()
        return result_dict

    # @utils.timing_decorator
    def get_cpu_elapsed_time(self, signature):
        """
        获得task关联总的CPU时间

        计算任务所消耗的时间
        由于并行计算的问题，这里结果可能还要调整。

        根据process来计算总时间
        + 对于完成的process，使用xxx_proc.json的信息来进行计算
        + 对于未完成的process，由psutil来进行时间的计算

        Args:
            arg1:
            arg2:
        Returns:
            proc_cpu_dict: 
            local_cpu_time: 
            finished_num: 
            running_num:
        """
        subquery_mapping, single_table_mapping = self.get_proc_mapping(signature)

        # 针对pid_list进行去重
        pid_list = list(set(subquery_mapping.values())) + list(set(single_table_mapping.values()))
        pid_list = list(set(pid_list))

        total_time = 0.0
        proc_cpu_dict = {}
        proc_time_dict = self.get_proc_cpu_time(signature)

        unfinished_list, error_list = [], []
        for pid in pid_list:
            if pid in proc_time_dict.keys():
                # 跑完的情况
                # print("get_cpu_elapsed_time: pid({}) has finished.".format(pid))
                total_time += proc_time_dict[pid]
                proc_cpu_dict[pid] = (proc_time_dict[pid], True)
            else:
                try:
                    # 没跑完的情况
                    curr_proc = psutil.Process(pid=pid)
                    # print("get_cpu_elapsed_time: pid({}) is still running.".format(pid))
                    curr_times = curr_proc.cpu_times()
                    curr_total = (curr_times.user + curr_times.system)
                    proc_cpu_dict[pid] = (curr_total, False)
                    total_time += curr_total
                    unfinished_list.append(pid)
                except Exception as e:
                    # 没有找到进程的情况
                    error_list.append(pid)
                    # print(f"get_cpu_elapsed_time: pid({pid}) not found. error is {e}")
                    total_time += 0.0
                    # raise e
                
        if len(error_list) > 0:
            print("get_cpu_elapsed_time: signature = {}. total_list = {}. finished_list = {}. unfinished_list = {}. error_list = {}.".\
              format(signature, pid_list, list(proc_time_dict.keys()), unfinished_list, error_list))

        return proc_cpu_dict, total_time, len(proc_time_dict), len(pid_list) - len(proc_time_dict)


    def get_server_cpu_state(self,):
        """
        获得整个服务器的CPU状态
    
        Args:
            arg1:
            arg2:
        Returns:
            total_num:
            total_cpu:
        """
        process_list = []

        for proc in psutil.process_iter(attrs=['pid', 'name', 'cpu_percent']):
            process_list.append(proc.info)

        db_proc_list = []
        for item in process_list:
            if item['name'] == "postgres":
                db_proc_list.append(item)

        db_proc_list = sorted(db_proc_list, key=lambda x: x['cpu_percent'], reverse=True)

        total_num, total_cpu = len(db_proc_list), sum([item['cpu_percent'] for item in db_proc_list])
        # return db_proc_list

        return total_num, total_cpu
    

    def get_server_memory_usage(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            free:
            used:
            buffers
        """
        scale_factor = 1e9
        memory_obj = psutil.virtual_memory()
        memory_free = memory_obj.free / scale_factor
        memory_used = memory_obj.used / scale_factor
        memory_buffers = memory_obj.buffers / scale_factor

        return memory_free, memory_used, memory_buffers
    

    def get_database_conn_state(self,):
        """
        获得当前连接数据库的状态
    
        Args:
            None
        Returns:
            conn_num:
            return2:
        """
        db_name = workload_spec.workload2database[self.workload]
        local_query = f"SELECT COUNT(*) FROM pg_stat_activity WHERE datname = '{db_name}';"
        try:
            with self.db_conn.cursor() as cursor:
                cursor.execute(local_query)
                conn_num = int(cursor.fetchall()[0][0])
        except pg.OperationalError as e:
            self.db_conn = pg.connect(**construct_input.conn_dict(self.workload))
            with self.db_conn.cursor() as cursor:
                cursor.execute(local_query)
                conn_num = int(cursor.fetchall()[0][0])
        return conn_num

    def get_correlated_processes(self, info, mode):
        """
        signature获得相关的进程信息，mode支持两种，一个获得其本身，
        二是获得所有，分别用own、all来表示这两种情况
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        subquery_mapping, single_table_mapping = self.get_proc_mapping(info)
        # print("subquery_mapping = {}".format(subquery_mapping))
        # print("single_table_mapping = {}".format(single_table_mapping))

        subquery_pid_list = [v for v in subquery_mapping.values()]
        single_table_pid_list = [v for v in single_table_mapping.values()] 
        pid_list = single_table_pid_list + subquery_pid_list   # 获得pid_list

        # print("pid_list = {}".format(pid_list))
        all_template = "SELECT pid, leader_pid FROM pg_stat_activity WHERE leader_pid IN ({});"

        # 针对pid_list进行去重
        pid_list = list(set(pid_list))

        if mode == "own":
            # 直接返回相关的pid_list即可
            return pid_list
        elif mode == "all":
            curr_query = all_template.format(",".join([str(pid) for pid in pid_list]))
            with self.db_conn.cursor() as cursor:
                cursor.execute(curr_query)
                result = cursor.fetchall()
            out_list = []

            # 整合result
            for pid, _ in result:
                out_list.append(pid)
            out_list = out_list + pid_list

            # 针对out_list进行去重
            out_list = list(set(out_list))
            return out_list
    

# %%
