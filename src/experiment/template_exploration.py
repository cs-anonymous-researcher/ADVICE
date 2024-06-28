#!/usr/bin/env python
# -*- coding: utf-8 -*-


# %%
import numpy as np
import pandas as pd
from os.path import join as p_join
from matplotlib import pyplot as plt
import json, os, time, re, math
import pickle, queue
from operator import add
from functools import reduce

# %%

from utility import utils
from plan import plan_init, plan_analysis, plan_template
from utility.utils import set_verbose_path, verbose
# from algo import thompson_sampling
from collections import defaultdict, Counter
from query import query_construction, ce_injection
from grid_manipulation import grid_preprocess

# %%

class TemplateExplorationExperiment(object):
    """
    有关template的相关探索程序，由于实验的问题，要考虑历史数据是否能复用的问题，
    复用数据可以提高实验的效率，但是对实验来说不公平

    Members:
        field1:
        field2:
    """

    def __init__(self, workload = "stats", ce_handler = "internal", table_num = 5, dynamic_config = {}, \
        dump_strategy = "remain", intermediate_path = "/home/lianyuan/Research/CE_Evaluator/intermediate/", \
        namespace = None, split_budget = 100):
        """
        {Description}

        Args:
            workload:
            table_num:
            dynamic_config: 动态的配置信息
            dump_strategy: 导出结果的策略
            intermediate_path: 中间结果的存储路径
            namespace: 命名空间
        """
        print(f"TemplateExplorationExperiment.__init__: workload = {workload}. ce_handler = {ce_handler}. namespace = {namespace}.")

        # 默认情况下选择所有的表
        self.table_num = table_num    # 
        self.workload = workload
        self.ce_handler_str = ce_handler
        
        # 新增的对象，需要指定ce_handler
        self.workload_manager = plan_init.WorkloadManager(workload=self.workload, ce_handler=ce_handler)

        self.construct_metrics_list = self.workload_manager.construct_metrics_list

        self.intermediate_path = intermediate_path
        def complement_path(suffix_str):
            return p_join(intermediate_path, suffix_str)

        if namespace is None:
            self.namespace = namespace
            # 设置TemplateManager的参数
            self.template_manager = plan_template.TemplateManager(workload=self.workload, inter_path= \
                complement_path(f"{workload}/template_obj/{ce_handler}"), dynamic_config=dynamic_config, \
                dump_strategy=dump_strategy, split_budget=split_budget)

            # 元信息包含workload的信息
            self.inter_template_path = complement_path(f"{workload}/template_obj/{ce_handler}")
        else:
            if namespace == "timestamp":
                # 创建时间戳
                self.namespace = time.strftime("%Y%m%d%H%M%S", time.localtime())
            else:
                self.namespace = namespace
            
            ns = self.namespace
            self.template_manager = plan_template.TemplateManager(workload=self.workload, inter_path=
                complement_path(f"{workload}/template_obj/{ce_handler}/{ns}"), dynamic_config=dynamic_config, 
                dump_strategy=dump_strategy, split_budget=split_budget)
            
            # self.inter_template_path = complement_path("{}/{}/template_obj".format(workload, ns))
            self.inter_template_path = complement_path(f"{workload}/template_obj/{ce_handler}/{ns}")   # 命名方式的修改

            print("TemplateExplorationExperiment.__init__: Don't use history. inter_template_path = {}".\
                  format(self.inter_template_path))
            
        # 创建路径
        os.makedirs(self.inter_template_path, exist_ok=True)

        self.template_meta_path = p_join(self.inter_template_path, "meta_info.json")
        self.query_stats_path = p_join(self.inter_template_path, "query_stats.json")

        self.parse_meta()

    # def complement_path(self, suffix_str):
    #     return p_join(self.intermediate_path, suffix_str)

    def get_meta_path(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        return self.template_meta_path
    
    def select_potential_templates(self, num, batch_id: str = "latest",
        strategy: str = "max-greedy", mode: str = "under", schema_duplicate: int = 2,
        existing_templates = []):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            result_list:
            filtered_list:
        """
        result_list, filtered_list = self.workload_manager.select_potential_templates(\
            num, batch_id, strategy, mode, schema_duplicate, return_inter=True)

        if mode == "under":
            kw = "under-estimation"
        elif mode == "over":
            kw = "over-estimation"
        else:
            raise ValueError(f"select_potential_templates: mode = {mode}")

        # 结果上添加关键字
        result_new = [(item + (kw,)) for item in result_list]

        # print(f"TemplateExplorationExperiment.select_potential_templates: kw = {kw}. "\
        #       f"before_length = {len(result_list[0])}. after_length = {len(result_new[0])}.")

        # 导出filtered_list信息
        self.update_ref_query_info(filtered_list)
        return result_new, filtered_list

    def update_ref_query_info(self, filtered_list):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        curr_time_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
        local_dict = {
            "gen_time": curr_time_str,
            "result_list": filtered_list
        }
        next_id = len(self.query_stats_dict) + 1
        self.query_stats_dict[next_id] = local_dict

        utils.dump_json(self.query_stats_dict, self.query_stats_path)
        return self.query_stats_dict
    
    def get_namespace(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        return self.namespace

    def parse_meta(self,):
        """
        解析元信息
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        self.template_meta_dict = utils.load_json(self.template_meta_path)
        self.query_stats_dict = utils.load_json(self.query_stats_path)

        return self.template_meta_dict, self.query_stats_dict

    def load_template_info(self, mode = "all"):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        def parse_template_name(tmpl_name_str:str):
            table_part, column_part, useless_part = tmpl_name_str.split("#")
            table_list = table_part.split("&")
            col_str_list = column_part.split("&")
            alias_mapping = query_construction.abbr_option[self.workload]
            column_list = [(alias_mapping[item[0]], item[1]) for item in query_construction.\
                parse_compound_column_list(col_str_list, self.workload)]

            return table_list, column_list
        
        def parse_meta_dict(in_dict: dict):
            template_meta_list = []

            for k, v in in_dict.items():
                template_meta_list.append(parse_template_name(v['template_key']))

            return template_meta_list
        
        template_info_list = []
        root_dir = p_join(self.intermediate_path, self.workload, "template_obj", self.ce_handler_str)
        root_meta_path = p_join(root_dir, "meta_info.json")

        if mode == "all":
            root_meta_dict = utils.load_json(root_meta_path)
            print(f"load_template_info: root_meta_dict = {root_meta_dict}")
            template_info_list.extend(parse_meta_dict(root_meta_dict))

            sub_dir_list = [d for d in os.listdir(root_dir) if \
                            os.path.isdir(os.path.join(root_dir, d))]
            
            print(f"load_template_info: sub_dir_list = {sub_dir_list}")
            if len(sub_dir_list) > 0:
                template_info_list.extend(self.load_template_info(mode=sub_dir_list))
        elif mode == "current":
            # utils.load_json()
            template_info_list.extend(parse_meta_dict(self.template_meta_dict))
        elif isinstance(mode, str):
            selected_meta_path = p_join(root_dir, mode, "meta_info.json")
            selected_meta_dict = utils.load_json(selected_meta_path)
            template_info_list.extend(parse_meta_dict(selected_meta_dict))
        elif isinstance(mode, list):
            for m in mode:
                selected_meta_path = p_join(root_dir, m, "meta_info.json")
                print(f"load_template_info: selected_meta_path = {selected_meta_path}.")

                selected_meta_dict = utils.load_json(selected_meta_path)
                template_info_list.extend(parse_meta_dict(selected_meta_dict))

        print(f"load_template_info: root_dir = {root_dir}.")
        print(f"load_template_info: template_info_list = {template_info_list}.")

        return template_info_list
    

    def get_workload_error_state(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            metrics_dict:
            return2:
        """
        manager = self.workload_manager
        # est_card_list, true_card_list = \
        #     manager.estimation_global, manager.label_global
        _, _, true_card_list, est_card_list = manager.construct_sub_workload(batch_id="all")
        
        q_error_list = [max((a + 1) / (b + 1), (b + 1) / (a + 1)) for \
                        a, b in zip(est_card_list, true_card_list)]

        pair_list = list(zip(est_card_list, true_card_list))
        print(f"get_workload_error_state: pair_list = {pair_list[:20]}.")
        print(f"get_workload_error_state: q_error_list = {q_error_list[:20]}.")

        description_list = ["max", "95th", "90th", "75th", "median"]
        value_list = [np.max(q_error_list), np.quantile(q_error_list, 0.95), \
            np.quantile(q_error_list, 0.9), np.quantile(q_error_list, 0.75), np.median(q_error_list)]
        metrics_dict = {k: v for k, v in zip(description_list, value_list)}

        return metrics_dict

    def update_template_meta(self, template_info_dict: dict):
        """
        更新模板的元信息
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        print("update_template_meta: template_info_dict = {}.".format(template_info_dict))

        for k, v in template_info_dict.items():
            if self.template_meta_dict == {}:
                next_template_id = 0
            else:
                next_template_id = max([int(k) for k in self.template_meta_dict.keys()]) + 1    # 需要先转化成数字

            # self.template_meta_dict[next_template_id] = {
            self.template_meta_dict[str(next_template_id)] = {
                "template_key": k, 
                "info": v
            }

        # 导出结果
        utils.dump_json(self.template_meta_dict, self.template_meta_path)


    def load_historical_workload(self, query_list, meta_list, label_list):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        self.workload_manager.load_queries(query_list=query_list, \
                    meta_list=meta_list, label_list=label_list)


    def show_workload_state(self,):
        """
        展示当前负载结果的状态
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        self.workload_manager.show_workload_state()


    def construct_hybrid_template_dict(self, over_template_num = 3, under_template_num = 3, \
            existing_range = "all", bins_builder: grid_preprocess.BinsBuilder = None, schema_duplicate = 2):
        """
        针对under-estimation和over-estimation这两种情况创建模版
    
        Args:
            over_template_num:
            under_template_num:
            existing_range:
            bins_builder:
        Returns:
            template_dict:
            info_dict:
        """
        print(f"construct_hybrid_template_dict: over_template_num = {over_template_num}. under_template_num = {under_template_num}.")

        if existing_range is not None:
            existing_templates = self.load_template_info(mode=existing_range)
        else:
            existing_templates = []
        
        # 20240130: 设置score为模版选择策略，期望获得更强的鲁棒性
        # strategy = "max-greedy"
        strategy = "score"
        # 考虑基数估计器under-estimation的情况，即估计的小了
        res_template_under, filtered_under = self.select_potential_templates(num=under_template_num, 
            strategy=strategy, mode="under", existing_templates=existing_templates, schema_duplicate=schema_duplicate)
        
        # 考虑基数估计器over-estimation的情况，即估计的大了
        res_template_over, filtered_over = self.select_potential_templates(num=over_template_num, 
            strategy=strategy, mode="over", existing_templates=existing_templates, schema_duplicate=schema_duplicate)
        
        res_template_merge = res_template_under + res_template_over
        filtered_merge = filtered_under + filtered_over

        # print("construct_promising_template_dict: len(res_template_list) = {}. len(filtered_list) = {}.".\
        #       format(len(res_template_merge), len(filtered_merge)))
        
        # 根据res_template_list构造cond_bound_dict_list

        if bins_builder is None:
            bins_builder = grid_preprocess.get_bins_builder_by_workload(workload = self.workload)
        
        iter_cond_builder = plan_init.IterationConditionBuilder(\
            workload=self.workload, bins_builder=bins_builder, split_budget=100)
        
        return self.build_template_dict_along_params(res_template_merge, filtered_merge, iter_cond_builder)


    def build_template_dict_along_params(self, res_template_list, filtered_list, iter_cond_builder):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            template_dict: 
            info_dict:
        """
        fact_num = 5
        cond_bound_dict_list = []

        for (query_meta, selected_columns, mode), filtered_item in zip(res_template_list, filtered_list):
            fact_records = self.workload_manager.find_fact_record(\
                in_key=filtered_item[0], num=fact_num)
            
            # print(f"construct_promising_template_dict: selected_columns = {selected_columns}")
            # for record in fact_records:
            #     print(f"build_template_dict_along_params: record = {record}")
            cond_bound_dict = iter_cond_builder.construct_condition_iteration_options(\
                fact_record_list = fact_records, column_list = selected_columns)
            cond_bound_dict['max_length'] = len(list(cond_bound_dict.values())[0])  # 设置其最大长度
            cond_bound_dict_list.append(cond_bound_dict)
    
        template_dict, output_path_dict = self.template_manager.create_templates_under_cond_bound(\
            parameter_list=res_template_list, cond_bound_list = cond_bound_dict_list)
        info_dict = {}
        
        for k, v in output_path_dict.items():
            info_dict[k] = {
                "path": v,
                "info": template_dict[k].make_info_dict()
            }

        self.update_template_meta(template_info_dict=info_dict)
        return template_dict, info_dict


    def construct_promising_template_dict(self, template_num = 3, existing_range = "all", \
                                          bins_builder: grid_preprocess.BinsBuilder = None):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        if existing_range is not None:
            existing_templates = self.load_template_info(mode=existing_range)
        else:
            existing_templates = []
        
        # 只考虑基数估计器under-estimation的情况，即估计的小了
        res_template_list, filtered_list = self.select_potential_templates(\
            num=template_num, strategy="max-greedy", mode="under", existing_templates=existing_templates)
        
        # print("construct_promising_template_dict: len(res_template_list) = {}. len(filtered_list) = {}.".\
        #       format(len(res_template_list), len(filtered_list)))
        
        # 根据res_template_list构造cond_bound_dict_list

        if bins_builder is None:
            bins_builder = grid_preprocess.get_bins_builder_by_workload(workload = self.workload)
        
        iter_cond_builder = plan_init.IterationConditionBuilder(\
            workload=self.workload, bins_builder=bins_builder, split_budget=100)
        
        return self.build_template_dict_along_params(res_template_list, filtered_list, iter_cond_builder)
    

    def construct_spec_template_dict(self, template_num = 3, template_params_list = None, existing_range = "all"):
        """
        构建模版字典
        
        Args:
            template_num:
            template_params_list:
            existing_range:
        Returns:
            template_dict:
            res2:
        """
        if template_params_list is None:
            if existing_range is not None:
                existing_templates = self.load_template_info(mode=existing_range)
            else:
                existing_templates = []
                
            res_template_list, _ = self.select_potential_templates(\
                num=template_num, strategy="max-greedy", mode="under", existing_templates=existing_templates)
        else:
            res_template_list = template_params_list

        print("construct_spec_template_dict: res_template_list = {}.".format(res_template_list))

        template_dict, output_path_dict = \
            self.template_manager.create_batch_templates(parameter_list=res_template_list)
        info_dict = {}
        
        for k, v in output_path_dict.items():
            info_dict[k] = {
                "path": v,
                "info": template_dict[k].make_info_dict()
            }

        self.update_template_meta(template_info_dict=info_dict)
        return template_dict, info_dict
    

    def get_final_manager(self,):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        return self.template_manager


    def template_status_evaluation(self,):
        """
        评测template_manager中模版的状态

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        self.template_manager.template_status_evaluation()
    
    def parse_missing_templates(self, template_dir = None):
        """
        解析模版
    
        Args:
            template_dir:
            arg2:
        Returns:
            local_meta_dict:
            return2:
        """
        file_suffix = "_output.pkl"    # 文件的后缀
        if template_dir is None:
            template_dir = self.inter_template_path

        file_list = os.listdir(template_dir)
        local_meta_dict = {}
        print("self.template_meta_dict = {}.".format(self.template_meta_dict))

        existing_path_set = set([item["info"]["path"] for item in self.template_meta_dict.values()])
        for f_name in file_list:
            f_path = p_join(template_dir, f_name)
            if f_path in existing_path_set:
                print("parse_missing_templates: f_path({}) is already in dict".format(f_path))
                continue
            elif f_name.endswith(file_suffix):
                local_template: plan_template.TemplatePlan = utils.load_pickle(f_path)
                tmpl_key = local_template.get_template_key()       # 构造template key
                local_meta_dict[tmpl_key] = {
                    "path": f_path,
                    "info": local_template.make_info_dict()
                }
            else:
                # 不符合匹配的条件
                continue

        self.update_template_meta(template_info_dict=local_meta_dict) # 更新全局的元信息
        # print("local_meta_dict = {}.".format(local_meta_dict))
        return local_meta_dict

# %%

class JoinOrderEvaluator(object):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, workload, selected_tables = ["badges", "comments", "posthistory", "postlinks", "posts", "tags", \
        "users", "votes"], data_dir = "/home/lianyuan/Research/CE_Evaluator/intermediate/{}/join_order_obj", \
        load_history = True, split_budget = 100):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        self.workload = workload
        self.selected_tables = selected_tables
        self.search_initilizer = plan_init.get_initializer_by_workload(\
            schema_list=self.selected_tables, workload=self.workload, split_budget=split_budget)
        self.data_dir = data_dir.format(self.workload)
        meta_path = p_join(self.data_dir, "meta_info.json")
        self.meta_path = meta_path
        self.meta_dict = utils.load_json(meta_path)
        # print(f"JoinOrderEvaluator.__init__: self.meta_dict = {self.meta_dict}.")
        self.result_dict = {}

        if load_history:
            self.load_join_order_result()

    
    def get_next_res_idx(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        return len(self.meta_dict) + 1

    def save_join_order_result(self, config, query_list, meta_list, jo_str_list):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        res_idx = self.get_next_res_idx()
        # res_idx = len(self.meta_dict) + 1
        res_signature = utils.get_signature(input_string=str(query_list), num_out = 8)
        res_path = p_join(self.data_dir, f"join_order_{res_signature}.pkl")
        print(f"save_join_order_result: res_path = {res_path}")

        self.meta_dict[res_idx] = {
            "path": res_path,
            "config": config
        }
        result = query_list, meta_list, jo_str_list
        utils.dump_json(res_dict=self.meta_dict, data_path=self.meta_path)
        utils.dump_pickle(result, res_path)

        return res_idx

    def show_existing_results(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        for k, v in self.meta_dict.items():
            print("key = {}. value = {}.".format(k, v))

        return self.meta_dict
    

    def get_total_query_number(self,):
        """
        获得当前总的查询数目
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        total_num = 0
        for _, v in self.result_dict.items():
            total_num += len(v[0])
        return total_num

    def load_join_order_result(self,):
        """
        加载所有的历史Join Order结果
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        for k, v in self.meta_dict.items():
            if k not in self.result_dict:
                self.result_dict[k] = utils.load_pickle(v['path'])

        return self.result_dict

    def result_aggregation(self, priori_num = 3, out_num = None):
        """
        {Description}
        
        Args:
            priori_num:
            arg2:
        Returns:
            leading_comb_list:
            res2:
        """
        # result_dict = self.load_join_order_result()
        result_dict = self.result_dict

        query_global, meta_global, jo_str_global = [], [], []
        for k, v in result_dict.items():
            query_local, meta_local, jo_str_local = v
            query_global.extend(query_local)
            meta_global.extend(meta_local)
            jo_str_global.extend(jo_str_local)

        leading_comb_list = list()

        # for jo_str in zigzag_list:
        for jo_str in jo_str_global:
            # print("jo_str = {}.".format(jo_str))
            analyzer = plan_analysis.JoinOrderAnalyzer(join_order_str=jo_str)
            # 暂时考虑3表join的情况
            table_subset = analyzer.get_leading_tables(table_num=priori_num)
            leading_comb_list.append(tuple(table_subset))

        # return leading_comb_list
        leading_comb_cnt = Counter(leading_comb_list)
        # print(f"leading_comb_cnt = {leading_comb_cnt}")
        if out_num is None or out_num > len(leading_comb_cnt):
            out_num = len(leading_comb_cnt)

        return [i[0] for i in leading_comb_cnt.most_common(out_num)]

    def join_leading_priori(self, schema_total = 5, comb_set_num = 10, num_per_comb = 10, save_result = True):
        """
        连接顺序开始的先验

        Args:
            schema_total:
            comb_set_num:
            num_per_comb:
            save_result:
        Returns:
            query_list: 
            meta_list: 
            jo_str_list:
        """
        table_comb_set = self.search_initilizer.random_schema_subset(\
            schema_num=schema_total, target_num=comb_set_num)
        
        # print(f"join_leading_priori: table_comb_set = {table_comb_set}")
        query_list, meta_list, join_order_list = [], [], []

        for table_subset in table_comb_set:
            query_local, meta_local, join_order_local = \
                self.search_initilizer.join_order_priori(table_subset=table_subset, num = num_per_comb)
            query_list.extend(query_local)
            meta_list.extend(meta_local)
            join_order_list.extend(join_order_local)

        bushy_num, bushy_list, zigzag_num, zigzag_list = \
            self.search_initilizer.join_order_type_analysis(join_order_list=join_order_list)
        # leading_comb_set = set()
        jo_str_list = [jo_str[1] for jo_str in zigzag_list]

        # print("join_leading_priori: zigzag_list = {}.".format(zigzag_list))
        # print("join_leading_priori: jo_str_list = {}.".format(jo_str_list))
        if save_result == True:
            config = {
                "schema_total": schema_total,
                "comb_set_num": comb_set_num,
                "num_per_comb": num_per_comb
            }
            res_idx = self.save_join_order_result(config=config, query_list=query_list, \
                                        meta_list=meta_list, jo_str_list=jo_str_list)
        else:
            res_idx = self.get_next_res_idx()

        # 加载结果
        result = query_list, meta_list, jo_str_list
        self.result_dict[res_idx] = result
        return query_list, meta_list, jo_str_list
# %%

class ReferenceQueryEvaluator(object):
    """
    {Description}

    Members:
        field1:
        field2:
    """

    def __init__(self, workload, selected_tables = ["badges", "comments", "posthistory", "postlinks", \
            "posts", "tags", "users", "votes"], data_dir = "/home/lianyuan/Research/CE_Evaluator/intermediate/{}/query_obj",
            split_budget = 100):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        self.workload = workload
        self.data_dir = data_dir.format(self.workload)
        self.query_batch_dict = {}
        self.meta_path = p_join(self.data_dir, "meta_info.json")
        self.search_initilizer = plan_init.get_initializer_by_workload(\
            schema_list=selected_tables, workload=self.workload, split_budget=split_budget)
        self.workload_meta_dict = self.load_meta_info()

        # 
        self.inter_query_path = self.data_dir

        # 当前实例产生的查询
        self.current_result = {
            "query_list": [],
            "meta_list": [],
            "card_list": []
        }
        
    def get_current_result(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        return self.current_result['query_list'], \
            self.current_result['meta_list'], self.current_result['card_list']

    def put_current_result(self, query_list, meta_list, card_list):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        self.current_result['query_list'].extend(query_list)
        self.current_result['meta_list'].extend(meta_list)
        self.current_result['card_list'].extend(card_list)

    def get_total_query_number(self,):
        """
        获得当前总的查询数目
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        query_aggr, _, _ = self.result_aggregation(config={})
        return len(query_aggr)


    def build_workload_name(self, extra_info):
        """
        暂定的workload命名格式为, {类型}_{查询数目}_{时间}, 对于类型的细化配置中间用"#"来进行分隔
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        name_tmpl = "{type}_{num}_{timestamp}"
        curr_timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())

        return name_tmpl.format(type = extra_info['type'], 
                                num = extra_info['num'], timestamp=curr_timestamp)
    
    def show_meta_info(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        for k, v in self.workload_meta_dict.items():
            print(f"idx = {k}. value_dict = {v}.")
        return self.workload_meta_dict

    def load_meta_info(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        return utils.load_json(self.meta_path)

    def dump_meta_info(self,):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        utils.dump_json(self.workload_meta_dict, self.meta_path)

    def condition_filter(self, cond_dict):
        """
        根据cond_dict筛选合法的结果，cond_dict包含了以下几个部分
        {
            "specified_list": "具体的ID列表",
            "generation_type": "生成查询的模式",
            "table_num": "生成表的数目"
        }
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        wkld_idx_list = []

        for idx in self.workload_meta_dict.keys():
            flag = True
            if "specified_list" in cond_dict:
                flag = flag and (idx in cond_dict["specified_list"])

            if "generation_type" in cond_dict:
                valid_obj = cond_dict['generation_type']
                local_dict = self.workload_meta_dict[idx]
                if isinstance(valid_obj):
                    flag = flag and (local_dict['info']['type'] in valid_obj)
                else:
                    flag = flag and (local_dict['info']['type'] == valid_obj)
            if flag:
                wkld_idx_list.append(idx)

        return wkld_idx_list


    def one_step_generation(self, schema_comb_list, total_query_num):
        """
        单步结果的生成
    
        Args:
            schema_comb_list:
            total_query_num:
        Returns:
            result_dict: 
            delta_time:
        """
        pass


    def iterative_workload_generation(self, schema_comb_list, error_threshold, batch_comb_num, \
            batch_query_num, time_limit, retain_limit, total_min, total_max, ce_str, query_timeout = 10000):
        """
        {Description}
    
        Args:
            schema_comb_list:
            error_threshold:
            batch_comb_num:
            batch_query_num:
            time_limit:
            retain_limit:
            total_min:
            total_max: 
            ce_str:
        Returns:
            query_list: 
            meta_list: 
            label_list:
        """
        assert total_min < total_max, f"iterative_workload_generation: total_min = {total_min}. total_max = {total_max}."

        query_list, meta_list, label_list = [], [], []
        start_time = time.time()
        active_comb_set = dict()

        comb_queue = queue.Queue()
        ce_handler = ce_injection.get_ce_handler_by_name(workload=self.workload, ce_type=ce_str)

        def add_item(key):
            active_comb_set[key] = 0

        def delete_item(key):
            del active_comb_set[key]

        def update_item(key):
            active_comb_set[key] += 1

        if batch_comb_num > len(schema_comb_list):
            for key in schema_comb_list:
                add_item(key)
            mode = "stable"
        else:
            # 填充active_comb_set
            for key in schema_comb_list[:batch_comb_num]:
                add_item(key)

            # 填充comb_queue
            for key in schema_comb_list[batch_comb_num:]:
                comb_queue.put(key)
            
            mode = "dynamic"
        init = self.search_initilizer

        def error_func(input):
            # 
            true_card_list, est_card_list = input
            # lambda a, b: (max((i + 1) / (j + 1), (j + 1) / (i + 1)) for i, j in zip(a, b))
            if len(true_card_list) == 0:
                return 1.0
            
            error_list = [max((i + 1) / (j + 1), (j + 1) / (i + 1)) \
                for i, j in zip(true_card_list, est_card_list)]
            return max(error_list)
        
        def dict_aggregation(val_dict, key_order):
            # 将字典的每一个结果进行聚合
            return list(reduce(add, [val_dict[k] for k in key_order], []))

        while True:
            alias_set_curr = list(active_comb_set.keys())
            if mode == "stable":
                query_dict, meta_dict, label_dict = init.workload_generation_under_spec(\
                    alias_set_curr, batch_query_num, return_mode = "dict", timeout=query_timeout)
            elif mode == "dynamic":
                query_dict, meta_dict, label_dict = init.workload_generation_under_spec(\
                    alias_set_curr, batch_query_num, return_mode = "dict", timeout=query_timeout)

                est_dict = {k: ce_handler.get_cardinalities(v) for k, v in query_dict.items()}
                error_dict = utils.dict_apply(utils.dict_concatenate(label_dict, est_dict), \
                    operation_func=error_func)
                
                #
                for key, max_error in error_dict.items():
                    if max_error < error_threshold:
                        delete_item(key)
                        comb_queue.put(key)
                    else:
                        update_item(key)
                        if active_comb_set[key] > retain_limit:
                            delete_item(key)
                            comb_queue.put(key)

                # 
                add_num = batch_comb_num - len(active_comb_set)
                for _ in range(add_num):
                    key = comb_queue.get()
                    add_item(key)

            # 添加当前迭代的结果
            key_list = list(query_dict.keys())
            query_local, meta_local, label_local = dict_aggregation(query_dict, key_list), \
                dict_aggregation(meta_dict, key_list), dict_aggregation(label_dict, key_list)
            query_list.extend(query_local), meta_list.extend(meta_local), label_list.extend(label_local)

            end_time = time.time()
            if end_time - start_time > time_limit and len(query_list) > total_min:
                print(f"iterative_workload_generation: exceed time limit({time_limit}).")
                break

            if len(query_list) > total_max:
                print(f"iterative_workload_generation: exceed query number limit({total_max}).")
                break

        print(f"iterative_workload_generation: exploration finish. len(query_list) = {len(query_list)}.")
        
        # 保存结果
        extra_info = { "num": total_max, "type": "iterative" }
        self.save_workload(query_list=query_list, meta_list=meta_list, \
                           label_list=label_list, extra_info=extra_info)
        
        #
        self.put_current_result(query_list, meta_list, label_list)
        return query_list, meta_list, label_list

    def custom_workload_generation(self, config:dict, gen_mode = "normal"):
        """
        向TemplateManager中导入workload

        Args:
            config: 相关配置参数
            arg2:
        Returns:
            query_list: 
            meta_list: 
            label_list:
        """
        extra_info = {}

        if config['mode'] == "random":
            # table_num_dist = {2: 0.4, 3:0.6}        # 多表查询的具体比例
            table_num_dist = {3: 1.0}
            total_num = config.get("num", 200)        # 总的查询数目
            timeout = config.get("timeout", 10000)    # 查询超时限制，ms为单位
            query_list, meta_list, label_list = self.search_initilizer.workload_generation(\
                table_num_dist=table_num_dist, total_num=total_num, timeout=timeout, gen_mode=gen_mode)
            extra_info = {
                "num": total_num,
                "type": "random"
            }
        elif config['mode'] == "priori":
            # 基于join结果的先验
            # workload_spec_set = self.join_leading_priori()
            workload_spec_set = config['schema_comb_set']
            print("workload_spec_set = {}.".format(workload_spec_set))

            total_num = config.get("num", 200)          # 总的查询数目
            timeout = config.get("timeout", 10000)      # 查询超时限制，ms为单位

            query_list, meta_list, label_list = self.search_initilizer.workload_generation_under_spec(\
                alias_set_list=workload_spec_set, total_num=total_num, timeout=timeout, gen_mode=gen_mode)
            extra_info = {
                "num": total_num,
                "type": "priori"
            }
        else:
            raise ValueError("custom_workload_loading. Unsupported mode: {}.".format(config['mode']))

        print("custom_workload_generation: len(query_list) = {}. len(meta_list) = {}. len(label) = {}.".\
              format(len(query_list), len(meta_list), len(label_list)))

        # 保存结果
        self.save_workload(query_list=query_list, meta_list=meta_list, \
                           label_list=label_list, extra_info=extra_info)
        
        #
        self.put_current_result(query_list, meta_list, label_list)

        return query_list, meta_list, label_list

    def load_workload(self, wkld_id):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        result_path = self.workload_meta_dict[wkld_id]['path']
        result_dict = utils.load_pickle(result_path)
        self.query_batch_dict[wkld_id] = result_dict
        # print(f"load_workload: wkld_id = {wkld_id}. result_path = {result_path}.")
        # print("self.query_batch_dict.keys() = {}.".format(self.query_batch_dict[wkld_id].keys()))
        # return self.query_batch_dict[wkld_id]

        return result_dict['query_list'], result_dict['meta_list'], result_dict['label_list']

    def save_workload(self, query_list, meta_list, label_list, extra_info = {}):
        """
        保存结果workload
    
        Args:
            query_list: 
            meta_list: 
            label_list: 
            extra_info:
        Returns:
            result_dict:
            return2:
        """
        result_dict = {
            "query_list": query_list,
            "meta_list": meta_list,
            "label_list": label_list
        }
        wkld_name = "{}.pkl".format(self.build_workload_name(extra_info=extra_info))      # 负载文件name
        data_path = p_join(self.inter_query_path, wkld_name)
        print("ReferenceQueryEvaluator.save_workload: data_path = {}.".format(data_path))
        utils.dump_pickle(res_obj=result_dict, data_path=data_path)
        self.update_workload_meta(extra_info, data_path)
        return result_dict

    def update_workload_meta(self, extra_info, data_path):
        """
        更新元信息
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        if self.workload_meta_dict == {}:
            next_wkld_id = 0
        else:
            next_wkld_id = max([int(k) for k in self.workload_meta_dict.keys()]) + 1

        self.workload_meta_dict[next_wkld_id] = {
            "info": extra_info, 
            "path": data_path
        }
        # 导出结果
        utils.dump_json(self.workload_meta_dict, self.meta_path)


    def single_column_eval(self, in_meta):
        """
        {Description}
    
        Args:
            in_meta:
            arg2:
        Returns:
            flag:
            return2:
        """
        schema_list, filter_list = in_meta

        if len(schema_list) != len(filter_list):
            return False
        else:
            if len(set([(item[0], item[1]) for item in \
                filter_list])) == len(filter_list):
                return True
            else:
                return False


    def filter_valid_records(self, query_list, meta_list, card_list, min_card = 1000, single_column = True):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        query_res, meta_res, card_res = [], [], []

        for query, meta, card in zip(query_list, meta_list, card_list):
            if card >= min_card:
                if single_column:
                    if self.single_column_eval(in_meta=meta):
                        query_res.append(query)
                        meta_res.append(meta)
                        card_res.append(card)
                else:
                    query_res.append(query)
                    meta_res.append(meta)
                    card_res.append(card)
            

        return query_res, meta_res, card_res

    def result_aggregation(self, config):
        """
        {Description}
    
        Args:
            config:
            arg2:
        Returns:
            query_aggr: 
            meta_aggr: 
            card_aggr: 
        """
        wkld_idx_list = self.condition_filter(cond_dict=config)
        print("result_aggregation: wkld_idx_list = {}.".format(wkld_idx_list))
        query_aggr, meta_aggr, card_aggr = [], [], []

        for idx in wkld_idx_list:
            query_local, meta_local, card_local = self.load_workload(wkld_id=idx)
            # print("result_aggregation: idx = {}. len(query_local) = {}. len(meta_local) = {}. len(card_local) = {}".\
            #       format(idx, len(query_local), len(meta_local), len(card_local)))
            query_aggr.extend(query_local)
            meta_aggr.extend(meta_local)
            card_aggr.extend(card_local)

        return query_aggr, meta_aggr, card_aggr

    def construct_log_bins(self, bin_num, data_vec):
        """
        构造log scale的bins
        
        Args:
            bin_num:
            data_vec:
        Returns:
            bin_list:
        """
        min_val, max_val = np.min(data_vec), np.max(data_vec)
        left_bound, right_bound = np.log(min_val), np.log(max_val)
        idx_list = np.linspace(left_bound, right_bound, bin_num + 1)
        bin_list = np.exp(idx_list)

        return bin_list

    def accuracy_measure(self, ce_str: str, error_mode = "over-estimation", data_range = "history"):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        def plot_bin_figure(in_val, bin_num, log_scale = True):
            """
            {Description}
        
            Args:
                arg1:
                arg2:
            Returns:
                return1:
                return2:
            """
            curr_bins = self.construct_log_bins(bin_num, in_val)
            print(f"plot_bin_figure: curr_bins = {curr_bins}")
            plt.hist(in_val, bins=curr_bins, log=log_scale)
            plt.show()

        ce_handler = ce_injection.get_ce_handler_by_name(workload=self.workload, ce_type=ce_str)

        assert data_range in ("history", "current")
        if data_range == "history":
            query_list, meta_list, card_list = self.result_aggregation({})
        elif data_range == "current":
            query_list, meta_list, card_list = self.get_current_result()

        est_list = ce_handler.get_cardinalities(query_list)
        
        assert error_mode in ("over-estimation", "under-estimation"), \
            f"accuracy_measure: error_mode = (over-estimation, under-estimation). input = {error_mode}"

        if error_mode == "over-estimation":
            error_list = [j/i for i, j in zip(card_list, est_list)]
        elif error_mode == "under-estimation":
            error_list = [i/j for i, j in zip(card_list, est_list)]

        # 只保留有效的结果
        error_filtered = [item for item in error_list if item > 1.0]

        print("accuracy_measure: max = {:.2f}. top_5 = {:.2f}. 90th = {:.2f}. median = {:.2f}.".
              format(np.max(error_filtered), np.sort(error_filtered)[-5], np.quantile(error_filtered, 0.9), np.median(error_filtered)))

        plot_bin_figure(error_filtered, bin_num=20)
        return 


# %%
