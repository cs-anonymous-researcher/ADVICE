#!/usr/bin/env python
# -*- coding: utf-8 -*-

# %%
import numpy as np
from plan import plan_template
from utility.utils import list_index_batch
from utility import utils, common_config, workload_spec
from functools import partial
from data_interaction import mv_management

# %%

def calculate_card_dist(log_card1, log_card2):
    # 计算card距离
    return np.abs(log_card2 - log_card1)


def calculate_meta_dist(meta_info1, meta_info2, norm):
    # 计算meta的距离
    if norm == 1:
        return np.linalg.norm(meta_info1 - meta_info2, ord=1)
    elif norm == 2:
        return np.linalg.norm(meta_info1 - meta_info2, ord=2)
    elif norm == "inf":
        return np.linalg.norm(meta_info1 - meta_info2, ord=np.inf)


# %%

class ExternalCaseMatcher(object):
    """
    利用外部实例匹配，用于从另一个角度筛选candidate root
    一种策略是根据ref case从随机生成的case中选择较优的(优先实现，用card_dict计算距离)
    另一种策略是直接从ref case中构造最接近的case(之后考虑实现，可能得依靠meta的信息)

    Members:
        field1:
        field2:
    """

    def __init__(self, template_plan_ref: plan_template.TemplatePlan):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        self.template_plan = template_plan_ref
        self.grid_plan_mapping = {}     # key为grid_plan_id，value为grid_plan_meta
        self.ref_case_list = []
        self.signature_set = set()

    def add_new_case(self, query_meta, card_dict, p_error):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        sig_curr = mv_management.meta_key_repr(query_meta, workload=self.template_plan.workload)

        if sig_curr not in self.signature_set:
            self.signature_set.add(sig_curr)
            self.ref_case_list.append((query_meta, card_dict, p_error))

    def select_best_case(self,):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        best_index, best_error = 0, 0.0
        for idx, item in enumerate(self.ref_case_list):
            if item[2] > best_error:
                best_index, best_error = idx, item[2]
        return best_index
    

    def case_candidates_match(self, instance_list, out_num = 1):
        """
        找到和历史case收益最高最匹配的
    
        Args:
            instance_list:
            out_num:
        Returns:
            return1:
            return2:
        """
        result_index = []
        iter_num = min(out_num, len(self.ref_case_list))
        ref_case_num = len(self.ref_case_list)
        for iter_idx in range(iter_num):
            score_list = []
            best_index = self.select_best_case()        # 
            meta_ref, card_ref, error_ref = self.ref_case_list[best_index]
            sub_meta, subquery_sub, single_table_sub = self.get_subinfo(meta_ref, card_ref)

            for query_meta, (subquery_true, single_table_true) in instance_list:
                match_score = self.card_dict_distance(subquery_true, 
                    single_table_true, subquery_sub, single_table_sub)
                score_list.append(match_score)

            index_order = np.argsort(match_score)[::-1]
            self.ref_case_list.pop(best_index)
            for idx in index_order:
                if idx not in result_index:
                    result_index.append(idx)
                    break

        print(f"case_candidates_match: len(instance_list) = {len(instance_list)}. len(ref_case_list) = {ref_case_num}. "
              f"out_num = {out_num}. out_actual = {len(result_index)}.")
        
        return result_index
        # return instance_list[index_order[0]]
        
    def delete_useless_cases(self, error_threshold):
        """
        删除无用的cases
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        filtered_list = [item for item in \
            self.ref_case_list if item[-1] > error_threshold]
        self.ref_case_list = filtered_list
        return filtered_list

    def select_target_template(self, root_meta):
        """
        找到root_meta最近的grid_plan，作为可能的探索起点
    
        Args:
            root_meta:
            arg2:
        Returns:
            return1:
            return2:
        """
        raise NotImplementedError("select_target_template")

    
    def card_dict_distance(self, subquery_dict1: dict, single_table_dict1: dict, subquery_dict2: dict, single_table_dict2: dict):
        """
        计算两个card_dict间的距离
    
        Args:
            subquery_dict1:
            single_table_dict1:
            subquery_dict2:
            single_table_dict2:
        Returns:
            return1:
            return2:
        """
        assert subquery_dict1.keys() == subquery_dict2.keys()
        assert single_table_dict1.keys() == single_table_dict2.keys()

        factor = 1000
        def dist_func(card1, card2):
            card1 += factor
            card2 += factor
            return np.log(max(card1 / card2, card2 / card1))

        subquery_keys = subquery_dict1.keys()
        single_table_keys = single_table_dict1.keys()
        total_dist = 0.0

        for k in subquery_keys:
            total_dist += dist_func(subquery_dict1[k], subquery_dict2[k])

        for k in single_table_keys:
            total_dist += dist_func(single_table_dict1[k], single_table_dict2[k])

        return total_dist

    def get_subinfo(self, query_meta: tuple, card_dict: dict):
        """
        {Description}
        
        Args:
            query_meta:
            card_dict:
        Returns:
            sub_meta:
            subquery_sub:
            single_table_sub:
        """
        template_meta = self.template_plan.query_meta
        schema_list, filter_list = query_meta
        schema_sub, filter_sub = template_meta[0], []
        assert set(schema_sub).issubset(schema_list)

        alias_mapping = workload_spec.abbr_option[self.template_plan.workload]
        alias_set = set([alias_mapping[s] for s in schema_sub])

        for item in filter_list:
            alias_name, column_name, start_val, end_val = item
            if alias_name in alias_set:
                filter_sub.append(item)

        sub_meta = schema_sub, filter_sub
        # sub_meta, sub_dict = None, {}
        subquery_true, single_table_true, subquery_estimation, \
            single_table_estimation = utils.extract_card_info(card_dict)

        subquery_sub, single_table_sub = {}, {}

        for k, v in subquery_true.items():
            if set(k).issubset(alias_set):
                subquery_sub[k] = v

        for k, v in single_table_true.items():
            if k in alias_set:
                single_table_sub[k] = v

        return sub_meta, subquery_sub, single_table_sub

    # def find_cloest_instance(self, query_meta, card_dict):
    #     """
    #     找到对应template下最接近的root_meta

    #     Args:
    #         query_meta:
    #         arg2:
    #     Returns:
    #         return1:
    #         return2:
    #     """
    #     ref_meta,  = None
    #     out_meta, card_dict = None, {}
    #     return out_meta, card_dict
    

# %%

class RootSelector(object):
    """
    根据sample的结果根节点选择器

    Members:
        field1:
        field2:
    """

    def __init__(self, grid_plan_ref: plan_template.GridPlan, num_limit: int = None, error_threshold: float = None):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        self.grid_plan = grid_plan_ref
        self.workload = grid_plan_ref.workload
        self.alias_inverse = {v: k for k, v in utils.abbr_option[self.workload].items()}

        if num_limit is None:
            self.num_limit = common_config.num_limit
        if error_threshold is None:
            self.error_threshold = common_config.error_threshold

        self.column_order, self.column_info = grid_plan_ref.construct_column_elements()

        print(f"RootSelector: workload = {self.workload}. column_order = {self.column_order}. num_limit = {self.num_limit}.")
        self.order_dict = {col: idx for idx, col in enumerate(self.column_order)}
    
    def card_item_func(self, meta, true_card, error):
        # 定义一个查询对应的card item
        return np.log(true_card)

    def idx2val(self, in_idx, column):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        bins_list = self.column_info[column]['bins_list']
        # slot_num = len(bins_list) - 1
        slot_num = len(bins_list) - 1
        # print(f"idx2val: slot_num = {slot_num}. in_idx = {in_idx}.")
        return (1.0 * in_idx / slot_num) + (0.5 / slot_num)
    
    def meta_item_func(self, meta, true_card, error):
        # 定义一个查询对应的meta item
        schema_list, filter_list = meta
        assert len(filter_list) == len(self.column_order), \
            f"meta_item_func: filter_list = {len(filter_list)}. column_order = {len(self.column_order)}."

        position_vector = [None for _ in filter_list]
        for alias_name, col_name, start_val, end_val in filter_list:
            tbl_name = self.alias_inverse[alias_name] # 
            order_idx = self.order_dict[(tbl_name, col_name)]
            
            reverse_dict = self.column_info[(tbl_name, col_name)]["reverse_dict"]
            start_idx, end_idx = utils.predicate_location(reverse_dict, start_val, end_val)
            center_idx = (start_idx + end_idx) / 2
            position_vector[order_idx] = self.idx2val(center_idx, (tbl_name, col_name))

        # return position_vector
        # 2024-03-11: 返回np.array类型
        return np.array(position_vector)
    
    
    def random_split(self, query_list, meta_list, true_card_list, error_list):
        """
        在一定范围里随机选择查询
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        selected_idx = list(np.where(np.array(error_list) > self.error_threshold)[0])

        print(f"random_split: selected_idx = {selected_idx}. threshold = "\
              f"{self.error_threshold}. \nerror_list = {utils.list_round(error_list, 2)}.")
        
        # query_filter, meta_filter, true_card_filter, error_filter = \
        #     list_index(query_list, selected_idx), list_index(meta_list, selected_idx),\
        #     list_index(true_card_list, selected_idx), list_index(error_list, selected_idx)

        if len(selected_idx) > 0:
            query_filter, meta_filter, true_card_filter, error_filter = \
                list_index_batch([query_list, meta_list, true_card_list, error_list], selected_idx)
            
            random_idx = np.random.choice(range(len(selected_idx)), self.num_limit)
            query_candidates, meta_candidates, error_candidates = \
                list_index_batch([query_filter, meta_filter, error_filter], random_idx)
            
        else:
            max_idx = np.argmax(error_list)
            query_candidates, meta_candidates, error_candidates = \
                list_index_batch([query_list, meta_list, error_list], [max_idx, ])
        
        return query_candidates, meta_candidates, error_candidates

    def naive_greedy_split(self, query_list, meta_list, true_card_list, error_list):
        """
        直接选择topk error的查询
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        # error_order = sorted(error_list, reverse=True)
        error_order = np.argsort(error_list)[::-1]
        selected_idx = error_order[:self.num_limit]

        query_candidates, meta_candidates, error_candidates = \
            list_index_batch([query_list, meta_list, error_list], selected_idx)
        
        return query_candidates, meta_candidates, error_candidates
    
    def distance_based_split(self, query_list, meta_list, true_card_list, error_list, item_func, dist_func):
        """
        基于距离的样本选择方法
    
        Args:
            query_list: 
            meta_list: 
            true_card_list: 
            error_list:
            item_func:
            dist_func:
        Returns:
            query_candidates:
            meta_candidates:
            error_candidates:
        """
        assert len(query_list) == len(meta_list) == len(true_card_list) == len(error_list)
        # 将相关信息转成用于距离计算的对象
        item_list = [item_func(meta, true_card, error) for (meta, true_card, error) \
                     in zip(meta_list, true_card_list, error_list)]

        query_candidates, meta_candidates, error_candidates = [], [], []
        idx_list = range(0, len(query_list))

        def select_best_case():
            # 
            pair_list = [(idx, error_list[idx]) for idx in idx_list]
            pair_list.sort(key=lambda a: a[1], reverse=True)
            return pair_list[0]

        def remove_close_cases(ref_idx):
            #
            idx_filtered = []
            for idx in idx_list:
                if dist_func(item_list[ref_idx], item_list[idx]) == True:
                    idx_filtered.append(idx)

            return idx_filtered

        for iter_num in range(self.num_limit):
            best_idx, best_error = select_best_case()
            if best_error < self.error_threshold:
                # 
                break
            
            idx_list = remove_close_cases(best_idx)

            query_candidates.append(query_list[best_idx])
            meta_candidates.append(meta_list[best_idx])
            error_candidates.append(error_list[best_idx])

            if len(idx_list) == 0:
                break
        
        if len(query_candidates) == 0:
            # 2024-03-16: 没有出现candidates，输出原因
            if iter_num == 0 and best_error < self.error_threshold:
                # 
                print(f"RootSelector.distance_based_split: no candidate case. best_error = {best_error:.2f}. error_threshold = {self.error_threshold:.2f}.")
            else:
                raise ValueError("RootSelector.distance_based_split: Unexpected case. "\
                    f"iter_num = {iter_num}. best_error = {best_error:.2f}. error_threshold = {self.error_threshold:.2f}.")
            
        return query_candidates, meta_candidates, error_candidates

    def card_based_split(self, query_list, meta_list, true_card_list, error_list, split_distance = 2.3026):
        """
        使用distance_based_split来改写函数

        Args:
            query_list:
            meta_list:
            true_card_list:
            error_list:
        Returns:
            query_candidates:
            meta_candidates:
            error_candidates:
        """
        assert len(query_list) == len(meta_list) == len(true_card_list) == len(error_list)

        def dist_func(item1, item2):
            card_res = calculate_card_dist(item1, item2)
            if card_res > split_distance:
                return True
            else:
                return False
        
        query_candidates, meta_candidates, error_candidates = \
            self.distance_based_split(query_list, meta_list, true_card_list, 
                error_list, self.card_item_func, dist_func)

        return query_candidates, meta_candidates, error_candidates

    # def card_based_split(self, query_list, meta_list, true_card_list, error_list, split_distance = 2.3026):
    #     """
    #     {Description}

    #     Args:
    #         query_list:
    #         meta_list:
    #         true_card_list:
    #         error_list:
    #     Returns:
    #         query_candidates:
    #         meta_candidates:
    #         error_candidates:
    #     """
    #     assert len(query_list) == len(meta_list) == len(true_card_list) == len(error_list)
    #     log_card_list = np.log(true_card_list)
    #     query_candidates, meta_candidates, error_candidates = [], [], []
    #     idx_list = range(0, len(query_list))

    #     def select_best_case():
    #         # 
    #         pair_list = [(idx, error_list[idx]) for idx in idx_list]
    #         pair_list.sort(key=lambda a: a[1], reverse=True)
    #         return pair_list[0]

    #     def remove_close_cases(target_log_card):
    #         #
    #         idx_filtered = []
    #         for idx in idx_list:
    #             if np.isclose(log_card_list[idx], \
    #                 target_log_card, atol=split_distance) == False:
    #                 idx_filtered.append(idx)
    #         return idx_filtered


    #     for _ in range(self.num_limit):
    #         best_idx, best_error = select_best_case()
    #         # print(f"card_based_split: best_idx = {best_idx}. best_error = {best_error:.2f}.")

    #         if best_error < self.error_threshold:
    #             #
    #             break
            
    #         best_log_card = log_card_list[best_idx]
    #         idx_list = remove_close_cases(best_log_card)

    #         query_candidates.append(query_list[best_idx])
    #         meta_candidates.append(meta_list[best_idx])
    #         error_candidates.append(error_list[best_idx])

    #         if len(idx_list) == 0:
    #             break

    #     return query_candidates, meta_candidates, error_candidates
    

    def meta_based_split(self, query_list, meta_list, true_card_list, error_list, split_distance = 0.1):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        def dist_func(item1, item2):
            meta_res = calculate_meta_dist(item1, item2, norm=1)
            if meta_res > split_distance:
                return True
            else:
                return False
            
        # query_candidates, meta_candidates, error_candidates = self.distance_based_split(
        #     query_list, meta_list, true_card_list, error_list, 
        #     self.meta_item_func, partial(calculate_meta_dist, norm=1))
        
        query_candidates, meta_candidates, error_candidates = self.distance_based_split(
            query_list, meta_list, true_card_list, error_list, 
            self.meta_item_func, dist_func)

        return query_candidates, meta_candidates, error_candidates

    def hybrid_split(self, query_list, meta_list, true_card_list, error_list, 
            card_distance = 2.3026, meta_distance = 0.1, alpha = 0.5):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        def item_func(meta, true_card, error):
            card_info = self.card_item_func(meta, true_card, error)
            meta_info = self.meta_item_func(meta, true_card, error)

            return card_info, meta_info

        def dist_func(item1, item2):
            # 计算两个item之间的距离
            card1, meta1 = item1
            card2, meta2 = item2

            card_res = calculate_card_dist(card1, card2)
            meta_res = calculate_meta_dist(meta1, meta2, norm=1)

            card_score = card_res / card_distance
            meta_score = meta_res / meta_distance

            return (card_score * alpha + (1 - alpha) * meta_score) >= 1.0

        query_candidates, meta_candidates, error_candidates = self.distance_based_split(\
            query_list, meta_list, true_card_list, error_list, item_func, dist_func)
        
        return query_candidates, meta_candidates, error_candidates
    
