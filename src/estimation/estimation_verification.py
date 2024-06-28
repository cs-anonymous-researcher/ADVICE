#!/usr/bin/env python
# -*- coding: utf-8 -*-


# %%

from comparison import result_in_depth
from utility import utils
from result_analysis import case_analysis
from data_interaction import mv_management
from pprint import pprint
# %%

class EstimationVerifier(object):
    """
    {Description}

    Members:
        field1:
        field2:
    """
    def __init__(self, workload, ce_type, intermediate_dir="/home/lianyuan/Research/CE_Evaluator/intermediate", 
            result_dir = "/home/lianyuan/Research/CE_Evaluator/result"):
        """
        {Description}

        Args:
            workload:
            ce_type:
            intermediate_dir:
            result_dir:
        """
        self.workload, self.ce_type = workload, ce_type
        self.instance_list = []
        self.card_analyzer = result_in_depth.EstCardMutationAnalyzer(workload = workload, \
            ce_str = ce_type, intermediate_dir=intermediate_dir, result_dir = result_dir)

    def dump_instance_list(self, out_path):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        utils.dump_pickle(self.instance_list, out_path)


    def load_instance_list(self, out_path, drop_duplicate = True):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        self.instance_list = utils.load_pickle(out_path)
        # 
        if drop_duplicate == True:
            self.drop_duplicate_instances()

    def drop_duplicate_instances(self, ):
        """
        把重复meta的case删除，提升验证的效率
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        meta_signature_set = set()
        self.instance_list.sort(key=lambda a: a[4], reverse=True)

        instance_filtered = []

        for query_text, query_meta, card_dict, prefix_num, expected_error in self.instance_list:
            curr_signature = mv_management.meta_key_repr(query_meta, workload=self.workload)
            if curr_signature not in meta_signature_set:
                meta_signature_set.add(curr_signature)
                instance_filtered.append((query_text, query_meta, card_dict, prefix_num, expected_error))
            else:
                continue
        
        print(f"drop_duplicate_instances: filter_before = {len(self.instance_list)}. filter_after = {len(instance_filtered)}.")
        self.instance_list = instance_filtered
        return instance_filtered

    def add_new_instance(self, query_text, query_meta, card_dict, prefix_num, expected_error):
        """
        {Description}
    
        Args:
            query_text: 
            query_meta: 
            card_dict: 
            expected_error:
        Returns:
            return1:
            return2:
        """
        self.instance_list.append((query_text, query_meta, card_dict, prefix_num, expected_error))

    
    def select_instance_subset(self, topk = 5):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        num_limit = min(topk, len(self.instance_list))
        instance_sorted = sorted(self.instance_list, key=lambda a: a[4], reverse=True)[:num_limit]

        return instance_sorted

    def verify_instance_subset(self, topk = 5):
        """
        {Description}
        
        Args:
            topk:
            arg2:
        Returns:
            result_list(error pair):
            meta_list:
            pesudo_card_dict_list:
            actual_card_dict_list:
        """
        instance_sorted = self.select_instance_subset(topk)
        result_list = []
        card_analyzer = self.card_analyzer

        # 估计的error值列表
        card_old_list = [item[2] for item in instance_sorted]
        est_error_list = [item[4] for item in instance_sorted]
        dummy_result = (1.0, 1.0, 1.0)
        for query_text, query_meta, card_dict, prefix_num, expected_error in instance_sorted:
            card_analyzer.add_instance_external(query_text, query_meta, card_dict, dummy_result)

        #
        est_card_list = card_analyzer.construct_new_estimation(mode="list", true_card=False, update_member=False)
        true_card_list = card_analyzer.construct_new_estimation(mode="list", true_card=True, update_member=False)

        #
        card_analyzer.card_dict_list = card_analyzer.replace_old_cards(\
            card_analyzer.card_dict_list, est_card_list, mode="estimation")
        card_analyzer.card_dict_list = card_analyzer.replace_old_cards(\
            card_analyzer.card_dict_list, true_card_list, mode="true")

        true_error_list = []
        idx = 0
        for query, meta, card_dict, card_old in zip(card_analyzer.query_list, \
                card_analyzer.meta_list, card_analyzer.card_dict_list, card_old_list):
            idx += 1
            analyzer_local = case_analysis.CaseAnalyzer(query, meta, (), card_dict, self.workload)
            analyzer_old = case_analysis.CaseAnalyzer(query, meta, (), card_old, self.workload)

            print("under new card")
            print("plan_true")
            print(analyzer_local.true_cost)
            print(analyzer_local.true_physical.show_plan())
            print("plan_estimation")
            print(analyzer_local.estimation_cost)
            print(analyzer_local.estimation_physical.show_plan())
            print("\nunder old card")
            print("plan_true")
            print(analyzer_old.true_cost)
            print(analyzer_old.true_physical.show_plan())
            print("plan_estimation")
            print(analyzer_old.estimation_cost)
            print(analyzer_old.estimation_physical.show_plan())
            print(f"verify_instance_subset: new_error = {analyzer_local.p_error:.2f}. old_error = {analyzer_old.p_error:.2f}.")
            print("\n")
            true_error_list.append(analyzer_local.p_error)

            result_old = analyzer_old.plot_plan_comparison()
            result_new = analyzer_local.plot_plan_comparison()

            result_old[0].render(f"./image/{idx}_old_out_true")
            result_old[1].render(f"./image/{idx}_old_out_est")

            result_new[0].render(f"./image/{idx}_new_out_true")
            result_new[1].render(f"./image/{idx}_new_out_est")

        est_error_list = utils.list_round(est_error_list)
        true_error_list = utils.list_round(true_error_list)

        for idx, (card_dict_new, card_dict_old) in enumerate(zip(card_analyzer.card_dict_list, card_old_list)):
            print(f"idx = {idx}.\n card_new = {pprint(card_dict_new)}.\n card_old = {pprint(card_dict_old)}.")

        result_list = list(zip(est_error_list, true_error_list))
        return result_list, card_analyzer.meta_list, card_old_list, card_analyzer.card_dict_list

# %%
global_verifier = None
save_estimation = False

def init_verifier(workload, ce_type):
    global global_verifier
    global_verifier = EstimationVerifier(workload, ce_type)

# %%
