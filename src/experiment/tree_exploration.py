#!/usr/bin/env python
# -*- coding: utf-8 -*-


# %%
from plan import stateful_search, advance_search, node_query, node_extension
from utility import utils, workload_spec

# %%
dummy_card_dict = {
    "true": {
        "subquery": {},
        "single_table": {}
    },
    "estimation": {
        "subquery": {},
        "single_table": {}
    }
}


class SingleTreeExplorer(object):
    """
    针对单棵树的收益分析

    Members:
        field1:
        field2:
    """

    def __init__(self, workload):
        """
        {Description}

        Args:
            arg1:
            arg2:
        """
        self.workload = workload
        self.selected_tables = workload_spec.total_schema_dict[workload]

    def create_instance(self, in_meta, card_dict = dummy_card_dict, tree_type = "advance"):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        root_query = node_query.get_query_instance(\
            workload=self.workload, query_meta=in_meta, \
            ce_handler = "DeepDB_jct")
        
        subquery_true, single_table_true, subquery_estimation, \
            single_table_estimation = utils.extract_card_info(card_dict=card_dict)
        
        root_query.add_true_card(subquery_true, mode="subquery")
        root_query.add_true_card(single_table_true, mode="single_table")

        root_query.add_estimation_card(subquery_estimation, mode="subquery")
        root_query.add_estimation_card(single_table_estimation, mode="single_table")

        root_query.complement_true_card(time_limit=60000)  # 补全真实基数
        external_info = {
            "query_instance": root_query,
            "selected_tables": self.selected_tables,
            "max_depth": 6,     # 最大深度hard-code进去,
            "timeout": 12000          # 查询时间限制在1min
        }
        max_step = 5

        assert tree_type in ("advance", "exploration", "stateful")

        if tree_type == "advance":
            search_tree = advance_search.AdvanceTree(external_info, max_step, mode="under-estimation", init_strategy = "multi-loop")
            # search_tree = advance_search.AdvanceTree(external_info, max_step, mode="under-estimation", init_strategy = "random")
        elif tree_type == "stateful":
            search_tree = stateful_search.StatefulTree(self.workload, external_info, \
                max_step, mode="under-estimation", init_strategy = "multi-loop")

        self.search_tree = search_tree
        return search_tree

    def dump_card_dict(self, out_name = "card_dict.pkl"):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        inst = self.search_tree.root.query_instance
        out_card_dict = utils.pack_card_info(inst.true_card_dict, \
            inst.true_single_table, inst.estimation_card_dict, inst.estimation_single_table)
        
        utils.dump_pickle(out_card_dict, out_name)

    def extend_new_table(self,):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            valid_flag:
            benefit:
            node:
        """
        node_signature, estimate_benefit, new_node = \
            self.search_tree.one_step_search(sync=True)
        if node_signature != "":
            return True, estimate_benefit, new_node
        else:
            return False, estimate_benefit, new_node
        
        
# %%
