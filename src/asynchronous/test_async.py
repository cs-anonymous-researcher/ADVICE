#!/usr/bin/env python
# -*- coding: utf-8 -*-


# %%

from utility import generator, workload_spec
from plan import node_query, node_extension, plan_init
from query import query_exploration, ce_injection
from data_interaction import data_management, mv_management
from grid_manipulation import grid_preprocess

# %%
class AsyncTestor(object):
    """
    {Description}

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
        schema_total = workload_spec.total_schema_dict[workload]
        self.search_initializer = plan_init.get_initializer_by_workload(\
            schema_list=schema_total, workload=workload)
        

        # 优化query instance的创建
        data_manager = data_management.DataManager(wkld_name=workload)
        mv_manager = mv_management.MaterializedViewManager(workload=workload)
        ce_handler = ce_injection.PGInternalHandler(workload=workload)
        multi_builder = grid_preprocess.MultiTableBuilder(workload = workload, \
            data_manager_ref = data_manager, mv_manager_ref = mv_manager)
        query_ctrl = query_exploration.QueryController(workload=workload)

        self.external_dict = {
            "data_manager": data_manager,
            "mv_manager": mv_manager,
            "ce_handler": ce_handler,
            "multi_builder": multi_builder,
            "query_ctrl": query_ctrl
        }


    def generate_single_test_instance(self, table_num):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            query_text:
            query_meta: 
            extension_instance:
        """
        search_initializer = self.search_initializer
        workload = self.workload

        schema_comb = search_initializer.random_schema_subset(schema_num = table_num, target_num = 1)[0]
        query_text, query_meta = search_initializer.single_query_generation(schema_subset=schema_comb)

        query_instance = node_query.get_query_instance(workload=workload, \
                query_meta=query_meta, external_dict=self.external_dict)
        
        subquery_dict, single_table_dict = query_instance.estimation_card_dict, query_instance.estimation_single_table

        query_ctrl = query_exploration.QueryController(workload=workload)
        extension_instance: node_extension.ExtensionInstance = node_extension.get_extension_instance(\
            workload=workload, query_text=query_text, query_meta=query_meta, subquery_dict=subquery_dict, 
            single_table_dict=single_table_dict, query_ctrl=query_ctrl)
        
        return query_text, query_meta, extension_instance

    def generate_batch_test_instances(self, table_num, instance_num):
        """
        {Description}

        Args:
            arg1:
            arg2:
        Returns:
            result_list: element(query_text, query_meta, extension_instance)
        """
        return [self.generate_single_test_instance(table_num) for _ in range(instance_num)]
    
# %%
