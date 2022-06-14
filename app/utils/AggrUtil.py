class AggrUtil:
    @classmethod
    def add_other_buckets(cls, infos, nodes, other_flag=False):
        error_doc = infos.get('doc_count_error_upper_bound')
        other_doc = infos.get('sum_other_doc_count')
        if other_flag:
            if error_doc:
                nodes.append({'name': 'error', 'value': error_doc})
            if other_doc:
                nodes.append({'name': 'other', 'value': other_doc})

    @classmethod
    def handle_aggr_result(cls, result, group_fields, value_fields=None, other_flag=False):
        result_map = {}
        aggregations = result.get('aggregations')
        # 针对每个聚合字段查看聚合结果
        for group_field in group_fields:
            # 字段名
            field_name = group_field.get('name')
            field_type = group_field.get('type')
            # 聚合的结果信息
            infos = aggregations.get(field_name + "_aggr")
            if field_type == 'value_count':
                result_map[field_name] = infos['value']
            else:
                result_nodes = []
                # 其它信息计数
                cls.add_other_buckets(infos, result_nodes, other_flag)
                buckets = infos.get('buckets')
                # 查看是否有结果
                if buckets and len(buckets) > 0:
                    # 结果信息遍历组合
                    for bucket in buckets:
                        cur_node = {'name': bucket.get('key'), 'value': bucket.get('doc_count')}
                        cls.add_other_value_field_value(cur_node, bucket, value_fields)
                        # 查看是否有子字段聚合
                        child_group_fields = group_field.get('children') or []
                        if child_group_fields and len(child_group_fields) > 0:  # 存在子字段聚合
                            # 针对父节点增加名为’child_value_map‘的字典数据容器，存储各个子聚合的数据
                            cur_node['child_value_map'] = {}
                            # 遍历子聚合字段
                            for child_group_field in child_group_fields:
                                # 子聚合字段名称
                                child_field_name = child_group_field.get('name')

                                infos_2 = bucket.get(child_field_name + "_aggr")
                                # 获取结果信息
                                cur_child_nodes = cls.build_result_to_nodes(infos_2, child_field_name,
                                                                            value_fields=value_fields,
                                                                            other_flag=other_flag)
                                # 以字段名称为key的属性
                                cur_node['child_value_map'][child_field_name] = cur_child_nodes
                        result_nodes.append(cur_node)
                result_map[field_name] = result_nodes
        return result_map

    @classmethod
    def add_other_value_field_value(cls, cur_node, bucket, value_fields=None):
        # 查看是否有其它值信息字段
        if value_fields and len(value_fields) > 0:
            for value_field in value_fields:
                aggr_name = value_field + '_aggr'
                if bucket.get(aggr_name):
                    cur_node[value_field] = bucket[aggr_name]['value']

    @classmethod
    def build_result_to_nodes(cls, infos, field_name=None, value_fields=None, other_flag=False):
        nodes = []
        cls.add_other_buckets(infos, nodes, other_flag)
        buckets = infos.get('buckets')
        if buckets and len(buckets) > 0:
            for bucket in buckets:
                name = bucket.get('key')
                cur_node = {'name': name, 'value': bucket.get('doc_count')}
                nodes.append(cur_node)
                cls.add_other_value_field_value(cur_node, bucket, value_fields)
        return nodes
