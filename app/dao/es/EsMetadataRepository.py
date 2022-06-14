from app.common.EsRepositoryInterface import EsRepositoryInterface
import json
import traceback
from json import JSONDecodeError
from config.constant import work_path
from elasticsearch import helpers
from app.utils import AggrUtil
import copy


class EsMetadataRepository(EsRepositoryInterface):
    def __init__(self):
        self.index_name = 'metadata'

    def create_index(self):
        mapping = None
        with open(work_path + '/app/mapping/es/metadata_mapping.json', 'r', encoding='utf8') as js:
            try:
                mapping = json.load(js)
            except JSONDecodeError:
                traceback.print_exc()
        self._create_index(mapping, self.index_name)

    def get_page(self, page=1, size=10):
        if page <= 0 or page > 10000:
            page = 1
        if size <= 0 or size > 100:
            size = 10
        from_ = (page - 1) * size

        match = {
            "match_all": {}
        }
        query = {
            'query': match,
            # 'sort': [
            #     {
            #         'generate_date': {
            #             'order': 'desc'
            #         }
            #     }
            # ]
        }
        total, list_ = self._get_list(self.index_name, size, query_str=json.dumps(query), from_num=from_)
        return {
            'total': total,
            'records': list_
        }

    def get_query_page(self, page=1, size=10, field=None, field_value=None):
        if page <= 0 or page > 10000:
            page = 1
        if size <= 0 or size > 100:
            size = 10
        from_ = (page - 1) * size
        if field and field_value:
            if field == '0':
                match = {
                    "multi_match": {
                        'query': field_value,
                        'fields': ['name', 'describe', 'keyword']
                    }
                }
            else:
                match = {
                    "match": {
                        field: field_value
                    }
                }
        else:
            match = {
                "match_all": {}
            }
        query = {
            'query': match,
            # 'sort': [
            #     {
            #         'generate_date': {
            #             'order': 'desc'
            #         }
            #     }
            # ]
        }
        total, list_ = self._get_list(self.index_name, size, query_str=json.dumps(query), from_num=from_)
        return {
            'total': total,
            'records': list_
        }

    def get_multi_query_page(self, page=1, size=10, word=None):
        if page <= 0 or page > 10000:
            page = 1
        if size <= 0 or size > 100:
            size = 10
        from_ = (page - 1) * size
        if word:
            match = {
                "multi_match": {
                    'query': word,
                    'fields': ['name', 'describe', 'keyword']
                }
            }
        else:
            match = {
                "match_all": {}
            }
        query = {
            'query': match,
            # 'sort': [
            #     {
            #         'generate_date': {
            #             'order': 'desc'
            #         }
            #     }
            # ]
        }
        total, list_ = self._get_list(self.index_name, size, query_str=json.dumps(query), from_num=from_)
        return {
            'total': total,
            'records': list_
        }

    def update_batch(self, data_list):
        data = [
            {
                '_op_type': 'update',
                "_index": self.index_name,
                "doc": value,
                "_id": value['id']
            } for value in data_list
        ]
        helpers.bulk(self.es, data)

    def count(self):
        return self._count(self.index_name)

    def get_by_id(self, doc_id):
        return self._get_by_id(self.index_name, doc_id)

    def save_batch(self, data_list):
        self._add_list(data_list, self.index_name)

    def save(self, source):
        self._add_doc(source, self.index_name)

    def group_by_field(self, group_fields, query_info=None):
        return self._group_by_field(self.index_name, group_fields, query_info=query_info, other_flag=True)

    def group_and_count(self, group_fields, query_info=None, other_flag=True):
        if not query_info:
            query_info = {
                "match_all": {}
            }

        aggr_info = {}
        for group_field in group_fields:
            field_name = group_field.get('name')
            size = group_field.get('size') or 20
            aggr_info[field_name + "_aggr"] = {
                "terms": {
                    "field": field_name,
                    "size": size
                }
            }
            child_group_fields = group_field.get('children') or []
            if child_group_fields and len(child_group_fields) > 0:
                aggr_info[field_name + "_aggr"]['aggs'] = {}
                for child_group_field in child_group_fields:
                    child_field_name = child_group_field.get('name')
                    child_size = child_group_field.get('size') or 20
                    aggr_info[field_name + "_aggr"]['aggs'][child_field_name + "_aggr"] = {
                        "terms": {
                            "field": child_field_name,
                            "size": child_size
                        }
                    }
        query = {
            "query": query_info,
            "size": 0,
            "aggs": aggr_info
        }
        result = self.es.search(index=self.index_name, body=json.dumps(query))
        return AggrUtil.handle_aggr_result(result, group_fields, other_flag=other_flag)
