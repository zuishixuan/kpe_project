import os
from elasticsearch import Elasticsearch, helpers, NotFoundError
import json
from config.config import settings
from time import *
from logger import logger
import json
from app.utils import AggrUtil

hosts_key = 'hosts'
elasticsearch_key = 'elasticsearch'


class EsRepositoryInterface:
    es_settings = settings.get(elasticsearch_key)
    if es_settings is None or es_settings[hosts_key] is None:
        raise Exception("'请在setting文件配置elasticsearch服务器host信息'")
    hosts = es_settings[hosts_key]
    username = es_settings.get('username')
    if username:
        password = es_settings.get('password')
        es = Elasticsearch(hosts, timeout=30,
                           http_auth=(username, password),
                           max_retries=10, retry_on_timeout=True)
    else:
        es = Elasticsearch(hosts, timeout=30, max_retries=10, retry_on_timeout=True)
    parent_path = os.path.dirname(__file__)

    def _create_index(self, mapping, index_name):
        if mapping is not None:
            self.es.indices.create(index=index_name, body=mapping, ignore=400)
        else:
            self.es.indices.create(index=index_name, ignore=400)

    """
    添加单个索引
    """

    def _delete_index(self, index_name):
        self.es.indices.delete(index_name)

    def _add_doc(self, source, index_name):
        self.es.index(index=index_name, document=source, id=source['id'])

    def _add_list(self, data_list, index_name, batch_size=1000):
        i = 1
        count = len(data_list)
        if count < 1:
            logger.warn('data is empty!')
        else:
            # 计算分页大小
            is_zero = count % batch_size
            pages = int(count / batch_size)
            if is_zero != 0:
                pages = pages + 1
            begin_time = time()
            while i <= pages:
                start = (i - 1) * batch_size
                end = i * batch_size
                cur_list = data_list[start:end]
                data = [
                    {"_index": index_name,
                     "_source": value,
                     "_id": value['id']
                     } for value in cur_list
                ]
                helpers.bulk(self.es, data)
                i += 1
            logger.info('================ 全部数据共[%s]条。用时为 [%s] ================' % (count, (time() - begin_time)))

    def _add_list_by_query(self, start, total, func, index_name):
        batch_size = 1000
        i = start
        cnt = total + i
        while i <= cnt:
            json_result = func(i, batch_size)
            data = [
                {"_index": index_name,
                 "_source": value,
                 "_id": value['id']
                 } for value in json_result
            ]
            helpers.bulk(self.es, data)
            i += 1

    """
     全量更新信息
    """

    def full_update(self, count_func, list_func, index_name, start_index=1, batch_size=600):
        count = count_func()
        i = start_index
        # 计算分页大小
        is_zero = count % batch_size
        pages = int(count / batch_size)
        if is_zero != 0:
            pages = pages + 1
        logger.info("=================分页查询数据库记录数据，共 " + str(pages) + " 页 ================")
        all_begin_time = time()
        while i <= pages:
            # 作用： 可以统计程序运行的时间
            logger.info("****************正查询 [ " + str(i) + "/" + str(pages) + " ] 页数据**************")
            begin_time = time()
            json_result = list_func(i, batch_size)
            db_end_time = time()
            logger.info('数据库查询用时时间：' + str(db_end_time - begin_time))
            data = [
                {
                    '_op_type': 'update',
                    "_index": index_name,
                    "doc": value,
                    "_id": value['id']
                } for value in json_result
            ]
            helpers.bulk(self.es, data)
            es_end_time = time()
            logger.info('插入索引用时为：' + str(es_end_time - db_end_time))
            i += 1
            if i % 100 == 0:
                logger.info('现插入数据共' + str(i) + '条。共用时为：' + str(time() - all_begin_time))

        all_end_time = time()
        logger.info('=================全部数据共' + str(count) + '条。用时为：' +
                    str(all_end_time - all_begin_time) + "======================")

    def full_import(self, count_func, list_func, index_name, start_index=1, batch_size=600):
        count = count_func()
        i = start_index
        # 计算分页大小
        is_zero = count % batch_size
        pages = int(count / batch_size)
        if is_zero != 0:
            pages = pages + 1
        logger.info("=================分页查询数据库记录数据，共 " + str(pages) + " 页 ================")
        all_begin_time = time()
        while i <= pages:
            # 作用： 可以统计程序运行的时间
            logger.info("****************正查询 [ " + str(i) + "/" + str(pages) + " ] 页数据**************")
            begin_time = time()
            json_result = list_func(i, batch_size)
            db_end_time = time()
            logger.info('数据库查询用时时间：' + str(db_end_time - begin_time))
            data = [
                {"_index": index_name,
                 "_source": value,
                 "_id": value['id']
                 } for value in json_result
            ]
            helpers.bulk(self.es, data)
            es_end_time = time()
            logger.info('插入索引用时为：' + str(es_end_time - db_end_time))
            i += 1
            if i % 100 == 0:
                logger.info('现插入数据共' + str(i) + '条。共用时为：' + str(time() - all_begin_time))

        all_end_time = time()
        logger.info('=================全部数据共' + str(count) + '条。用时为：' +
                    str(all_end_time - all_begin_time) + "======================")

    """
    根据最后更新时间来进行增量数据导入
    """

    def delta_import(self, query_func, last_update_time, index_name):
        # TODO 待完成
        pass

    def _get_list(self, index_name, batch_size, query_str=None, from_num=1):
        if not query_str:
            query_str = json.dumps({
                "query": {
                    'match_all': {}
                }
            })
        query_result = self.es.search(index=index_name, size=batch_size, body=query_str, from_=from_num)

        # 查询第一页数据
        hits = query_result['hits']['hits']
        # 总量
        total = query_result['hits']['total']
        result_list = []
        for item in hits:
            source = item['_source']
            result_list.append(source)
            # tmp = {}
            # for i_key in source: # 只处理有值的数据
            #     if source[i_key]:
            #         tmp[i_key] = source[i_key]

        return total['value'], result_list

    def _count(self, index_name, query_str=None):
        if query_str:
            count_result = self.es.count(index=index_name, body=query_str)
        else:
            count_result = self.es.count(index=index_name)
        if count_result:
            return count_result['count']
        return 0

    def _get_by_id(self, index_name, doc_id):
        try:
            res = self.es.get(index=index_name, id=doc_id)
            return res['_source']
        except NotFoundError as err:
            logger.error('es根据id查询出错：%s' % err)
            return None

    def _group_by_field(self, index_name, group_fields, query_info=None, other_flag=False):
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
                for child_group_field in child_group_fields:
                    child_field_name = child_group_field.get('name')
                    child_size = child_group_field.get('size') or 20
                    aggr_info[field_name + "_aggr"]['aggs'] = {
                        child_field_name + "_aggr": {
                            "terms": {
                                "field": child_field_name,
                                "size": child_size
                            }
                        }
                    }
        query = {
            "query": query_info,
            "size": 0,
            "aggs": aggr_info
        }
        result = self.es.search(index=index_name, body=json.dumps(query))
        return AggrUtil.handle_aggr_result(result, group_fields, other_flag=other_flag)

    def _store_all_data_to_file(self, index_name, query, file_obj, batch_size=500):
        query_result = self.es.search(index=index_name,
                                      scroll='5m',
                                      size=batch_size,
                                      body=json.dumps(query))

        # 查询第一页数据
        hits = query_result['hits']['hits']
        # 总量
        total = query_result['hits']['total']
        # 查询游标
        scroll_id = query_result['_scroll_id']
        source_to_json_file(hits, file_obj)

        pages = int(total["value"] / batch_size) + 1
        start_time = time()
        # for i in range(0, 2):
        for i in range(0, pages):
            page_start_time = time()
            logger.info('处理：%s/%s' % (i, pages))
            scroll_results = self.es.scroll(scroll_id=scroll_id,
                                            scroll='5m')
            hits = scroll_results['hits']['hits']
            source_to_json_file(hits, file_obj)
            logger.info('处理耗时：%s' % (time() - page_start_time))
        end_time = time()
        logger.info('总共获取数据 %s 条，共花费时间：%s' % (total["value"], (end_time - start_time)))


def write_to_file(json_str, file_obj):
    if json_str:
        file_obj.writelines(json_str)


def source_to_json_file(hits, file_obj):
    json_str = ''
    for item in hits:
        source = item['_source']
        tmp = {}
        for i_key in source:
            if source[i_key]:
                tmp[i_key] = source[i_key]
        json_str += str(tmp) + '\n'
    write_to_file(json_str, file_obj)
