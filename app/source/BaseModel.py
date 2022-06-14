from app.dao.mysql import PatentStopwordsDAO
from app.dao.es.EsModelRepository import EsModelRepository
import json
from config.constant import LABEL_LIST, LABEL_DICT

class BaseModel:
    def __init__(self):
        # stopwords = PatentStopwordsDAO.do_query_by_sql_filter('status', '1')
        stopwords = []
        self.stopwords = [word[1] for word in list(stopwords)]
        self.client = EsModelRepository()
        # self.stopwords = PatentStopwordsDAO.do_query_by_sql_filter('status', '1')

    def init(self):
        raise NotImplemented

    # def init_data(self):
    #     raise NotImplemented

    def change_stopwords(self, stopwords):
        self.stopwords = stopwords

    def add_stopwords(self, words):
        if isinstance(words, list):
            for w in words:
                if w not in self.stopwords:
                    self.stopwords.append(w)
        else:
            if words not in self.stopwords:
                self.stopwords.append(words)

    def get_all_data(self):
        batch_size = 4000
        result = []
        query_result = self.client.es.search(index=self.client.index_name,
                                             scroll='5m',
                                             size=batch_size,
                                             body=json.dumps({}))

        # 查询第一页数据
        hits = query_result['hits']['hits']
        result += hits
        # 总量
        total = query_result['hits']['total']
        # 查询游标
        scroll_id = query_result['_scroll_id']

        pages = int(total["value"] / batch_size) + 1
        # pages = int(total["value"] / batch_size) - 110
        # start_time = time.time()
        # for i in range(0, 2):
        for i in range(0, pages):
            # page_start_time = time.time()
            # logger.info('处理：%s/%s' % (i, pages))
            scroll_results = self.client.es.scroll(scroll_id=scroll_id,
                                                   scroll='5m')
            hits = scroll_results['hits']['hits']
            result += hits
        with open('/Users/hzy/Projects/Labratory/python/tpidentify/app/data/data_v0.2.txt', 'w') as f:
            for r in result:
                _d = r['_source']
                c = _d.get('field_mark', None)
                # if c is None:
                #     st = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(_d['id'], _d['iter_num'],
                #                                                                        _d['title'], _d['abstract']
                #                                                                        , _d['tec_keyw'], _d['applicant'],
                #                                                                        'nan', _d['patent_type'],
                #                                                                        _d['appli_day'], _d['public_day'],
                #                                                                        _d['province'], _d['city'], 2,'nan'
                #                                                                        )
                if c is not None:
                    st = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(_d['id'], _d['iter_num'],
                                                                                       _d['title'], _d['abstract']
                                                                                       , _d['tec_keyw'], _d['applicant'],
                                                                                       'nan', _d['patent_type'],
                                                                                       _d['appli_day'], _d['public_day'],
                                                                                       _d['province'], _d['city'], 2,
                                                                                       ','.join(c))
                    f.write(st)
        print("ssss")
        # for d in result:
        # return result
        # source_to_json_file(hits, file_obj)
        # logger.info('处理耗时：%s' % (time.time() - page_start_time))
        # end_time = time.time()
        # logger.info('总共获取数据 %s 条，共花费时间：%s' % (total["value"], (end_time - start_time)))
