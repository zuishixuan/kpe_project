import json
from manager import app
from app.dao.es.EsPatentRepository import EsPatentRepository
from app.dao.es.EsWordsRepository import EsWordsRepository
from config.constant import work_path
import ast
from logger import logger
import time
from app.utils import TextUtil
import csv
import hashlib


def get_patents_by_keywords(category):
    category_id = category.get('id')
    category_name = category.get('name')
    filename = 'category_keyword_patent.txt'
    file_obj = open(filename, 'w', encoding='utf8')
    query = {
        "query": {
            "bool": {
                "should": [
                    {"term": {"title": category_name}},
                    {"term": {"summary": category_name}},
                    {"term": {"signory": category_name}}
                ]
            }
        }
    }
    patent_repository.store_all_data_to_file(query, file_obj)
    tmp = 0
    with open(filename, 'r', encoding='utf8') as f:
        line = f.readline()
        begin_time = time.time()
        while line and line.strip() != '':
            tmp += 1
            logger.info("第%s条，category_id:%s,name:%s" % (tmp, category_id, category_name))
            try:
                p_dict = ast.literal_eval(line)
            except SyntaxError:
                line = f.readline()
                continue
            # p_dict = eval(line)
            application_area_code = p_dict.get('application_area_code')
            application_date = p_dict.get('application_date')
            if not application_date:
                application_date = p_dict.get('publication_date')
            if application_date:
                strptime = time.strptime(application_date, '%Y-%m-%d %H:%M:%S')
                year_month = time.strftime("%Y-%m", strptime)
            else:
                year_month = '0'
            year = p_dict.get('year')
            words = get_words(p_dict)
            words = [n for n in words if n['w'] > 1]

            result_words = []
            for word in words:
                id_str = '%s.%s.%s' % (category_id, year_month,
                                       get_hash(application_area_code, word['n']))
                words_item = keywords_repository.get_by_id(id_str)
                tmp_info = {
                    'id': id_str,
                    'country': application_area_code,
                    'province': application_area_code,
                    'category_id': category_id,
                    'year': year,
                    'month': year_month,
                    'name': word['n'],
                }
                if words_item:
                    tmp_info['doc_num'] = words_item['doc_num'] + 1
                    tmp_info['weight'] = words_item['weight'] + word['w']
                else:
                    tmp_info['doc_num'] = 1
                    tmp_info['weight'] = word['w']
                result_words.append(tmp_info)
            keywords_repository.save_batch(result_words)
            line = f.readline()
        logger.info("共加载数据：%s，用时：%s" % (tmp, (time.time() - begin_time)))


def get_word_index(words, target):
    for index, word in enumerate(words):
        if word['n'] == target:
            return index
    return -1


def get_words(patent):
    title = patent.get('title')
    summary = patent.get('summary') or ''
    signory = patent.get('signory') or ''
    words = TextUtil.get_terms_weights(title + ' ' + summary + ' ' + signory)
    return words


def get_hash(area, text):
    if not text:
        text = ''
    else:
        text = str(text)
    if not area:
        area = ''
    else:
        area = str(area)
    text = area + text
    md5.update(text.encode('utf-8'))
    return md5.hexdigest()


md5 = hashlib.md5()
patent_repository = EsPatentRepository()
keywords_repository = EsWordsRepository()

if __name__ == '__main__':
    with app.app_context():
        keywords_repository.create_index()
        file_path = work_path + '/config/data/all.csv'
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for item in reader:
                get_patents_by_keywords(item)
