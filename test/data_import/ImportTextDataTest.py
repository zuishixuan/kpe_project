from manager import app
from app.dao.es.EsPatentRepository import EsPatentRepository
from config.constant import work_path
import ast
from logger import logger
import time

if __name__ == '__main__':
    with app.app_context():
        es_repository = EsPatentRepository()
        es_repository.create_index()
        patent_path = work_path + '/test/dao/patents_keywords.txt'
        with open(patent_path, 'r', encoding='utf-8') as f:
            line = f.readline()
            tmp = 1
            data_list = []
            start_time = time.time()
            while line and line.strip() != '':
                p_dict = ast.literal_eval(line)
                data_list.append(p_dict)
                if tmp % 500 == 0:
                    es_repository.save_batch(data_list)
                    data_list = []
                tmp += 1
                line = f.readline()
            if len(data_list) > 0:
                es_repository.save_batch(data_list)
            logger.info("共加载数据：%s，用时：%s" % (tmp, (time.time() - start_time)))
