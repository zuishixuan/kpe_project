from manager import app
from app.dao.es.EsPatentRepository import EsPatentRepository
from config.constant import work_path
import ast
from logger import logger
import time
from app.utils import TextUtil


# def test_dynamic_add_field():
#     with app.app_context():
#         es_repository = EsPatentRepository()
#         es_repository.add_fields()


if __name__ == '__main__':
    with app.app_context():
        es_repository = EsPatentRepository()
        # es_repository.create_index()
        patent_path = work_path + '/test/dao/patents.txt'
        with open(patent_path, 'r', encoding='utf-8') as f:
            line = f.readline()
            tmp = 1
            data_list = []
            start_time = time.time()
            while line and line.strip() != '':
                p_dict = ast.literal_eval(line)
                # data_list.append(p_dict)
                title = p_dict.get('title')
                summary = p_dict.get('summary') or ''
                signory = p_dict.get('signory') or ''
                words = TextUtil.get_terms_weights(title + ' ' + summary + ' ' + signory)
                update_info = {
                    'id': p_dict.get('id'),
                    'words': words
                }
                data_list.append(update_info)
                if tmp % 500 == 0:
                    es_repository.update_batch(data_list)
                    data_list = []
                tmp += 1
                line = f.readline()
            if len(data_list) > 0:
                es_repository.update_batch(data_list)
                print(data_list)
            logger.info("共加载数据：%s，用时：%s" % (tmp, (time.time() - start_time)))
