from app.dao.es.EsPatentRepository import EsPatentRepository
from config.constant import LABEL_LIST

if __name__ == '__main__':
    # 调用pytest的main函数执行测试
    keywords = [item['name'] for item in LABEL_LIST]
    print(keywords)
    print(len(keywords))
    repository = EsPatentRepository()
    repository.get_and_store_by_keywords(keywords)
