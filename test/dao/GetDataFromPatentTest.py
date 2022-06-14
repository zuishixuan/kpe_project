from app.dao.es.EsPatentRepository import EsPatentRepository
from config.constant import work_path
import json

with open(work_path + '/config/data/plastic_packaging.json', 'r', encoding="UTF-8") as f:
    CODES_JSON = json.load(f)

if __name__ == '__main__':
    # 调用pytest的main函数执行测试
    codes = [item['category_id'] for item in CODES_JSON['RECORDS']]
    repository = EsPatentRepository()
    repository.get_and_store_by_codes(codes)
