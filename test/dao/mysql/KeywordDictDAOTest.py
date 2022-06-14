import datetime

from app.dao.mysql import KeywordDictDAO
from manager import app

if __name__ == '__main__':
    with app.app_context():
        #detail = PatentDAO.count_by_filter()
        #print(detail)
        c = KeywordDictDAO.KeywordDictDAO.get_detail(1)
        print(c)
        entity = {'id': 2, 'word': '新型冠状病毒', 'weight': 5.0, 'status': 2, 'add_date': datetime.date(2022, 3, 28)}
        b= KeywordDictDAO.KeywordDictDAO.save_one(entity)
        print("bbbbbbbbbbbb")
        print(b)

