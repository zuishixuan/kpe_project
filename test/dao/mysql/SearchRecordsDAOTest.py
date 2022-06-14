import datetime
import time

from app.dao.mysql import SearchRecordsDAO
from manager import app

if __name__ == '__main__':
    with app.app_context():
        # detail = PatentDAO.count_by_filter()
        # print(detail)
        a = {'word': '西喀西', 'username': 'admin'}
        b = SearchRecordsDAO.SearchRecordsDAO.save_one(a)
        print(b)
        c = SearchRecordsDAO.SearchRecordsDAO.count_by_group_column_time_scope("2022-01-25", "2022-04-01")
        print(c)
        # c = SearchRecordsDAO.SearchRecordsDAO.get_detail(1)
        # print(c)
        # print(datetime.datetime(2022, 4, 1, 7, 22, 37))
