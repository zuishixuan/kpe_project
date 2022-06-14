from app.common.BaseDAO import BaseDAO, get_session
from app.entity.mysql.SearchRecordsEntity import SearchRecordsEntity
import re


class SearchRecordsDAO:

    @staticmethod
    def count():
        return BaseDAO.do_count(SearchRecordsEntity)

    @staticmethod
    def get_list(pageno=1, pagesize=10):
        return BaseDAO.do_get_list(SearchRecordsEntity, pageno, pagesize, 'id')

    @staticmethod
    def get_list_by_filter(pageno=1, pagesize=10):
        return BaseDAO.do_get_list_by_filter(SearchRecordsEntity, pageno, pagesize, 'id')

    def get_detail(sid):
        return BaseDAO.do_get_detail(SearchRecordsEntity, sid)

    @staticmethod
    def save_one(source: dict):
        if source is not None:
            entity = SearchRecordsEntity()
            entity.dict2entity(source)
            return BaseDAO.do_save_one(entity)
        return False

    @staticmethod
    def count_by_group_column_time_scope(be, end):
        result = BaseDAO.do_count_by_group_column_time_scope(SearchRecordsEntity,'word', be, end)
        return result
