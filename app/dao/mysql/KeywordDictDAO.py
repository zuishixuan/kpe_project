from app.common.BaseDAO import BaseDAO, get_session
from app.entity.mysql.KeywordDictEntity import KeywordDictEntity
import re


class KeywordDictDAO:

    @staticmethod
    def count():
        return BaseDAO.do_count(KeywordDictEntity)

    @staticmethod
    def count_by_filter():
        return BaseDAO.do_count_by_filter(KeywordDictEntity)

    @staticmethod
    def get_list(pageno=1, pagesize=10):
        return BaseDAO.do_get_list(KeywordDictEntity, pageno, pagesize, 'id')

    @staticmethod
    def get_list_by_filter(pageno=1, pagesize=10):
        return BaseDAO.do_get_list_by_filter(KeywordDictEntity, pageno, pagesize, 'id')

    def get_detail(sid):
        return BaseDAO.do_get_detail(KeywordDictEntity, sid)

    @staticmethod
    def save_one(source: dict):
        if source is not None:
            entity = KeywordDictEntity()
            entity.dict2entity(source)
            return BaseDAO.do_save_one(entity)
        return False

    @staticmethod
    def query_all(pageno=1, pagesize=10):
        return BaseDAO.do_query_all(KeywordDictEntity, pageno, pagesize, 'id')

    @staticmethod
    def update_one(source: dict):
        if source is not None:
            return BaseDAO.do_update_one(KeywordDictEntity, source)
        return False

    @staticmethod
    def delete_one(id):
        if id is not None:
            return BaseDAO.do_delete_one(KeywordDictEntity, id)
        return False

