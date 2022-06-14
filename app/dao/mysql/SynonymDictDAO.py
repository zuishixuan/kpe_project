from app.common.BaseDAO import BaseDAO, get_session
from app.entity.mysql.SynonymDictEntity import SynonymDictEntity
import re


class SynonymDictDAO:

    @staticmethod
    def count():
        return BaseDAO.do_count(SynonymDictEntity)

    @staticmethod
    def count_by_filter():
        return BaseDAO.do_count_by_filter(SynonymDictEntity)

    @staticmethod
    def get_list(pageno=1, pagesize=10):
        return BaseDAO.do_get_list(SynonymDictEntity, pageno, pagesize, 'id')

    @staticmethod
    def get_list_by_filter(pageno=1, pagesize=10):
        return BaseDAO.do_get_list_by_filter(SynonymDictEntity, pageno, pagesize, 'id')

    def get_detail(sid):
        return BaseDAO.do_get_detail(SynonymDictEntity, sid)

    @staticmethod
    def save_one(source: dict):
        if source is not None:
            entity = SynonymDictEntity()
            entity.dict2entity(source)
            return BaseDAO.do_save_one(entity)
        return False

    @staticmethod
    def query_all(pageno=1, pagesize=10):
        return BaseDAO.do_query_all(SynonymDictEntity, pageno, pagesize, 'id')

    @staticmethod
    def update_one(source: dict):
        if source is not None:
            return BaseDAO.do_update_one(SynonymDictEntity, source)
        return False

    @staticmethod
    def delete_one(id):
        if id is not None:
            return BaseDAO.do_delete_one(SynonymDictEntity, id)
        return False