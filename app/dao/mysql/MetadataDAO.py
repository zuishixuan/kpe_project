from app.common.BaseDAO import BaseDAO, get_session
from app.entity.mysql.MetadataEntity import MetadataEntity
import re


class MetadataDAO:

    @staticmethod
    def count():
        return BaseDAO.do_count(MetadataEntity)

    @staticmethod
    def get_list(pageno=1, pagesize=10):
        return BaseDAO.do_get_list(MetadataEntity, pageno, pagesize, 'id')

    @staticmethod
    def get_list_by_filter(pageno=1, pagesize=10):
        return BaseDAO.do_get_list_by_filter(MetadataEntity, pageno, pagesize, 'id')

    def get_detail(sid):
        return BaseDAO.do_get_detail(MetadataEntity, sid)

    @staticmethod
    def save_one(source: dict):
        if source is not None:
            entity = MetadataEntity()
            entity.dict2entity(source)
            return BaseDAO.do_save_one(entity)
        return False

    @staticmethod
    def query_all(pageno=1, pagesize=10):
        return BaseDAO.do_query_all(MetadataEntity, pageno, pagesize, 'id')

    @staticmethod
    def get_detail_by_identifier(iden):
        res = BaseDAO.do_query_by_sql_filter(MetadataEntity, 'identifier', iden).first()
        print(res)
        if res:
            return res
        else:
            return None
