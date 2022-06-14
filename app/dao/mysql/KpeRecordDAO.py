from app.common.BaseDAO import BaseDAO, get_session
from app.entity.mysql.KpeRecordEntity import KpeRecordEntity
import re


class KpeRecordDAO:

    @staticmethod
    def count():
        return BaseDAO.do_count(KpeRecordEntity)

    @staticmethod
    def get_list(pageno=1, pagesize=10):
        return BaseDAO.do_get_list(KpeRecordEntity, pageno, pagesize, 'id')

    @staticmethod
    def get_list_by_filter(pageno=1, pagesize=10):
        return BaseDAO.do_get_list_by_filter(KpeRecordEntity, pageno, pagesize, 'id')

    def get_detail(sid):
        return BaseDAO.do_get_detail(KpeRecordEntity, sid)

    @staticmethod
    def save_one(source: dict):
        if source is not None:
            entity = KpeRecordEntity()
            entity.dict2entity(source)
            return BaseDAO.do_save_one(entity)
        return False

    @staticmethod
    def save_one_with_id_return(source: dict):
        if source is not None:
            entity = KpeRecordEntity()
            entity.dict2entity(source)
            return BaseDAO.do_save_one_with_id_return(entity)
        return -1

    @staticmethod
    def query_all(pageno=1, pagesize=10):
        return BaseDAO.do_query_all(KpeRecordEntity, pageno, pagesize, 'id')

    @staticmethod
    def delete_one(id):
        if id is not None:
            return BaseDAO.do_delete_one(KpeRecordEntity, id)
        return False

    @staticmethod
    def get_detail_by_identifier(iden):
        res = BaseDAO.do_query_by_sql_filter(KpeRecordEntity, 'identifier', iden).first()
        print(res)
        if res:
            return res
        else:
            return None
