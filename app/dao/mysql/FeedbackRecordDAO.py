from app.common.BaseDAO import BaseDAO, get_session
from app.entity.mysql.FeedbackRecordEntity import FeedbackRecordEntity
import re


class FeedbackRecordDAO:

    @staticmethod
    def count():
        return BaseDAO.do_count(FeedbackRecordEntity)

    @staticmethod
    def get_list(pageno=1, pagesize=10):
        return BaseDAO.do_get_list(FeedbackRecordEntity, pageno, pagesize, 'id')

    @staticmethod
    def get_list_by_filter(pageno=1, pagesize=10):
        return BaseDAO.do_get_list_by_filter(FeedbackRecordEntity, pageno, pagesize, 'id')

    def get_detail(sid):
        return BaseDAO.do_get_detail(FeedbackRecordEntity, sid)

    @staticmethod
    def save_one(source: dict):
        if source is not None:
            entity = FeedbackRecordEntity()
            entity.dict2entity(source)
            return BaseDAO.do_save_one(entity)
        return False

    @staticmethod
    def query_all(pageno=1, pagesize=10):
        return BaseDAO.do_query_all(FeedbackRecordEntity, pageno, pagesize, 'id')

    @staticmethod
    def delete_one(id):
        if id is not None:
            return BaseDAO.do_delete_one(FeedbackRecordEntity, id)
        return False

    @staticmethod
    def get_list_by_kpe_record_id(kpe_record_id):
        res = BaseDAO.do_query_by_sql_filter(FeedbackRecordEntity, 'kpe_record_id', kpe_record_id).all()
        if res:
            li = []
            for r in res:
                li.append({
                    "id": r[0],
                    "kpe_record_id": r[1],
                    "name": r[2],
                    "keyword": r[3],
                    "submit_date": r[4],
                    "score": r[5]
                })
            print(li)
            return li
        else:
            return None
