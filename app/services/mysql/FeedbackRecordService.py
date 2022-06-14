from app.dao.mysql.FeedbackRecordDAO import FeedbackRecordDAO
from werkzeug.security import generate_password_hash, check_password_hash


class FeedbackRecordService:

    @classmethod
    def count(cls):
        return FeedbackRecordDAO.count()

    @classmethod
    def query_all(cls, pageno=1, pagesize=10):
        return FeedbackRecordDAO.query_all(pageno, pagesize)

    @classmethod
    def get_list(cls, pageno=1, pagesize=10):
        return FeedbackRecordDAO.get_list(pageno, pagesize)

    @classmethod
    def get_detail(cls, tid):
        return FeedbackRecordDAO.get_detail(tid)

    @classmethod
    def save_one(cls, source: dict):
        return FeedbackRecordDAO.save_one(source)

    @classmethod
    def delete_one(cls, id):
        return FeedbackRecordDAO.delete_one(id)

    @classmethod
    def get_list_by_kpe_record_id(cls, kpe_record_id):
        return FeedbackRecordDAO.get_list_by_kpe_record_id(kpe_record_id)



