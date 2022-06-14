from app.dao.mysql.KpeRecordDAO import KpeRecordDAO
from werkzeug.security import generate_password_hash, check_password_hash


class KpeRecordService:

    @classmethod
    def count(cls):
        return KpeRecordDAO.count()

    @classmethod
    def query_all(cls, pageno=1, pagesize=10):
        return KpeRecordDAO.query_all(pageno, pagesize)

    @classmethod
    def get_list(cls, pageno=1, pagesize=10):
        return KpeRecordDAO.get_list(pageno, pagesize)

    @classmethod
    def get_detail(cls, tid):
        return KpeRecordDAO.get_detail(tid)

    @classmethod
    def save_one(cls, source: dict):
        return KpeRecordDAO.save_one(source)

    @classmethod
    def save_one_with_id_return(cls, source: dict):
        return KpeRecordDAO.save_one_with_id_return(source)

    @classmethod
    def delete_one(cls, id):
        return KpeRecordDAO.delete_one(id)

    @classmethod
    def get_detail_by_iden(cls, iden):
        return KpeRecordDAO.get_detail_by_identifier(iden)



