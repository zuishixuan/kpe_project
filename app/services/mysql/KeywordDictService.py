from app.dao.mysql.KeywordDictDAO import KeywordDictDAO
from werkzeug.security import generate_password_hash, check_password_hash


class KeywordDictService:

    @classmethod
    def count(cls):
        return KeywordDictDAO.count()

    @classmethod
    def query_all(cls, pageno=1, pagesize=10):
        return KeywordDictDAO.query_all(pageno, pagesize)

    @classmethod
    def get_list(cls, pageno=1, pagesize=10):
        return KeywordDictDAO.get_list(pageno, pagesize)

    @classmethod
    def get_detail(cls, tid):
        return KeywordDictDAO.get_detail(tid)

    @classmethod
    def save_one(cls, source: dict):
        return KeywordDictDAO.save_one(source)

    @classmethod
    def delete_one(cls, id):
        return KeywordDictDAO.delete_one(id)

    @classmethod
    def update_one(cls, source: dict):
        return KeywordDictDAO.update_one(source)



