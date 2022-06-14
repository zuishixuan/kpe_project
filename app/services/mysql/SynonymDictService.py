from app.dao.mysql.SynonymDictDAO import SynonymDictDAO
from werkzeug.security import generate_password_hash, check_password_hash


class SynonymDictService:

    @classmethod
    def count(cls):
        return SynonymDictDAO.count()

    @classmethod
    def query_all(cls, pageno=1, pagesize=10):
        return SynonymDictDAO.query_all(pageno, pagesize)

    @classmethod
    def get_list(cls, pageno=1, pagesize=10):
        return SynonymDictDAO.get_list(pageno, pagesize)

    @classmethod
    def get_detail(cls, tid):
        return SynonymDictDAO.get_detail(tid)

    @classmethod
    def save_one(cls, source: dict):
        return SynonymDictDAO.save_one(source)

    @classmethod
    def delete_one(cls, id):
        return SynonymDictDAO.delete_one(id)

    @classmethod
    def update_one(cls, source: dict):
        return SynonymDictDAO.update_one(source)