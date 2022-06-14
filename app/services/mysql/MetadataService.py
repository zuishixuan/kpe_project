from app.dao.mysql.MetadataDAO import MetadataDAO
from werkzeug.security import generate_password_hash, check_password_hash


class MetadataService:

    @classmethod
    def count(cls):
        return MetadataDAO.count()

    @classmethod
    def query_all(cls, pageno=1, pagesize=10):
        return MetadataDAO.query_all(pageno, pagesize)

    @classmethod
    def get_list(cls, pageno=1, pagesize=10):
        return MetadataDAO.get_list(pageno, pagesize)

    @classmethod
    def get_detail(cls, tid):
        return MetadataDAO.get_detail(tid)

    @classmethod
    def save_one(cls, source: dict):
        return MetadataDAO.save_one(source)

    @classmethod
    def get_detail_by_iden(cls, iden):
        return MetadataDAO.get_detail_by_identifier(iden)



