from app.dao.mysql import PatentDAO


class PatentService:

    @classmethod
    def count(cls):
        return PatentDAO.count()

    @classmethod
    def query_all(cls, pageno=1, pagesize=10):
        return PatentDAO.query_all(pageno, pagesize)

    @classmethod
    def get_list(cls, pageno=1, pagesize=10):
        return PatentDAO.get_list(pageno, pagesize)

    @classmethod
    def get_detail(cls, tid):
        return PatentDAO.get_detail(tid)
