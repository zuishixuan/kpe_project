from app.dao.mysql.SearchRecordsDAO import SearchRecordsDAO


class SearchRecordsService:

    @classmethod
    def count(cls):
        return SearchRecordsDAO.count()

    @classmethod
    def query_all(cls, pageno=1, pagesize=10):
        return SearchRecordsDAO.query_all(pageno, pagesize)

    @classmethod
    def get_list(cls, pageno=1, pagesize=10):
        return SearchRecordsDAO.get_list(pageno, pagesize)

    @classmethod
    def get_detail(cls, tid):
        return SearchRecordsDAO.get_detail(tid)

    @classmethod
    def save_one(cls, source: dict):
        return SearchRecordsDAO.save_one(source)

    @classmethod
    def count_by_group_column_time_scope(cls, be, end):
        result = SearchRecordsDAO.count_by_group_column_time_scope(be, end)
        re = []
        for item in result:
            re.append({
                'word': item[0],
                'frequency': item[1]
            })
        return re
