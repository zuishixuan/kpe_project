from app.dao.mysql import PatentDAO
from manager import app

if __name__ == '__main__':
    with app.app_context():
        #detail = PatentDAO.count_by_filter()
        #print(detail)
        c = PatentDAO.get_dongguang_enter_patent_count()
        print(c)

