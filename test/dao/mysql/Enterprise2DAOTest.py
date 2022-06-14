from app.dao.mysql import Enterprise2DAO
from manager import app

if __name__ == '__main__':
    with app.app_context():
        res = Enterprise2DAO.Enterprise2DAO.get_detail(1)
        print(res)
        # c = EnterpriseDAO.EnterPriseDAO.get_top_k_enterprise_sort_by_capital(10)
        # for i in c:
        #     print(i)

