from app.dao.mysql import EnterpriseDAO
from manager import app

if __name__ == '__main__':
    with app.app_context():
        res = EnterpriseDAO.EnterPriseDAO.get_business_scope_words_by_id(1)
        print(res)
        # c = EnterpriseDAO.EnterPriseDAO.get_top_k_enterprise_sort_by_capital(10)
        # for i in c:
        #     print(i)

