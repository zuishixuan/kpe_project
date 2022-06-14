from app.dao.es.EsDongGuangEnterpriseRepository import EsDongGuangEnterpriseRepository
from app.dao.mysql.EnterpriseDAO import EnterPriseDAO
from manager import app

if __name__ == '__main__':
    with app.app_context():
        client = EsDongGuangEnterpriseRepository()
        client.create_index()
        client.full_import(EnterPriseDAO.count, EnterPriseDAO.get_list, 'dongguang_enterprise')
