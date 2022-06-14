from app.dao.es.EsEnterpriseRepository import EsEnterpriseRepository
from app.dao.mysql.Enterprise2DAO import Enterprise2DAO
from manager import app

if __name__ == '__main__':
    with app.app_context():
        client = EsEnterpriseRepository()
        client.create_index()
        client.full_import(Enterprise2DAO.count, Enterprise2DAO.get_list, 'hb_enterprise')
