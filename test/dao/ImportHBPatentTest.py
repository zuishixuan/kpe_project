from app.dao.es.EsHBPatentRepository import EsHBPatentRepository
from app.dao.mysql.PatentDAO import PatentDAO
from manager import app

if __name__ == '__main__':
    with app.app_context():
        client = EsHBPatentRepository()
        client.create_index()

        client.full_import(PatentDAO.count, PatentDAO.get_list, 'hb_patent')
        # client.delete_index()