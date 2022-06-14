from app.dao.es.EsMetadataRepository import EsMetadataRepository
from app.dao.mysql.MetadataDAO import MetadataDAO
from manager import app

if __name__ == '__main__':
    with app.app_context():
        client = EsMetadataRepository()
        client.create_index()

        client.full_import(MetadataDAO.count, MetadataDAO.get_list, 'metadata')
        # client.delete_index()