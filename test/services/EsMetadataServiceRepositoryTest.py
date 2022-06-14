from app.services.es.EsMetadataService import EsMetadataRepository

from manager import app

if __name__ == '__main__':
    with app.app_context():
        res = EsMetadataRepository()
        a = res.get_multi_query_page(1,10,'视频')
        print(a)
