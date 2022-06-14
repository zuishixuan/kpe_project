from app.dao.es.EsHBPatentRepository import EsHBPatentRepository
from manager import app

if __name__ == '__main__':
    with app.app_context():
        client = EsHBPatentRepository()
        # group_fields = [
        #     {'name': 'town'},
        #     {'name': 'business_scope_words', 'size': 10}
        # ]
        group_fields = [
            {
                'name': 'field_mark',
                # 'children': [
                #     {'name': 'business_scope_words', 'size': 10}
                # ]
            },
        ]
        results = client.group_and_count(group_fields=group_fields)
        print(results)
