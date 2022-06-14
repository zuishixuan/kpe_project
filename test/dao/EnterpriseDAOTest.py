from app.dao.es.EsEnterpriseRepository import EsEnterpriseRepository
from manager import app

if __name__ == '__main__':
    with app.app_context():
        client = EsEnterpriseRepository()
        # group_fields = [
        #     {'name': 'town'},
        #     {'name': 'business_scope_words', 'size': 10}
        # ]
        group_fields = [
            {
                'name': 'year',
                'children': [
                    {'name': 'business_scope_words', 'size': 10}
                ]
            },
        ]

        # group_fields = [
        #     {
        #         'name': 'year',
        #         'children': [
        #             {'name': 'business_scope_words', 'size': 10}
        #         ]
        #     },
        # ]
        results = client.group_and_count(group_fields)
        print(results)
