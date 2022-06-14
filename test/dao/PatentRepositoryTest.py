from app.dao.es.EsPatentRepository import EsPatentRepository

if __name__ == '__main__':
    repository = EsPatentRepository()
    group_fields = [
        {
            "name": "application_area_code",
            "children": [
                {
                    "name": "year",
                    "size": 4
                }
            ]
        },
        {
            "name": "applicant_name",
            "size": 5
        }
    ]
    results = repository.group_by_field(group_fields)
    print(results)
