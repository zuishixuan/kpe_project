from app.services.es.EsEnterpriseService import EsEnterpriseService
from manager import app

if __name__ == '__main__':
    with app.app_context():
        #res = EsEnterpriseService.query_and_group_by_year_match("province","河北")
        #res = EsEnterpriseService.query_and_group_by_year_match("county", "东光县")
        #print(len(res["year"]))
        res = EsEnterpriseService.query_group_business_scope_words(20)
        print(res)
