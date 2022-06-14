from app.services.es.EsHBPatentService import EsHBPatentService

from manager import app

if __name__ == '__main__':
    with app.app_context():
        res = EsHBPatentService.group_field_year_range_keywords(year_beg="2002", year_end="2002")
        print(res)
