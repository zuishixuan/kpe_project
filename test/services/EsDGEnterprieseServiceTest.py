from app.services.es.EsDGEnterprieseService import EsDGEnterprieseService
from manager import app

if __name__ == '__main__':
    with app.app_context():
        s = "东光镇；于桥乡"
        res = EsDGEnterprieseService.query_group_towns(s)
        print(res)
