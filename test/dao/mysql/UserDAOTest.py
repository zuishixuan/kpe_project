from app.dao.mysql import UserDAO
from manager import app

if __name__ == '__main__':
    with app.app_context():
        #detail = PatentDAO.count_by_filter()
        #print(detail)
        #c = UserDAO.UserDAO.get_detail_by_username(username='admin')
        #print(c)
        entity = {'id': 2, 'username': 'xuan','hashpwd':'123456','role': 'editor'}
        b= UserDAO.UserDAO.save_one(entity)
        # print("bbbbbbbbbbbb")
        print(b)

