from app.services.mysql import UserService
from manager import app

if __name__ == '__main__':
    with app.app_context():
        a = UserService.UserService.get_list(1,10)
        print(a)