from app.dao.mysql.UserDAO import UserDAO
from werkzeug.security import generate_password_hash, check_password_hash


class UserService:

    @classmethod
    def count(cls):
        return UserDAO.count()

    @classmethod
    def query_all(cls, pageno=1, pagesize=10):
        return UserDAO.query_all(pageno, pagesize)

    @classmethod
    def get_list(cls, pageno=1, pagesize=10):
        return UserDAO.get_list(pageno, pagesize)

    @classmethod
    def get_detail(cls, tid):
        return UserDAO.get_detail(tid)

    @classmethod
    def get_detail_by_username(cls, username):
        return UserDAO.get_detail_by_username(username)

    @classmethod
    def login(cls, username, pwd):
        userdetail = UserDAO.get_detail_by_username(username)
        if userdetail:
            if check_password_hash(userdetail['hashpwd'], pwd):
                return True
        return False

    @classmethod
    def save_one(cls, source: dict):
        return UserDAO.save_one(source)

    @classmethod
    def delete_one(cls, id):
        return UserDAO.delete_one(id)

    @classmethod
    def update_one(cls, source: dict):
        return UserDAO.update_one(source)
