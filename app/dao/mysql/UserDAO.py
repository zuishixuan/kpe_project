from app.common.BaseDAO import BaseDAO, get_session
from app.entity.mysql.UserEntity import UserEntity
from werkzeug.security import generate_password_hash, check_password_hash
import re


class UserDAO:

    @staticmethod
    def count():
        return BaseDAO.do_count(UserEntity)

    @staticmethod
    def count_by_filter():
        return BaseDAO.do_count_by_filter(UserEntity)

    @staticmethod
    def get_list(pageno=1, pagesize=10):
        return BaseDAO.do_get_list(UserEntity, pageno, pagesize, 'id')

    @staticmethod
    def get_list_by_filter(pageno=1, pagesize=10):
        return BaseDAO.do_get_list_by_filter(UserEntity, pageno, pagesize, 'id')

    def get_detail(sid):
        return BaseDAO.do_get_detail(UserEntity, sid)

    def get_detail_by_username(username):
        res = BaseDAO.do_query_by_sql_filter(UserEntity, 'username', username).first()
        print(res)
        if res:
            (id, usern, hashpwd, role, add_date) = res
            detail = {
                'id': id,
                'username': usern,
                'hashpwd': hashpwd,
                'role': role,
                'add_date': add_date
            }
            return detail
        else:
            return None

    @staticmethod
    def save_one(source: dict):
        if source is not None:
            entity = UserEntity()
            source['hashpwd'] = generate_password_hash(source['pwd'])
            entity.dict2entity(source)
            return BaseDAO.do_save_one(entity)
        return False

    @staticmethod
    def update_one(source: dict):
        if source is not None:
            return BaseDAO.do_update_one(UserEntity, source)
        return False

    @staticmethod
    def query_all(pageno=1, pagesize=10):
        return BaseDAO.do_query_all(UserEntity, pageno, pagesize, 'id')

    @staticmethod
    def delete_one(id):
        if id is not None:
            return BaseDAO.do_delete_one(UserEntity, id)
        return False
