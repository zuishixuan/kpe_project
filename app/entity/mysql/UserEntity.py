from app.common.MysqlBaseEntity import db, MySqlBaseEntity
from werkzeug.security import generate_password_hash, check_password_hash


class UserEntity(db.Model, MySqlBaseEntity):
    unused_field = []
    # 表名
    __tablename__ = 'users'

    id = db.Column(db.INTEGER, autoincrement=True, primary_key=True)
    username = db.Column(db.String(64))
    hashpwd = db.Column(db.String(128))
    role = db.Column(db.String(20))
    add_date = db.Column(db.String(255))


    def to_json(self):
        return_dict = {c.name: getattr(self, c.name) for c in self.__table__.columns}
        return user_dict_json(return_dict, self.unused_field)


def user_dict_json(return_dict, unused_field):
    for i in unused_field:
        if return_dict.get(i):
            return_dict.pop(i)
    return return_dict
