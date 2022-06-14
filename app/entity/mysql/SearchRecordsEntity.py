from datetime import datetime

from app.common.MysqlBaseEntity import db, MySqlBaseEntity


class SearchRecordsEntity(db.Model, MySqlBaseEntity):
    unused_field = []
    # 表名
    __tablename__ = 'search_records'

    id = db.Column(db.INTEGER, autoincrement=True, primary_key=True)
    word = db.Column(db.String(255))
    username = db.Column(db.String(255))
    search_date =db.Column(db.DateTime,default=datetime.now())


    def to_json(self):
        return_dict = {c.name: getattr(self, c.name) for c in self.__table__.columns}
        return user_dict_json(return_dict, self.unused_field)


def user_dict_json(return_dict, unused_field):
    for i in unused_field:
        if return_dict.get(i):
            return_dict.pop(i)
    return return_dict
