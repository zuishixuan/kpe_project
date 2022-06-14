from app.common.MysqlBaseEntity import db, MySqlBaseEntity


class SynonymDictEntity(db.Model, MySqlBaseEntity):
    unused_field = []
    # 表名
    __tablename__ = 'synonym_dict'

    id = db.Column(db.INTEGER, autoincrement=True, primary_key=True)
    word = db.Column(db.String(50))
    main_word = db.Column(db.String(50))
    add_date = db.Column(db.String(255))
    status = db.Column(db.String)
    ## 0禁用状态 1启用

    def to_json(self):
        return_dict = {c.name: getattr(self, c.name) for c in self.__table__.columns}
        return user_dict_json(return_dict, self.unused_field)


def user_dict_json(return_dict, unused_field):
    for i in unused_field:
        if return_dict.get(i):
            return_dict.pop(i)
    return return_dict
