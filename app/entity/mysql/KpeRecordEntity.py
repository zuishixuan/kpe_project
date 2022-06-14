from app.common.MysqlBaseEntity import db, MySqlBaseEntity


class KpeRecordEntity(db.Model, MySqlBaseEntity):
    unused_field = []
    # 表名
    __tablename__ = 'kpe_record'

    id = db.Column(db.INTEGER, autoincrement=True, primary_key=True)
    name = db.Column(db.String(255))
    describe = db.Column(db.String(3000))
    keyword = db.Column(db.String(255))
    keyword_kpe = db.Column(db.String(255))
    keyword_score = db.Column(db.FLOAT)
    submit_date = db.Column(db.String(255))

    def to_json(self):
        return_dict = {c.name: getattr(self, c.name) for c in self.__table__.columns}
        return kpe_record_dict_json(return_dict, self.unused_field)


def kpe_record_dict_json(return_dict, unused_field):
    if return_dict.get('keyword'):
        return_dict['keyword'] = return_dict.get('keyword').split(';')
    if return_dict.get('keyword_kpe'):
        return_dict['keyword_kpe'] = return_dict.get('keyword_kpe').split(';')
    for i in unused_field:
        if return_dict.get(i):
            return_dict.pop(i)
    return return_dict
