from app.common.MysqlBaseEntity import db, MySqlBaseEntity


class FeedbackRecordEntity(db.Model, MySqlBaseEntity):
    unused_field = []
    # 表名
    __tablename__ = 'feedback_record'

    id = db.Column(db.INTEGER, autoincrement=True, primary_key=True)
    kpe_record_id = db.Column(db.INTEGER)
    name = db.Column(db.String(255))
    keyword = db.Column(db.String(255))
    score = db.Column(db.FLOAT)
    submit_date = db.Column(db.String(255))

    def to_json(self):
        return_dict = {c.name: getattr(self, c.name) for c in self.__table__.columns}
        return feedback_record_dict_json(return_dict, self.unused_field)


def feedback_record_dict_json(return_dict, unused_field):
    for i in unused_field:
        if return_dict.get(i):
            return_dict.pop(i)
    return return_dict
