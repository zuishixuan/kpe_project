from app.common.MysqlBaseEntity import db, MySqlBaseEntity


class MetadataEntity(db.Model, MySqlBaseEntity):
    unused_field = []
    # 表名
    __tablename__ = 'metadata'

    id = db.Column(db.INTEGER, autoincrement=True, primary_key=True)
    name = db.Column(db.String(255))
    identifier = db.Column(db.String(255))
    subject_category = db.Column(db.String(255))
    theme_category = db.Column(db.String(255))
    describe = db.Column(db.String(3000))
    keyword = db.Column(db.String(255))
    keyword_deny = db.Column(db.String(255))
    keyword_score = db.Column(db.String(255))
    keyword_deny_score = db.Column(db.String(255))
    generate_date = db.Column(db.String(255))
    submit_date = db.Column(db.String(255))
    submit_user = db.Column(db.String(255))

    def to_json(self):
        return_dict = {c.name: getattr(self, c.name) for c in self.__table__.columns}
        return metadata_dict_json(return_dict, self.unused_field)


def metadata_dict_json(return_dict, unused_field):
    if return_dict.get('subject_category'):
        return_dict['subject_category'] = return_dict.get('subject_category').split(' ')
    if return_dict.get('theme_category'):
        return_dict['theme_category'] = return_dict.get('theme_category').split('>')
    if return_dict.get('keyword'):
        return_dict['keyword'] = return_dict.get('keyword').split(';')
    if return_dict.get('keyword_deny'):
        return_dict['keyword_deny'] = return_dict.get('keyword_deny').split(';')
    if return_dict.get('keyword_score'):
        return_dict['keyword_score'] = return_dict.get('keyword_score').split(';')
    if return_dict.get('keyword_deny_score'):
        return_dict['keyword_deny_score'] = return_dict.get('keyword_deny_score').split(';')
    for i in unused_field:
        if return_dict.get(i):
            return_dict.pop(i)
    return return_dict
