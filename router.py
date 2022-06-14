
from app.controllers.mysql.UserController import user
from app.controllers.KpeController import kpe
from app.controllers.mysql.MetadataController import metadata
from app.controllers.mysql.KeywordDictController import keyword_dict
from app.controllers.mysql.SynonymDictController import synonym_dict
from app.controllers.EsMetadataController import es_metadata
from app.controllers.mysql.SearchRecordsController import search_record
from app.controllers.mysql.KpeRecordController import kpe_record
from app.controllers.mysql.FeedbackRecordController import feedback_record

from config.config import app

# 注册路由

app.register_blueprint(user, url_prefix='/user')
app.register_blueprint(kpe, url_prefix='/kpe')
app.register_blueprint(metadata, url_prefix='/metadata')
app.register_blueprint(keyword_dict, url_prefix='/keyword-dict')
app.register_blueprint(synonym_dict, url_prefix='/synonym-dict')
app.register_blueprint(es_metadata, url_prefix='/es-metadata')
app.register_blueprint(search_record, url_prefix='/search-record')
app.register_blueprint(kpe_record, url_prefix='/kpe-record')
app.register_blueprint(feedback_record, url_prefix='/feedback-record')
