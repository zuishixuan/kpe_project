from flask import Blueprint, request
from flask import jsonify
from app.common.Response import R
from app.services.mysql.MetadataService import MetadataService
from app.services.es.EsMetadataService import EsMetadataService

metadata = Blueprint('metadata', __name__)


@metadata.route('/page', methods=['GET'])
def page():
    pageno = request.args.get('pageno') or None
    pagesize = request.args.get('pagesize') or None
    data = MetadataService.get_list(pageno, pagesize)
    totalnum = MetadataService.count()
    r = R.ok({'data':data,'totalnum':totalnum})
    return jsonify(r.to_json())

@metadata.route('/save', methods=['POST'])
def save():
    params = request.json
    # name = params.get("title", None)
    # subject_category = params.get("subject_category", None)
    # theme_category = params.get("theme_category", None)
    # describe = params.get("describe", None)
    # keyword = params.get("keyword", None)
    # keyword_deny = params.get("keyword_deny", None)
    print(params)
    if params and MetadataService.save_one(params):
        re = MetadataService.get_detail_by_iden(params['identifier'])
        params['id']=re[0]
        EsMetadataService().save(params)
        r = R.ok("success")
    else:
        r = R.fail("fail")
    return jsonify(r.to_json())
