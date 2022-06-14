from flask import Blueprint, request
from flask import jsonify
from app.common.Response import R
from app.services.mysql.KeywordDictService import KeywordDictService

keyword_dict = Blueprint('keyword-dict', __name__)


@keyword_dict.route('/page', methods=['GET'])
def page():
    pageno = request.args.get('pageno') or None
    pagesize = request.args.get('pagesize') or None
    data = KeywordDictService.get_list(pageno, pagesize)
    totalnum = KeywordDictService.count()
    r = R.ok({'data':data,'totalnum':totalnum})
    return jsonify(r.to_json())

@keyword_dict.route('/save', methods=['POST'])
def save():
    params = request.json
    print(params)
    if params and KeywordDictService.save_one(params):
        r = R.ok("success")
    else:
        r = R.fail("fail")
    return jsonify(r.to_json())


@keyword_dict.route('/delete/<id>', methods=['GET'])
def delete(id):
    params = request.json
    print(params)
    if id and KeywordDictService.delete_one(id):
        r = R.ok("success")
    else:
        r = R.fail("fail")
    return jsonify(r.to_json())

@keyword_dict.route('/update', methods=['POST'])
def update():
    params = request.json
    if params and KeywordDictService.update_one(params):
        r = R.ok("success")
    else:
        r = R.fail("fail")
    return jsonify(r.to_json())
