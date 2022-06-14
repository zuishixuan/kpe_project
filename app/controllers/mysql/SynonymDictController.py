from flask import Blueprint, request
from flask import jsonify
from app.common.Response import R
from app.services.mysql.SynonymDictService import SynonymDictService

synonym_dict = Blueprint('synonym-dict', __name__)


@synonym_dict.route('/page', methods=['GET'])
def page():
    pageno = request.args.get('pageno') or None
    pagesize = request.args.get('pagesize') or None
    data = SynonymDictService.get_list(pageno, pagesize)
    totalnum = SynonymDictService.count()
    r = R.ok({'data':data,'totalnum':totalnum})
    return jsonify(r.to_json())

@synonym_dict.route('/save', methods=['POST'])
def save():
    params = request.json
    print(params)
    if params and SynonymDictService.save_one(params):
        r = R.ok("success")
    else:
        r = R.fail("fail")
    return jsonify(r.to_json())


@synonym_dict.route('/delete/<id>', methods=['GET'])
def delete(id):
    params = request.json
    print(params)
    if id and SynonymDictService.delete_one(id):
        r = R.ok("success")
    else:
        r = R.fail("fail")
    return jsonify(r.to_json())

@synonym_dict.route('/update', methods=['POST'])
def update():
    params = request.json
    print(params)
    if params and SynonymDictService.update_one(params):
        r = R.ok("success")
    else:
        r = R.fail("fail")
    return jsonify(r.to_json())
