from flask import Blueprint, request
from flask import jsonify
from app.common.Response import R
from app.services.mysql.KpeRecordService import KpeRecordService

kpe_record = Blueprint('kpe-record', __name__)


@kpe_record.route('/page', methods=['GET'])
def page():
    pageno = request.args.get('pageno') or None
    pagesize = request.args.get('pagesize') or None
    data = KpeRecordService.get_list(pageno, pagesize)
    totalnum = KpeRecordService.count()
    r = R.ok({'data': data, 'totalnum': totalnum})
    return jsonify(r.to_json())


@kpe_record.route('/save', methods=['POST'])
def save():
    params = request.json
    print(params)
    if not params:
        r = R.fail("fail")
    id = KpeRecordService.save_one_with_id_return(params)
    if id != -1:
        r = R.ok(id)
    else:
        r = R.fail("fail")
    return jsonify(r.to_json())


@kpe_record.route('/delete/<id>', methods=['GET'])
def delete(id):
    params = request.json
    print(params)
    if id and KpeRecordService.delete_one(id):
        r = R.ok("success")
    else:
        r = R.fail("fail")
    return jsonify(r.to_json())


@kpe_record.route('/update', methods=['POST'])
def update():
    params = request.json
    if params and KpeRecordService.update_one(params):
        r = R.ok("success")
    else:
        r = R.fail("fail")
    return jsonify(r.to_json())
