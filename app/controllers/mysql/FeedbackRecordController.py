from flask import Blueprint, request
from flask import jsonify
from app.common.Response import R
from app.services.mysql.FeedbackRecordService import FeedbackRecordService

feedback_record = Blueprint('feedback-record', __name__)


@feedback_record.route('/page', methods=['GET'])
def page():
    pageno = request.args.get('pageno') or None
    pagesize = request.args.get('pagesize') or None
    data = FeedbackRecordService.get_list(pageno, pagesize)
    totalnum = FeedbackRecordService.count()
    r = R.ok({'data':data,'totalnum':totalnum})
    return jsonify(r.to_json())

@feedback_record.route('/save', methods=['POST'])
def save():
    params = request.json
    print(params)
    if params and FeedbackRecordService.save_one(params):
        r = R.ok("success")
    else:
        r = R.fail("fail")
    return jsonify(r.to_json())


@feedback_record.route('/delete/<id>', methods=['GET'])
def delete(id):
    params = request.json
    print(params)
    if id and FeedbackRecordService.delete_one(id):
        r = R.ok("success")
    else:
        r = R.fail("fail")
    return jsonify(r.to_json())

@feedback_record.route('/update', methods=['POST'])
def update():
    params = request.json
    if params and FeedbackRecordService.update_one(params):
        r = R.ok("success")
    else:
        r = R.fail("fail")
    return jsonify(r.to_json())

@feedback_record.route('/get-by-kpe-record-id/<kpe_record_id>', methods=['GET'])
def get_list_by_kpe_record_id(kpe_record_id):
    if kpe_record_id:
        r = R.ok(FeedbackRecordService.get_list_by_kpe_record_id(kpe_record_id))
    else:
        r = R.fail("fail")
    return jsonify(r.to_json())