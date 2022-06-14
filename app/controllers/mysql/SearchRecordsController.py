from flask import Blueprint, request
from flask import jsonify
from app.common.Response import R
from app.services.mysql.SearchRecordsService import SearchRecordsService

search_record = Blueprint('search-record', __name__)


@search_record.route('/search', methods=['GET'])
def search():
    be = request.args.get('be') or None
    end = request.args.get('end') or None
    print(be)
    print(end)
    res = SearchRecordsService.count_by_group_column_time_scope(be,end)
    print(res)
    r = R.ok(res)
    return jsonify(r.to_json())


@search_record.route('/save', methods=['POST'])
def save():
    params = request.json
    print(params)
    if params and SearchRecordsService.save_one(params):
        r = R.ok("success")
    else:
        r = R.fail("fail")
    return jsonify(r.to_json())