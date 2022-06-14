from flask import Blueprint, request
from flask import jsonify
from app.common.Response import R
from app.services.mysql.MetadataService import MetadataService
from app.services.es.EsMetadataService import EsMetadataService
from app.services.mysql.SearchRecordsService import SearchRecordsService
import datetime

es_metadata = Blueprint('es-metadata', __name__)


@es_metadata.route('/page', methods=['GET'])
def page():
    pageno = request.args.get('pageno') or None
    pagesize = request.args.get('pagesize') or None
    print(pageno)
    print(pagesize)
    res = EsMetadataService.get_page(int(pageno), int(pagesize))
    r = R.ok({'data': res['records'], 'totalnum': res['total']})
    return jsonify(r.to_json())


@es_metadata.route('/match-page', methods=['GET'])
def match_page():
    pageno = request.args.get('pageno') or None
    pagesize = request.args.get('pagesize') or None
    search_word = request.args.get('search_word') or None
    search_select = request.args.get('search_select') or None
    username = request.args.get('username') or None
    # ftime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    record = {
        'word': search_word,
        'username': username
    }
    re = SearchRecordsService.save_one(record)
    print(re)

    res = EsMetadataService.get_query_page(int(pageno), int(pagesize), search_select, search_word)
    r = R.ok({'data': res['records'], 'totalnum': res['total']})
    return jsonify(r.to_json())

# @metadata.route('/save', methods=['POST'])
# def save():
#     params = request.json
#     # name = params.get("title", None)
#     # subject_category = params.get("subject_category", None)
#     # theme_category = params.get("theme_category", None)
#     # describe = params.get("describe", None)
#     # keyword = params.get("keyword", None)
#     # keyword_deny = params.get("keyword_deny", None)
#     print(params)
#     if params and MetadataService.save_one(params):
#         EsMetadataRepository().save(params)
#         r = R.ok("success")
#     else:
#         r = R.fail("fail")
#     return jsonify(r.to_json())
