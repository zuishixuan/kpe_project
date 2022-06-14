import re

from flask import Blueprint, request
from flask import jsonify
from app.common.Response import R
from app.model.kpe import keyword_extract_metadata, keyword_extract_cnki, keyword_extract_combine
from flask_cors import cross_origin
from flask_cors import CORS
from fuzzywuzzy import fuzz, process

kpe = Blueprint('kpe', __name__)

CORS(kpe)


@kpe.route('/extract', methods=['POST'])
# @cross_origin()
def kpe_metadata():
    params = request.json
    title = params.get("title", None)
    desc = params.get("desc", None)
    text = ','.join([title, desc])
    print(text)
    if text:
        r = R.ok(keyword_extract_combine(text))
    else:
        r = R.fail("fail")
    return jsonify(r.to_json())


@kpe.route('/eval', methods=['POST'])
# @cross_origin()
def eval_metadata():
    global rec
    params = request.json
    title = params.get("title", None)
    desc = params.get("desc", None)
    keyword = params.get("keyword", None)
    threshold = 59
    text = None
    if title and desc:
        text = ','.join([title, desc])
    if text:
        kpe = keyword_extract_combine(text)
        if keyword:
            pattern = '[;|；|(|)|（|）|/|、]'
            keyword_list = re.split(pattern, keyword)
            keyword_list = [k.strip() for k in keyword_list if len(k.strip()) != 0]
            keyword_eval = []
            kpe_eval = []
            hit = []
            for key in keyword_list:
                matchname = process.extractOne(key, kpe, scorer=fuzz.token_set_ratio)
                if matchname[1] > threshold: hit.append(matchname[0])
                keyword_eval.append({
                    "origin": key,
                    "target": matchname[0],
                    "score": matchname[1]
                })
            hit2 = []
            for key in kpe:
                matchname = process.extractOne(key, keyword_list, scorer=fuzz.token_set_ratio)
                if matchname[1] > threshold: hit2.append(matchname[0])
                kpe_eval.append({
                    "origin": key,
                    "target": matchname[0],
                    "score": matchname[1]
                })
            prec = len(hit) / len(keyword_list)
            rec = len(hit2) / len(kpe)
            f1 = 0
            if prec + rec != 0:
                f1 = 2 * prec * rec / (prec + rec)
            score = {
                "prec": prec,
                "rec": rec,
                "f1": f1
            }
            resul = {
                "keyword_eval": keyword_eval,
                "kpe": kpe,
                "score": score,
                "kpe_eval": kpe_eval,
                "threshold": threshold
            }
        else:
            resul = {
                "kpe": kpe
            }
        r = R.ok(resul)
    else:
        r = R.fail("fail")
    return jsonify(r.to_json())


@kpe.route('/extract_cnki', methods=['POST'])
# @cross_origin()
def kpe_cnki():
    params = request.json
    title = params.get("title", None)
    desc = params.get("desc", None)
    text = ','.join([title, desc])
    print(text)
    if text:
        r = R.ok(keyword_extract_cnki(text))
    else:
        r = R.fail("fail")
    return jsonify(r.to_json())
