from flask import Blueprint, request
from flask import jsonify
from app.common.Response import R
from app.services.mysql.UserService import UserService
from app.utils.token import generate_token
from app.utils.RedisUtil import get_redis_cli, USER_TOKEN_EXPIRE_TIME
import json

user = Blueprint('user', __name__)


@user.route('/page', methods=['GET'])
def page():
    pageno = request.args.get('pageno') or None
    pagesize = request.args.get('pagesize') or None
    data = UserService.get_list(pageno, pagesize)
    totalnum = UserService.count()
    r = R.ok({'data': data, 'totalnum': totalnum})
    return jsonify(r.to_json())


@user.route('/detail/<user_id>', methods=['GET'])
def user_detail(user_id):
    r = R.ok(UserService.get_detail(user_id))
    return jsonify(r.to_json())


@user.route('/detail_by_username/<username>', methods=['GET'])
def user_detail_by_username(username):
    r = R.ok(UserService.get_detail_by_username(username))
    return jsonify(r.to_json())


@user.route('/login', methods=['POST'])
def login():
    params = request.json
    username = params.get("username", None)
    pwd = params.get("password", None)
    if username and pwd and UserService.login(username, pwd):
        token = generate_token(username)
        userinfo = {"username": username}
        res = {
            'token': token
        }
        # redis待补充
        redis_cli = get_redis_cli()
        redis_cli["token"].set(name=token, value=json.dumps(userinfo), ex=USER_TOKEN_EXPIRE_TIME)
        r = R.ok(res)
    else:
        r = R.fail("登陆失败")
    return jsonify(r.to_json())


@user.route('/info/<token>', methods=['GET'])
def user_info(token):
    redis_cli = get_redis_cli()
    info = redis_cli["token"].get(token)
    if info:
        username = json.loads(info)['username']
        detail = UserService.get_detail_by_username(username)
        data = {
            "roles": [detail['role']],
            "introduction": "this is introduction",
            "name": detail['username'],
            "avatar": "https://wpimg.wallstcn.com/f778738c-e4f8-4870-b634-56703b4acafe.gif",
        }
        r = R.ok(data)
    else:
        r = R.fail('fail')
    return jsonify(r.to_json())


@user.route('/info/username/<username>', methods=['GET'])
def user_info_by_username(username):
    detail = UserService.get_detail_by_username(username)
    data = {
        "roles": [detail['role']],
        "introduction": "this is introduction",
        "name": detail['username'],
        "avatar": "https://wpimg.wallstcn.com/f778738c-e4f8-4870-b634-56703b4acafe.gif",
    }
    r = R.ok(data)
    return jsonify(r.to_json())


@user.route('/logout', methods=['GET'])
def logout():
    r = R.ok('')
    return jsonify(r.to_json())


@user.route('/save', methods=['POST'])
def save():
    params = request.json
    print(params)
    if params and UserService.save_one(params):
        r = R.ok("success")
    else:
        r = R.fail("fail")
    return jsonify(r.to_json())


@user.route('/delete/<id>', methods=['GET'])
def delete(id):
    params = request.json
    print(params)
    if id and UserService.delete_one(id):
        r = R.ok("success")
    else:
        r = R.fail("fail")
    return jsonify(r.to_json())

@user.route('/update', methods=['POST'])
def update():
    params = request.json
    print(params)
    if params and UserService.update_one(params):
        r = R.ok("success")
    else:
        r = R.fail("fail")
    return jsonify(r.to_json())
