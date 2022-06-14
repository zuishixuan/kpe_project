from flask import jsonify
from config import app
from app.common.Response import R
from app.common.MysqlBaseEntity import db
import os
# 注册路由
import router
# 日志信息
from logger import logger

app.router = router
app.my_logger = logger

mode = os.getenv('FLASK_CONFIG') or 'dev'
# 数据库
db.init_app(app)


@app.route('/')
def index():
    r = R.ok({
        'homepage': 'https://xxxx.com',
        'desc': '技术链路识别'
    })
    return jsonify(r.to_json())


# 接口跨域处理
@app.after_request
def cors(environ):
    environ.headers['Access-Control-Allow-Origin'] = '*'
    environ.headers['Access-Control-Allow-Method'] = '*'
    environ.headers['Access-Control-Allow-Headers'] = '*'
    return environ


@app.errorhandler(Exception)
def handler(exception):
    print(exception)
    msg = '捕获到异常信息:%s' % exception
    logger.error(msg)
    r = R.fail(msg)
    return jsonify(r.to_json())


if __name__ == '__main__':
    if mode != 'dev':
        from gevent import pywsgi

        server = pywsgi.WSGIServer(('0.0.0.0', 5000), app)
        server.serve_forever()
    else:
        from flask_script import Manager

        manager = Manager(app)
        manager.run()
