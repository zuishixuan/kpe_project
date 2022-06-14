from flask import Flask
import os
import config.settings
from logger import logger


def create_app(flask_config_name=None):
    """
    创建配置
    :return:
    """

    app = Flask(__name__, template_folder='../templates', static_folder="../static", static_url_path="/static")

    # 默认是开发
    settings = config.settings.settings
    debug_flag = config.settings.DEBUG

    env_mode = flask_config_name or os.getenv('FLASK_CONFIG') or 'dev'
    logger.info("Your config mode is %s" % env_mode)
    if env_mode == 'pro':
        from werkzeug.contrib.fixers import ProxyFix
        from config.pro_settings import settings as pro_settings

        app.wsgi_app = ProxyFix(app.wsgi_app)
        settings = pro_settings
        debug_flag = pro_settings.get("DEBUG") or False
    # 装载配置
    app.debug = debug_flag
    app.config.update(settings)

    return app, settings


app, settings = create_app()
