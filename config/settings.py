import os

# 上线改完False
DEBUG = True

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DIALECT = 'mysql'
DRIVER = 'pymysql'
USERNAME = 'root'
PASSWORD = '123456'
HOST = 'localhost'
PORT = '3307'
DATABASE = 'kpe'
SQLALCHEMY_DATABASE_URI = "{}+{}://{}:{}@{}:{}/{}?charset=utf8".format(DIALECT, DRIVER, USERNAME, PASSWORD, HOST, PORT,
                                                                       DATABASE)
SQLALCHEMY_ECHO = True

settings = {
    'SQLALCHEMY_DATABASE_URI': SQLALCHEMY_DATABASE_URI,
    'SQLALCHEMY_TRACK_MODIFICATIONS': True,
    'SQLALCHEMY_ECHO': SQLALCHEMY_ECHO,
    'DEBUG': DEBUG,
    'SECRET_KEY': 'test',
    # cookie缓存时间
    'cookie_max_age': 60 * 60 * 24 * 2,
    'ENV': "development",
    'threaded': True,
    # 连接池回收时间，设置应该小于mysql的wait_timeout或者-1，
    # flask自身的设置会在一定时间之后主动断掉，-1表示永不回收；
    # Deprecated,use SQLALCHEMY_ENGINE_OPTIONS
    "SQLALCHEMY_POOL_RECYCLE ": 18000,  # 五个小时
    "SQLALCHEMY_ENGINE_OPTIONS": {
        'pool_size': 10,
        'pool_recycle': 120,
        'pool_pre_ping': True,
        "pool_timeout": 7200
    },
    #'ES_URI': 'http://124.207.169.14:8888',
    'ES_URI': 'http://localhost:92001',
    'elasticsearch': {
        # 数据导出用的es
        'hosts': ['10.2.2.62:9201'],
        # 项目自身的es
        #'hosts': ['39.106.49.131:9006'],
        # 'username': 'TechLink',
        # 'password': 'TechLink_Idf_HB'
    }
}

redis_config = {
    'REDIS_HOST': "localhost",
    'REDIS_PORT': "6379",
    'REDIS_TOEKN_DB': 15
}

USER_TOKEN_EXPIRE_TIME = 24 * 60 * 60
