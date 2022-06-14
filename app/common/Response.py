import json
import decimal


class R:
    """统一返回体"""
    '消息'
    msg = 'success'
    '状态码'
    code = 200
    '内容'
    data = None

    def __init__(self, code, msg):
        self.code = code
        self.msg = msg
        self.cost = 0

    def set_cost(self, value):
        self.cost = value

    @staticmethod
    def ok(data):
        r = R(200, 'success')
        r.data = data
        return r

    @staticmethod
    def fail(msg):
        r = R(500, msg)
        return r

    def to_json(self):
        """将实例对象转化为json"""
        item = self.__dict__
        return item


class DecimalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, decimal.Decimal):
            return float(o)
        super(DecimalEncoder, self).default(o)
