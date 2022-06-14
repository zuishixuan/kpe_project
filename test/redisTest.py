from app.utils.RedisUtil import get_redis_cli
from werkzeug.security import generate_password_hash, check_password_hash
import pytest
import datetime

if __name__ == '__main__':
    # redis_cli = get_redis_cli()
    # redis_cli["token"].set(name='123', value='456', ex=1234)
    # ans = redis_cli["token"].get('123')
    # print(ans)
    now_time = datetime.datetime.now()
    print(now_time)
    ftime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(ftime)