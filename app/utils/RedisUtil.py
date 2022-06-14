from config.settings import redis_config,USER_TOKEN_EXPIRE_TIME
import redis

def get_redis_cli():
    redis_cli = {
        "token": redis.StrictRedis(host=redis_config['REDIS_HOST'], port=redis_config['REDIS_PORT']
                                   , db=redis_config['REDIS_TOEKN_DB'])
    }
    return redis_cli
