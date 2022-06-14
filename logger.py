import logging.config
import os
from logging import handlers
import time

# PermissionError: [WinError 32] 另一个程序正在使用此文件，进程无法访问。
# from concurrent_log_handler import ConcurrentRotatingFileHandler
#
# logfile = os.path.abspath("mylogfile.log")
# # Rotate log after reaching 512K, keep 5 old copies.
# rotateHandler = ConcurrentRotatingFileHandler(logfile, "a", 512*1024, 5)
# 日志级别关系映射
level_relations = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'crit': logging.CRITICAL
}


class MyLogger:
    file_name = '/data/logs/info.' + time.strftime('%Y-%m-%d', time.localtime(time.time())) + '.log'
    # 获取当前的文件路径
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    fmt = '%(asctime)s - %(pathname)s[func:%(funcName)s line:%(lineno)d] - %(levelname)s: %(message)s'

    def __init__(self, default_level=logging.INFO,
                 env_key="LOG_CFG"):
        self.default_level = default_level
        self.env_key = env_key
        self.my_logger = logging.getLogger()
        self.setup_logging()

    def setup_logging(self):
        # 日志重复打印 [ 判断是否已经有这个对象，有的话，就再重新添加]
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        if not self.my_logger.handlers:
            level = 'info'
            when = 'D'
            # backupCount：允许存储的文件个数，如果大于这个数系统会删除最早的日志文件
            back_count = 0,
            # 设置日志格式
            format_str = logging.Formatter(self.fmt)
            # 设置日志级别
            self.my_logger.setLevel(level_relations.get(level))
            # 往屏幕上输出
            stream_handler = logging.StreamHandler()
            # 设置屏幕上显示的格式
            stream_handler.setFormatter(format_str)
            # 往文件里写入#指定间隔时间自动生成文件的处理器
            log_full_path = self.log_path + self.file_name
            time_handler = handlers.TimedRotatingFileHandler(
                filename=log_full_path,
                when=when,
                backupCount=back_count,
                encoding='utf-8'
            )
            # S 秒
            # M 分
            # H 小时、
            # D 天、
            # W 每星期（interval==0时代表星期一）
            # midnight 每天凌晨
            # 设置文件里写入的格式
            time_handler.setFormatter(format_str)
            # 把对象加到logger里
            self.my_logger.addHandler(stream_handler)
            self.my_logger.addHandler(time_handler)
            self.stream_handler = stream_handler
            self.time_handler = time_handler

    def info(self, message, *args, **kwargs):
        self.my_logger.info(msg=message, *args, **kwargs)
        self.post_handler()

    def warn(self, message, *args, **kwargs):
        self.my_logger.warning(msg=message, *args, **kwargs)
        self.post_handler()

    def debug(self, message, *args, **kwargs):
        self.my_logger.debug(msg=message, *args, **kwargs)
        self.post_handler()

    def error(self, message, *args, **kwargs):
        self.my_logger.error(msg=message, *args, **kwargs)
        self.post_handler()

    def exception(self, message, *args, **kwargs):
        self.my_logger.exception(msg=message, *args, **kwargs)
        self.post_handler()

    def post_handler(self):
        #  添加下面一句，在记录日志之后移除句柄
        # self.my_logger.removeHandler(self.stream_handler)
        # self.my_logger.removeHandler(self.time_handler)
        # error_file_handler = logging.getLogger("error_file_handler")
        # self.my_logger.removeHandler(error_file_handler)
        pass


logger = MyLogger().my_logger


def get_logger():
    return logger


def init_print():
    logger.info("日志模块启动")


if __name__ == "__main__":
    init_print()
    pass
