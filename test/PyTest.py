# file_name: PyTest.py
# 引入pytest包
import time
from werkzeug.security import generate_password_hash, check_password_hash

import pytest


def test_1():
    date_1 = '2015-02-26 00:00:00'
    strptime = time.strptime(date_1, '%Y-%m-%d %H:%M:%S')
    print('------------------------------------------')
    print(strptime)
    print(strptime.tm_year)
    print(strptime.tm_mon)
    print(time.strftime("%Y-%m", strptime))


def test_a():  # test开头的测试函数
    print("------->test_a")
    assert 1  # 断言成功


def test_b():
    print("------->test_b")
    assert 0  # 断言失败

def test_c():
    hash = generate_password_hash("111111")
    print(hash)


if __name__ == '__main__':
    # 调用pytest的main函数执行测试
    # pytest.main(["-s", "PyTest.py"])
    print(test_c())
