import urllib.parse

gb_url = "http://www.baidu.com"


def start():
    print("hello imxiaoqi\n")

    print("原值 = " + gb_url)

    # urlencode 编码操作
    gb_url_encode = urllib.parse.quote(gb_url)
    print("urlencode 编码后值 = " + gb_url_encode)
    print("\n")

    # urldecode 解码操作
    print("urldecode 解码前值 = " + gb_url_encode)
    gb_url_decode = urllib.parse.unquote(gb_url_encode)
    print("urldecode 解码后值 = " + gb_url_decode)


# 入口函数
if __name__ == '__main__':
    #start()
    print(urllib.parse.unquote("%E5%85%B3%E8%81%94%E8%B5%84%E6%BA%90/"))