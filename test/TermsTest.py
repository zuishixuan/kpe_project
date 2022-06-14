#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 分词统计词频
import jieba
import re
from collections import Counter

filename = r"./data/term_test.txt"
result = "result_com.txt"
r = '[0-9\s+\.\!\/_,$%^*()?;；:-【】+\"\']+|[+——！，;：。？、 ~@#￥%……&*（）]+'
with open(filename, 'r', encoding='utf-8') as fr:
    print("ss")
    content = re.sub(r, " ", fr.read())
    # re.sub(pattern, repl, string, count=0, flags=0)
    # pattern：表示正则表达式中的模式字符串；
    # repl：被替换的字符串（既可以是字符串，也可以是函数）；
    # string：要被处理的，要被替换的字符串；
    # count：匹配的次数, 默认是全部替换
    # flags：具体用处不详
    data = jieba.cut(content, cut_all=False)
    print(data)
    data = dict(Counter(data))  # dict() 函数用于创建一个字典。Counter 是实现的 dict 的一个子类，可以用来方便地计数。
    data2 = list(Counter(data))
    print('data2data2data2')
    print(data2)
    with open(result, 'w', encoding="utf-8") as fw:
        for k, v in data.items():
            if len(k) > 1:
                fw.write(k)
                fw.write("\t%d\n" % v)
