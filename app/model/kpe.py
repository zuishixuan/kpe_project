# -*- encoding:utf-8 -*-
from __future__ import print_function

import sys

import codecs
from model.kpe.source.TextRank4Keyword import TextRank4Keyword
from model.kpe.source import model_metadata, model_cnki


def keyword_extract_metadata(text):
    model = model_metadata
    print(text)

    keyword_kpe = model.kpe(text=text, has_score=False, min_num=3, supplement=True)
    print("basic:")
    for key in keyword_kpe:
        print(key)
    return keyword_kpe


def keyword_extract_cnki(text):
    model = model_cnki
    print(text)

    keyword_kpe = model.kpe(text=text, has_score=False, min_num=3, supplement=True)
    print("basic:")
    for key in keyword_kpe:
        print(key)

    return keyword_kpe


def keyword_extract_combine(text):
    print(text)
    threshold = 0.5

    keyword_kpe_metadata = model_metadata.kpe(text=text, has_score=True, min_num=3, supplement=True)
    keyword_kpe_cnki = model_cnki.kpe(text=text, has_score=True, min_num=3, supplement=True)

    for i in keyword_kpe_cnki:
        for j in reversed(keyword_kpe_metadata):
            if j[0] == i[0]:
                keyword_kpe_metadata.remove(j)
    keyword_kpe = []
    keyword_kpe.extend(keyword_kpe_cnki)
    keyword_kpe.extend(keyword_kpe_metadata)
    keyword_kpe.sort(key=lambda x: (x[1]), reverse=True)
    remo = set()
    for i in keyword_kpe:
        for j in keyword_kpe:
            print(i, j)
            if i[0] in j[0] and j[0] not in i[0]:
                count_i = text.count(i[0])
                count_j = text.count(j[0])
                print("ccccccccccccccccccccc")
                print(count_i, count_j)
                print(i, j)
                if count_j / count_i > threshold:
                    if i not in remo:
                        remo.add(i)
    print(remo)
    for r in remo:
        if r in keyword_kpe:
            keyword_kpe.remove(r)

    keyword_kpe = [r[0] for r in keyword_kpe if not r[0].encode('UTF-8').isalnum()]

    print("basic:")
    for key in keyword_kpe:
        print(key)

    return keyword_kpe
