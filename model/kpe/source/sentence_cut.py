import pandas as pd
import logging
import jieba
import jieba.posseg as psg

from constant import *

# 创建停用词列表
def get_stopword_list():
    stopwords = [line.strip() for line in open(stopwords_path, encoding='UTF-8').readlines()]
    # stopwords_plus =  [line.strip() for line in open(stopwords_plus_path, encoding='UTF-8').readlines()]
    return stopwords # +stopwords_plus