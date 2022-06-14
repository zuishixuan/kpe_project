import os
import re

import networkx as nx
import pandas as pd
from txt_process import save_list_as_text
import jieba.posseg as pseg
import jieba
import constant
import numpy as np
from matplotlib import pyplot as plt


def load1():
    # 存数据
    for i in range(1,5):
        dir = './data/cnki/CNKI-20220415-'+str(i)+'.xls'
        # 存数据
        print(dir)
        df = pd.read_excel(dir,engine='xlrd')
        df_pro = pd.DataFrame()
        df_pro['name'] = df['题名']
        df_pro['keyword'] = df['关键词']
        df_pro['describe'] = df['摘要']
        print(df_pro)
        df_pro.to_csv('./data/cnki/data.csv')

def load2(dir):
    df = pd.read_csv(dir, index_col=0)
    df = df.dropna(axis=0, how='any')  # 删掉含有空值的行
    # def filter(x,y):
    #     return ','.join([i for i in x.split(',') if i in y])
    # df['keyword'] = df.apply(lambda x: filter(x['keyword'], x['text']), axis=1)
    df.to_csv('./data/cnki/data.csv')



def load3(file_dir):
    # 关键词库
    keywords = []
    df = pd.read_csv(file_dir, index_col=0)
    pattern = ';|；'
    for line in df['keyword']:
        res = re.split(pattern, line)
        res = [r.strip()  for r in res]
        keywords.extend(res)
    keywords = list(set(keywords))
    keywords = [word for word in keywords if len(word)>1]
    save_list_as_text(keywords, './data/cnki/keywords_dict.txt', ' n\n')
    print(keywords)

def main():
    file_dir = '../data/cnki/data.csv'
    load2(file_dir)


if __name__ == "__main__":
    main()
