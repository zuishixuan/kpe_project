import argparse
import re
from enum import Enum

import pandas as pd
from tqdm import tqdm
from constant import *

import TextRank4Keyword

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--file_dir", type=str, default='../data/cnki_2/data_filter.csv')
    parser.add_argument("--pretrained", type=str, default="hfl/chinese-bert-wwm-ext")
    parser.add_argument("--user_dict", type=str, default='../data/cnki_2/keywords_dict.txt')
    parser.add_argument("--simcse_model_path", type=str, default="./model/SimCSE/cnki/out/epoch_1-batch_500")
    parser.add_argument("--exc_num", type=int, default=3)
    parser.add_argument("--window", type=int, default=2)
    args = parser.parse_args()
    return args


def eval(args):
    print(args.exc_num)
    print(args.window)
    user_dict = args.user_dict
    tr4w = TextRank4Keyword.TextRank4Keyword(simcse_model_path=args.simcse_model_path, user_dict=user_dict)
    df = pd.read_csv(args.file_dir)
    acl_acc = 0
    acl_rec = 0
    i = 0
    p_temp = 0
    r_temp = 0
    f_temp = 0
    for item in tqdm(df.itertuples()):
        pattren = r',|;|；| '
        print(item.keyword)
        keywords = re.split(pattern=pattren,string=item.keyword)
        #keywords = item.keyword.split(',')
        if len(keywords) == 0: continue
        i += 1
        text = item.text
        tr4w.analyze(text=text, lower=True, window=args.window, pagerank_config={'alpha': 0.85})
        exc = tr4w.get_keywords(args.exc_num, word_min_len=2)
        exc_keyword = [word['word'] for word in exc]
        acc = sum([k in exc_keyword for k in keywords]) / args.exc_num
        rec = sum([k in exc_keyword for k in keywords]) / len(keywords)
        # print('P=')
        # print(acc)
        # print('R=')
        # print(rec)
        acl_acc += acc
        acl_rec += rec
        p_temp = acl_acc / i
        r_temp = acl_rec / i
        f_temp = 2 * p_temp * r_temp / (p_temp + r_temp)
        print("当前平均精确率P=")
        print(p_temp)
        print("当前平均召回率R=")
        print(r_temp)
        print("当前平均F值=")
        print(f_temp)
    print("最终精确率P=")
    print(p_temp)
    print("最终召回率R=")
    print(r_temp)
    print("最终F值=")
    print(f_temp)


def kpe(text):
    # text = codecs.open('./doc/02.txt', 'r', 'utf-8').read()
    # text = "世界的美好。世界美国英国。 世界和平。"

    tr4w = TextRank4Keyword.TextRank4Keyword(simcse_model_path="./model/SimCSE/cnki/out/epoch_1-batch_500")
    tr4w.analyze(text=text, lower=True, window=3, pagerank_config={'alpha': 0.85})

    print('关键词前五：')
    for item in tr4w.get_keywords(5, word_min_len=2):
        print(item.word, item.weight, type(item.word))

    print('关键词短语：')

    for phrase in tr4w.get_keyphrases(keywords_num=5, min_occur_num=0):
        print(phrase, type(phrase))


def main():
    args = parse_args()
    eval(args)


if __name__ == "__main__":
    main()
