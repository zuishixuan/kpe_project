import re

import networkx as nx
import pandas as pd
import torch

from .txt_process import save_list_as_text, get_list_by_txt
from . import constant
import numpy as np
from matplotlib import pyplot as plt
from .constant import tag_map
from .Segmentation import Segmentation


def load1():
    # 存数据
    df = pd.read_csv('./data/data_1m.csv', index_col=0)
    df['text'] = df.apply(lambda x: (x["name"] + '。' + x["describe"]), axis=1)
    df.to_csv('./data/data_1m_2.csv')


def load2():
    # 去重
    df = pd.read_csv('../data/data2.csv', index_col=0)
    newdata = df.drop_duplicates(subset=['identifier'], keep='first')
    newdata.to_csv('./data/data3.csv')


def load3(file_dir):
    # 关键词库
    keywords = []
    df = pd.read_csv(file_dir, index_col=0)
    pattern = ';|；'
    for line in df['keyword']:
        res = re.split(pattern, line)
        res = [r.strip() for r in res]
        keywords.extend(res)
    df = pd.read_csv('./data/data_1m.csv', index_col=0)
    for line in df['keyword']:
        res = re.split(pattern, line)
        res = [r.strip() for r in res]
        keywords.extend(res)
    keywords = list(set(keywords))
    keywords = [word for word in keywords if len(word) > 1]
    save_list_as_text(keywords, './data/keywords_dict.txt', ' n\n')
    print(keywords)


def load4(file_dir):
    # 关键词库
    keywords = []
    df = pd.read_csv(file_dir, index_col=0)
    pattern = ';|；'
    for line in df['keyword']:
        res = re.split(pattern, line)
        res = [r.strip() for r in res]
        keywords.extend(res)
    keywords = list(set(keywords))
    keywords = [word for word in keywords if len(word) > 1]
    save_list_as_text(keywords, './data/keywords_dict.txt', ' n\n')
    print(keywords)


def load5():
    # 统计学科分类
    category = {}
    df = pd.read_csv('./data/data_1m.csv', index_col=0)
    for line in df['subject_category']:
        # print(line)
        res = line.split(' ')
        # print(res)
        for re in res:
            if re in constant.subject_category_correct:
                re = constant.subject_category_correct[re]
            if re in category:
                category[re] += 1
            else:
                category[re] = 1
    print(category)
    category_sort = sorted(category.items(), key=lambda x: -x[1])
    print(category_sort)
    print(len(category))
    df2 = pd.read_csv('../data/subject_category.csv')
    print(df2['学科分类名称'].values)
    for ca in category:
        if ca not in df2['学科分类名称'].values:
            print(ca)
            print(category[ca])


def load6():
    # 存数据
    df = pd.ExcelFile('../data/元数据列表.xlsx')
    print(df.sheet_names)  # 查看所有sheet 名字
    df_concat = pd.concat([pd.read_excel(df, sheet) for sheet in df.sheet_names])
    # 将所有sheet中数据合并到一个df中
    df_pro = pd.DataFrame()
    df_pro['name'] = df_concat['中文名称']
    df_pro['identifier'] = df_concat['标识符']
    df_pro['subject_category'] = df_concat['学科分类']
    df_pro['theme_category'] = df_concat['主题分类']
    df_pro['keyword'] = df_concat['关键词']
    df_pro['describe'] = df_concat['描述']
    df_pro['generate_date'] = df_concat['资源生成日期']
    df_pro['submit_date'] = df_concat['最近发布日期']
    print(df_pro)
    df_pro.to_csv('./data/data_1m.csv')


def load7():
    category = {}
    df = pd.read_csv('./data/data_1m.csv')
    df.drop('Unnamed: 0', axis=1, inplace=True)
    print(df)
    df.to_csv('./data/data_1m.csv')


def load8(file_dir):
    category = {}
    df = pd.read_csv(file_dir, index_col=0)
    print(df['text'][10])
    # df['subject_class'] = df.apply(lambda x: get_class(x['subject_category']), axis=1)
    # print(df)
    # df.to_csv('./data/data4.csv')


def get_class(line):
    res = line.split(' ')
    # print(res)
    resul = []
    for re in res:
        if re in constant.subject_category_correct:
            re = constant.subject_category_correct[re]
        if re in constant.subject_category_tag:
            resul.append(constant.subject_category_tag[re])
        else:
            if constant.subject_category_tag['其他'] not in resul:
                resul.append(constant.subject_category_tag['其他'])
    resul = [str(i) for i in resul]
    return ' '.join(resul)


def graph_t():
    A = np.array([[0, 0.1], [0.1, 0]])
    G = nx.from_numpy_matrix(A)
    # print(G[1][1])
    res = nx.pagerank(G)
    print(res)

    nx.draw_networkx(G, with_labels=True, font_weight='bold')
    # 这个draw_shell好像是按照某种叫shell的布局绘制

    plt.show()


def get_sim_matrix(g, dict):
    print('')


def cnki_2_load(file_dir):
    # 存数据
    df = pd.read_excel(file_dir)
    # print(df)  # 查看所有sheet 名字
    # df_concat = pd.concat([pd.read_excel(df, sheet) for sheet in df.sheet_names])
    print(df)

    # 将所有sheet中数据合并到一个df中
    def process(st, form):
        pattern = ' |《|》|' + form
        # st = st.replace(' ', '')
        st = re.sub(pattern, '', st)
        if st[-1] == ';':
            st = st[0:len(st) - 1]
        # if st[0:len(form)] == form:
        #     return st[len(form):]
        # else:
        return st

    df_pro = pd.DataFrame()
    df_pro['name'] = df['标题']
    df_pro['subject_category'] = df['分类']
    df_pro['keyword'] = df['关键词']
    df_pro['describe'] = df['摘要']
    df_pro = df_pro[df_pro['keyword'].str[0:4] == '关键词：']
    df_pro = df_pro[df_pro['describe'].str[0:3] == '摘要：']
    df_pro['keyword'] = df_pro.apply(lambda x: process(x['keyword'], '关键词：'), axis=1)
    df_pro['describe'] = df_pro.apply(lambda x: process(x['describe'], '摘要：'), axis=1)
    df_pro['text'] = df_pro.apply(lambda x: x['name'] + '。' + x['describe'], axis=1)

    def filter(keyword, text):
        return ';'.join([w for w in keyword.split(';') if w in text])

    df_pro['keyword'] = df_pro.apply(lambda x: filter(x['keyword'], x['text']), axis=1)
    # df_pro['length'] = df_pro.apply(lambda x: len(x['text']), axis=1)
    df_pro['num'] = df_pro.apply(lambda x: len(x['keyword'].split(";")), axis=1)
    df_pro.dropna(axis=0, subset=['keyword'], inplace=True)
    #
    # print(df_pro['length'])
    # print(df_pro['num'])
    print(df_pro.describe())
    df_pro.to_csv('../data/cnki_2/data_filter.csv', index=None)


def cnki_2_keywords_dict(file_dir, out_dir):
    # 关键词库
    file_name = '/keywords_dict.txt'
    keywords = []
    df = pd.read_csv(file_dir, index_col=0)
    pattern = ';|；'
    for line in df['keyword']:
        res = re.split(pattern, line)
        res = [r.strip() for r in res]
        keywords.extend(res)
    keywords = list(set(keywords))
    keywords = [word for word in keywords if len(word) > 1]
    save_list_as_text(keywords, out_dir + file_name, ' n\n')
    print(keywords)


def remove_nan(file_dir):
    df = pd.read_csv(file_dir)
    df.dropna(axis=0, subset=['keyword'], inplace=True)
    print(df.describe())
    df.to_csv('../data/cnki_2/data_filter.csv', index=None)


# 提取标签
def get_tag(file_dir):
    df = pd.read_csv(file_dir)
    df.dropna(axis=0, subset=['subject_category'], inplace=True)
    tag = []

    def get_cate(line):
        res = line.split(';')
        for re in res:
            if re and re[0].isalpha():
                if re[0] == 'T':
                    tag.append(re[0:2])
                else:
                    tag.append(re[0:1])
        return

    df['tag'] = df.apply(lambda x: get_cate(x['subject_category']), axis=1)

    print(tag)
    # print(df['tag'])
    s = set(tag)
    print(s)
    print(len(s))
    dict = {}
    for d in tag:
        if d not in dict:
            dict[d] = 1
        else:
            dict[d] += 1
    print(dict)
    category_sort = sorted(dict.items(), key=lambda x: -x[1])
    print(category_sort)
    # df.to_csv('../data/cnki_2/data_filter.csv', index=None)


# 提取标签
def get_mark(file_dir, out_dir):
    def padding(li, maxlen=300, pad_token=''):
        le = len(li)
        if len(li) < maxlen:
            li.extend([pad_token for i in range(maxlen - le)])
        else:
            if len(li) > maxlen:
                li = li[0:maxlen]
                le = maxlen
        return li, le

    def get_seg(text):
        max_len = 500
        max_word_count = 300
        seg = Segmentation(user_dict=None)

        text = text
        text = text[:min(len(text), max_len)]

        word_segments = seg.segment(text=text)['words_no_filter']

        word_li = []
        for li in word_segments:
            word_li.extend(li)
        res = ';'.join(word_li)
        # print(res)
        return res

    def get_mark(text, keywords):
        max_len = 500
        max_word_count = 300
        seg = Segmentation(user_dict=None)

        text = text
        text = text[:min(len(text), max_len)]
        pattern = ';|；| '
        keywords = keywords
        res = re.split(pattern, keywords)
        keyword_list = [r.strip() for r in res]

        word_segments = seg.segment(text=text)['words_no_filter']
        keyword_list_seg = [seg.segment(text=w)['words_no_filter'][0] for w in keyword_list]

        word_li = []
        for li in word_segments:
            word_li.extend(li)

        word_list_len = len(word_li)

        label = [0] * word_list_len  # torch.zeros(max_word_count, dtype=torch.long)
        for ws in keyword_list_seg:
            leng = len(ws)
            for i in range(word_list_len - leng + 1):
                if ws == word_li[i:i + leng]:
                    if leng == 1:
                        label[i] = tag_map['S-KEY']
                    if leng == 2:
                        label[i] = tag_map['B-KEY']
                        label[i + 1] = tag_map['E-KEY']
                    if leng > 2:
                        label[i] = tag_map['B-KEY']
                        label[i + leng - 1] = tag_map['E-KEY']
                        label[i + 1:i + leng - 1] = [tag_map['I-KEY'] for k in range(i + 1, i + leng - 1)]

        mark_label = [str(l) for l in label]
        res = ';'.join(mark_label)
        # print(res)
        return res

    df = pd.read_csv(file_dir)
    df['seg'] = df.apply(lambda x: get_seg(x['text']), axis=1)
    df['mark'] = df.apply(lambda x: get_mark(x['text'], x['keyword']), axis=1)
    print(df['seg'])
    print(df['mark'])
    df.to_csv(out_dir, index=None)


def metadata_fileter(file_dir, out_dir):
    df_pro = pd.read_csv(file_dir, index_col=0)
    pattern = ';|；| '
    df_pro['num'] = df_pro.apply(lambda x: len(re.split(pattern, x['keyword'])), axis=1)

    print(df_pro.describe())

    def filter(keyword, text):
        pattern = ';|；| '
        keywords = keyword
        res = re.split(pattern, keywords)
        res = [str.strip(r) for r in res if str.strip(r) != '']
        res = [w for w in res if w in text]
        if len(res) != 0:
            return ';'.join([w for w in res if w in text])
        else:
            return np.nan

    df_pro['keyword'] = df_pro.apply(lambda x: filter(x['keyword'], x['text']), axis=1)
    df_pro.dropna(axis=0, subset=['keyword'], inplace=True)
    df_pro['num'] = df_pro.apply(lambda x: len(re.split(pattern, x['keyword'])), axis=1)
    #
    # print(df_pro['length'])
    # print(df_pro['num'])
    print(df_pro.describe())

    # df_pro.to_csv('../data/cnki_2/data_filter.csv', index=None)
    df_pro.to_csv(out_dir, index=None)


def data_combine():
    file_dir_1 = '../data/cnki_2/data_mark.csv'
    file_dir_2 = '../data/metadata_all/data_mark.csv'
    out_dir = '../data/combine_2/data_mark.csv'
    file_dir_1 = constant.get_real_dir(file_dir_1)
    file_dir_2 = constant.get_real_dir(file_dir_2)
    out_dir = constant.get_real_dir(out_dir)
    df1 = pd.read_csv(file_dir_1)
    df2 = pd.read_csv(file_dir_2)
    df_out = pd.concat([df1, df2])
    print(df1.describe())
    print(df2.describe())
    print(df_out.describe())
    df_out.to_csv(out_dir, index=None)


def data_combine2():
    file_dir_1 = '../data/metadata_1/data_mark.csv'
    file_dir_2 = '../data/metadata_2/data_mark.csv'
    out_dir1 = '../data/metadata_all/data_mark.csv'
    out_dir2 = '../data/metadata_test/data_mark.csv'
    file_dir_1 = constant.get_real_dir(file_dir_1)
    file_dir_2 = constant.get_real_dir(file_dir_2)
    out_dir_1 = constant.get_real_dir(out_dir1)
    out_dir_2 = constant.get_real_dir(out_dir2)
    df1 = pd.read_csv(file_dir_1)
    df2 = pd.read_csv(file_dir_2)
    # 按100%的比例抽样即达到打乱数据的效果
    df2 = df2.sample(frac=1.0)
    # 打乱数据之后index也是乱的，如果你的index没有特征意义的话，直接重置就可以了，否则就在打乱之前把index加进新的一列，再生成无意义的index
    df2 = df2.reset_index()
    train = df2.loc[0:17000]
    test = df2.loc[17000 + 1:]

    df_out = pd.concat([df1, train])
    df_out = df_out.sample(frac=1.0)
    # 打乱数据之后index也是乱的，如果你的index没有特征意义的话，直接重置就可以了，否则就在打乱之前把index加进新的一列，再生成无意义的index
    df_out = df_out.reset_index()
    print(test)

    print(df_out.describe())
    print(test.describe())
    df_out.to_csv(out_dir_1, index=None)
    test.to_csv(out_dir_2, index=None)


def keyword_dict(file_dir=None):
    # 关键词库
    file_dir = '../data/keyword_dict/keyword_dict.txt'
    out_dir = '../data/keyword_dict/keyword_dict_filter.csv'
    file_dir = constant.get_real_dir(file_dir)
    out_dir = constant.get_real_dir(out_dir)
    keyword_li = get_list_by_txt(file_dir)
    keyword_li2 = []
    for w in keyword_li:
        w = w.strip()
        # print(w)
        pattern = '[;|；|(|)|（|）|/|、]'
        res = re.split(pattern, w)
        # print(res)
        keyword_li2.extend(res)
    keyword_li3 = []
    for w in keyword_li2:
        if w is None: continue
        w = w.strip()
        if len(w) < 2: continue
        if w.isdigit(): continue
        keyword_li3.append(w)
    df = pd.DataFrame({'word':keyword_li3})
    df['status'] = 1
    df['add_date'] = '2022/5/20'
    df['weight'] = 5
    df.to_csv(out_dir)
    #save_list_as_text(keyword_li3, out_dir, ' n\n')
    # if w.isalpha(): print(w)
    # if w.isalnum(): print(w)
    # keywords = []
    # df = pd.read_csv(file_dir, index_col=0)
    # pattern = ';|；'
    # for line in df['keyword']:
    #     res = re.split(pattern, line)
    #     res = [r.strip() for r in res]
    #     keywords.extend(res)
    # keywords = list(set(keywords))
    # keywords = [word for word in keywords if len(word) > 1]
    # save_list_as_text(keywords, './data/keywords_dict.txt', ' n\n')
    # print(keywords)


def main():
    file_dir = '../data/metadata_2/data_filter.csv'
    out_dir = '../data/metadata_2/data_mark.csv'
    file_dir = constant.get_real_dir(file_dir)
    out_dir = constant.get_real_dir(out_dir)
    # cnki_2_keywords_dict(file_dir, out_dir)
    # file_dir = '../data/cnki_2/论文数据集.xlsx'
    # metadata_fileter(file_dir, out_dir)
    # get_mark(file_dir, out_dir)
    # data_combine2()
    keyword_dict()


if __name__ == "__main__":
    main()
