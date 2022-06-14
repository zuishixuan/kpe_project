import pandas as pd

#df = pd.read_json('./data/test.json', lines=True)

df = pd.read_excel('./data/data.xlsx')
docs = df['abst'].values

# print(df['keyword'])

# 1 加载模块
import math
import jieba
import jieba.posseg as psg
from gensim import corpora, models
from jieba import analyse
import functools


# 2 定义好停用词表的加载方法
def get_stopword_list():
    # 停用词表存储路径，每一行为一个词，按行读取进行加载
    # 进行编码转换确保匹配准确率
    stop_word_path = '../data/stopWord.txt'
    stopword_list = [sw.replace('/n', '') for sw in open(stop_word_path, encoding='utf-8').readlines()]
    return stopword_list


# 3 定义一个分词方法
def seg_to_list(sentence, pos=False):
    ''' 分词方法，调用结巴接口。pos为判断是否采用词性标注 '''
    if not pos:
        # 不进行词性标注的分词方法
        seg_list = jieba.cut(sentence)
    else:
        # 进行词性标注的分词方法
        seg_list = psg.cut(sentence)
    return seg_list


# 4 定义干扰词过滤方法
def word_filter(seg_list, pos=False):
    '''
        1. 根据分词结果对干扰词进行过滤；
        2. 根据pos判断是否过滤除名词外的其他词性；
        3. 再判断是否在停用词表中，长度是否大于等于2等；
    '''
    stopword_list = get_stopword_list()  # 获取停用词表
    filter_list = []  # 保存过滤后的结果
    #  下面代码： 根据pos参数选择是否词性过滤
    ## 下面代码： 如果不进行词性过滤，则将词性都标记为n，表示全部保留
    for seg in seg_list:
        if not pos:
            word = seg
            flag = 'n'
        else:
            word = seg.word  # 单词
            flag = seg.flag  # 词性
        if not flag.startswith('n'):
            continue
        # 过滤停用词表中的词，以及长度为<2的词
        if not word in stopword_list and len(word) > 1:
            filter_list.append(word)

    return filter_list


# 5 加载数据集，并对数据集中的数据分词和过滤干扰词
def load_data(pos=False, corpus_path='./corpus.txt'):
    '''
        目的：
            调用上面方法对数据集进行处理，处理后的每条数据仅保留非干扰词
        参数：
            1. 数据加载
            2. pos: 是否词性标注的参数
            3. corpus_path: 数据集路径
    '''
    doc_list = []  # 结果
    # for line in open(corpus_path, 'r'):
    for line in docs:
        content = line.strip()  # 每行的数据
        seg_list = seg_to_list(content, pos)  # 分词
        filter_list = word_filter(seg_list, pos)  # 过滤停用词
        doc_list.append(filter_list)  # 将处理后的结果保存到doc_list
    return doc_list


# 6 IDF 训练
# TF-IDF的训练主要是根据数据集生成对应的IDF值字典，后续计算每个词的TF-IDF时，直接从字典中读取。

def train_idf(doc_list):
    idf_dic = {}  # idf对应的字典
    tt_count = len(doc_list)  # 总文档数
    # 每个词出现的文档数
    for doc in doc_list:
        for word in set(doc):
            idf_dic[word] = idf_dic.get(word, 0.0) + 1.0
    # 按公式转换为idf值，分母加1进行平滑处理
    for k, v in idf_dic.items():
        idf_dic[k] = math.log(tt_count / (1.0 + v))
    # 对于没有在字典中的词，默认其尽在一个文档出现，得到默认idf值
    default_idf = math.log(tt_count / (1.0))
    return idf_dic, default_idf


# 7 LSI 训练
# LSI的训练时根据现有的数据集生成文档-主题分布矩阵和主题-词分布矩阵，Gensim中有实现好的方法，可以直接调用。

def train_lsi(self):
    lsi = models.LsiModel(self.corpus_tfidf, id2word=self.dictionary, num_topics=self.num_topics)
    return lsi


# 8 LDA训练
# LDA的训练时根据现有的数据集生成文档-主题分布矩阵和主题-词分布矩阵，Gensim中有实现好的方法，可以直接调用。

def train_lda(self):
    lda = models.LdaModel(self.corpus_tfidf, id2word=self.dictionary, num_topics=self.num_topics)
    return lda


# 9 cmp函数
# 为了输出top关键词时，先按照关键词的计算分值排序，在得分相同时，根据关键词进行排序

def cmp(e1, e2):
    ''' 排序函数，用于topK关键词的按值排序 '''
    import numpy as np
    res = np.sign(e1[1] - e2[1])
    if res != 0:
        return res
    else:
        a = e1[0] + e2[0]
        b = e2[0] + e1[0]
        if a > b:
            return 1
        elif a == b:
            return 0
        else:
            return -1


# 10 TF-IDF实现方法
class TfIdf(object):
    # 四个参数分别是：训练好的idf字典，默认idf字典，处理后的待提取文本， 关键词数量
    def __init__(self, idf_dic, default_idf, word_list, keyword_num):
        self.idf_dic, self.default_idf = idf_dic, default_idf
        self.word_list = word_list
        self.tf_dic = self.get_tf_dic()  # 统计tf值
        self.keyword_num = keyword_num

    def get_tf_dic(self):
        # 统计tf值
        tf_dic = {}
        for word in self.word_list:
            tf_dic[word] = tf_dic.get(word, 0.0) + 1.0
        tt_count = len(self.word_list)
        for k, v in tf_dic.items():
            tf_dic[k] = float(v) / tt_count  # 根据tf求值公式

        return tf_dic

    def get_tfidf(self):
        # 计算tf-idf值
        tfidf_dic = {}
        for word in self.word_list:
            idf = self.idf_dic.get(word, self.default_idf)
            tf = self.tf_dic.get(word, 0)

            tfidf = tf * idf
            tfidf_dic[word] = tfidf

        tfidf_dic.items()
        # 根据tf-idf排序，去排名前keyword_num的词作为关键词
        for k, v in sorted(tfidf_dic.items(), key=functools.cmp_to_key(cmp), reverse=True)[:self.keyword_num]:
            print(k + '/', end='')
        print()


# 11 完整的主题模型实现方法
class TopicModel(object):
    # 三个传入参数：处理后的数据集，关键词数量，具体模型（LSI，LDA），主题数量
    def __init__(self, doc_list, keyword_num, model='LSI', num_topics=4):
        # 使用gensim接口，将文本转为向量化表示
        # 先构建词空间
        self.dictionary = corpora.Dictionary(doc_list)
        # 使用BOW模型向量化
        corpus = [self.dictionary.doc2bow(doc) for doc in doc_list]
        # 对每个词，根据tf-idf进行加权，得到加权后的向量表示
        self.tfidf_model = models.TfidfModel(corpus)
        self.corpus_tfidf = self.tfidf_model[corpus]

        self.keyword_num = keyword_num
        self.num_topics = num_topics

        # 选择加载的模型
        if model == "LSI":
            self.model = self.train_lsi()
        else:
            self.model = self.train_lda()

        # 得到数据集的主题-词分布
        word_dic = self.word_dictionary(doc_list)
        self.wordtopic_dic = self.get_wordtopic(word_dic)

    # LSI的训练时根据现有的数据集生成文档-主题分布矩阵和主题-词分布矩阵，Gensim中有实现好的方法，可以直接调用。
    def train_lsi(self):
        lsi = models.LsiModel(self.corpus_tfidf, id2word=self.dictionary, num_topics=self.num_topics)
        return lsi

    # LDA的训练时根据现有的数据集生成文档-主题分布矩阵和主题-词分布矩阵，Gensim中有实现好的方法，可以直接调用。
    def train_lda(self):
        lda = models.LdaModel(self.corpus_tfidf, id2word=self.dictionary, num_topics=self.num_topics)
        return lda

    def get_wordtopic(self, word_dic):
        wordtopic_dic = {}
        for word in word_dic:
            single_list = [word]
            wordcorpus = self.tfidf_model[self.dictionary.doc2bow(single_list)]
            wordtopic = self.model[wordcorpus]
            wordtopic_dic[word] = wordtopic
        return wordtopic_dic

    def get_simword(self, word_list):
        # 计算词的分布和文档的分布的相似度，去相似度最高的keyword_num个词作为关键词
        sentcorpus = self.tfidf_model[self.dictionary.doc2bow(word_list)]
        senttopic = self.model[sentcorpus]

        # 余弦相似度计算
        def calsim(l1, l2):
            a, b, c = 0.0, 0.0, 0.0
            for t1, t2 in zip(l1, l2):
                x1 = t1[1]
                x2 = t2[1]
                a += x1 * x1
                b += x1 * x1
                c += x2 * x2
            sim = a / math.sqrt(b * c) if not (b * c) == 0.0 else 0.0
            return sim

        # 计算输入文本和每个词的主题分布相似度
        sim_dic = {}
        for k, v in self.wordtopic_dic.items():
            if k not in word_list:
                continue
            sim = calsim(v, senttopic)
            sim_dic[k] = sim

        for k, v in sorted(sim_dic.items(), key=functools.cmp_to_key(cmp), reverse=True)[:self.keyword_num]:
            print(k + '/', end='')

        print()

    def word_dictionary(self, doc_list):
        # 词空间构建方法和向量化方法，在没有gensim接口时的一般处理方法
        dictionary = []
        for doc in doc_list:
            dictionary.extend(doc)
        dictionary = list(set(dictionary))
        return dictionary

    def doc2bowvec(self, word_list):
        vec_list = [1 if word in word_list else 0 for word in self.dictionary]
        return vec_list


# 12 对上面的各个方法进行封装，统一算法调用接口
def tfidf_extract(word_list, pos=False, keyword_num=10):
    doc_list = load_data(pos)
    idf_dic, default_idf = train_idf(doc_list)
    tfidf_model = TfIdf(idf_dic, default_idf, word_list, keyword_num)
    tfidf_model.get_tfidf()


def textrank_extract(text, pos=False, keyword_num=10):
    textrank = analyse.textrank
    keywords = textrank(text, keyword_num)
    # 输出抽取出的关键词
    for keyword in keywords:
        print(keyword + "/", end='')
    print()


def topic_extract(word_list, model, pos=False, keyword_num=10):
    doc_list = load_data(pos)
    topic_model = TopicModel(doc_list, keyword_num, model=model)
    topic_model.get_simword(word_list)


text = df['abst'].values[3]
print(text)
keyword_num = 3

pos = False
seg_list = seg_to_list(text, pos)
filter_list = word_filter(seg_list, pos)

print("TF-IDF模型结果：")
tfidf_extract(filter_list, keyword_num=keyword_num)
print("TextRank模型结果：")
textrank_extract(text, keyword_num=keyword_num)
print("LSI模型结果：")
topic_extract(filter_list, 'LSI', pos, keyword_num=keyword_num)
print("LDA模型结果：")
topic_extract(filter_list, 'LDA', pos, keyword_num=keyword_num)
