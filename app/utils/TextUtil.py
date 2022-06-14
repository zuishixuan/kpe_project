# encoding=utf-8
import jieba
import jieba.analyse
from config.constant import work_path
from collections import Counter
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

# 导入自定义词典 自定义词典的路径
jieba.load_userdict(work_path + "/config/data/jieba.dic")
stopwords = []
with open(work_path + '/config/data/stopwords.txt', 'r', encoding="UTF-8") as f:
    line = f.readline()
    while line:
        if line.strip():
            stopwords.append(line.strip())
        line = f.readline()


class TextUtil:
    # 根据路径将数据读取为str
    @staticmethod
    def add_tokens(arr, fields):
        if arr is None or len(arr) == 0:
            return None
        for item in arr:
            text = ''
            for field in fields:
                if item.get(field):
                    text += item[field]
            item['tokens'] = " ".join(jieba.cut(text, cut_all=False))
        return arr

    @staticmethod
    def cut(text, cut_all=False):
        return " ".join(jieba.cut(text, cut_all=cut_all)).split(" ")

    @staticmethod
    def cut_for_search(text):
        return " ".join(jieba.cut_for_search(text)).split(" ")

    @staticmethod
    def extract_tags(text, top_k=3, algorithm='tf-idf', with_weight=True):
        allow_pos = ('n', 'nr', 'an', 'i', 'ns', 'nt')  # , 'vn'
        if algorithm == 'text-rank':
            tags = jieba.analyse.textrank(text, topK=top_k, withWeight=with_weight,
                                          allowPOS=allow_pos)
        else:
            tags = jieba.analyse.extract_tags(text, topK=top_k, withWeight=with_weight,
                                              allowPOS=allow_pos)
        result = []
        for tag in tags:
            name = tag[0]
            if name not in stopwords:
                result.append((name, tag[1]))
        return result

    @staticmethod
    def remove_stopwords(seg_list):
        result = []
        for seg in seg_list:
            if seg not in stopwords:
                result.append(seg)
        return result

    @staticmethod
    def get_tf_idf(word_list):
        vector = CountVectorizer(stop_words=stopwords)
        # 该类会统计每个词语的tf-idf权值
        transformer = TfidfTransformer()
        # 第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
        tf_idf = transformer.fit_transform(vector.fit_transform(word_list))
        # 获取词袋模型中的所有词语
        words = vector.get_feature_names()
        # 将tf-idf矩阵抽取出来，元素w[i][j]表示j词在i类文本中的tf-idf权重
        weights = tf_idf.toarray()
        return words, weights, tf_idf

    @staticmethod
    def get_terms_weights(text):
        def get_w(i):
            return i['w']

        # 去除停用词
        data = [i for i in jieba.lcut(text) if i.strip() and i not in stopwords]
        data = dict(Counter(data))
        words = [{'n': k, 'w': v} for k, v in data.items()]
        words.sort(key=get_w, reverse=True)
        return words
