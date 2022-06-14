# -*- encoding:utf-8 -*-
"""
@author:   letian
@homepage: http://www.letiantian.me
@github:   https://github.com/someus/
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import copy
from argparse import ArgumentParser

import torch
from transformers import BertTokenizer

from . import util
from .Segmentation import Segmentation
from .util import *
from .SimCSE.SimCSE import SimCSE

textrank_type = ['basic', 'advance']
class TextRank4Keyword(object):

    def __init__(self, stop_words_file=None,
                 allow_speech_tags=util.allow_speech_tags,
                 delimiters=util.sentence_delimiters,
                 simcse_model_path="./model/out/epoch_1-batch_1250",
                 MODEL_DIR='./model/chinese-bert-wwm-ext',
                 algo_type = 'basic',
                 window = 2,
                 pagerank_config={'alpha': 0.85},
                 user_dict = None):
        """
        Keyword arguments:
        stop_words_file  --  str，指定停止词文件路径（一行一个停止词），若为其他类型，则使用默认停止词文件
        delimiters       --  默认值是`?!;？！。；…\n`，用来将文本拆分为句子。
        
        Object Var:
        self.words_no_filter      --  对sentences中每个句子分词而得到的两级列表。
        self.words_no_stop_words  --  去掉words_no_filter中的停止词而得到的两级列表。
        self.words_all_filters    --  保留words_no_stop_words中指定词性的单词而得到的两级列表。
        """

        self.seg = Segmentation(stop_words_file=stop_words_file,
                                allow_speech_tags=allow_speech_tags,
                                delimiters=delimiters,
                                user_dict= user_dict)
        self.algo_type = algo_type
        self.window = window
        self.pagerank_config=pagerank_config

        self.text = ''
        self.sentences = None
        self.words_no_filter = None  # 2维列表
        self.words_no_stop_words = None
        self.words_all_filters = None

        self.word_index = {}
        self.index_word = {}
        self.keywords = []
        self.graph = None

        self.sim_matrix = None
        self.position_matrix = None

        self.words_num = 0
        self.word_embedding = None
        self.document_embedding = None

        if self.algo_type == 'advance':
            self.tokenizer = BertTokenizer.from_pretrained(MODEL_DIR, mirror="tuna")
            self.encoder = SimCSE(pretrained=MODEL_DIR).to('cuda')
            self.encoder.load_state_dict(torch.load(simcse_model_path))
            self.encoder.eval()

    def analyze(self, text,
                window=2,
                lower=False,
                vertex_source='all_filters',
                edge_source='no_stop_words',
                pagerank_config={'alpha': 0.85, }):
        """分析文本

        Keyword arguments:
        text       --  文本内容，字符串。
        window     --  窗口大小，int，用来构造单词之间的边。默认值为2。
        lower      --  是否将文本转换为小写。默认为False。
        vertex_source   --  选择使用words_no_filter, words_no_stop_words, words_all_filters中的哪一个来构造pagerank对应的图中的节点。
                            默认值为`'all_filters'`，可选值为`'no_filter', 'no_stop_words', 'all_filters'`。关键词也来自`vertex_source`。
        edge_source     --  选择使用words_no_filter, words_no_stop_words, words_all_filters中的哪一个来构造pagerank对应的图中的节点之间的边。
                            默认值为`'no_stop_words'`，可选值为`'no_filter', 'no_stop_words', 'all_filters'`。边的构造要结合`window`参数。
        """

        # self.text = util.as_text(text)
        self.text = text
        self.sentences = None
        self.words_no_filter = None  # 2维列表
        self.words_no_stop_words = None
        self.words_all_filters = None

        self.word_index = {}
        self.index_word = {}
        self.keywords = []
        self.graph = None

        self.sim_matrix = None
        self.position_matrix = None

        self.words_num = 0

        self.word_embedding = None
        self.document_embedding = None

        result = self.seg.segment(text=text, lower=lower)
        self.sentences = result.sentences
        self.words_no_filter = result.words_no_filter
        self.words_no_stop_words = result.words_no_stop_words
        self.words_all_filters = result.words_all_filters

        util.debug(20 * '*')
        util.debug('self.sentences in TextRank4Keyword:\n', ' || '.join(self.sentences))
        util.debug('self.words_no_filter in TextRank4Keyword:\n', self.words_no_filter)
        util.debug('self.words_no_stop_words in TextRank4Keyword:\n', self.words_no_stop_words)
        util.debug('self.words_all_filters in TextRank4Keyword:\n', self.words_all_filters)

        options = ['no_filter', 'no_stop_words', 'all_filters']

        if vertex_source in options:
            _vertex_source = result['words_' + vertex_source]
        else:
            _vertex_source = result['words_all_filters']

        if edge_source in options:
            _edge_source = result['words_' + edge_source]
        else:
            _edge_source = result['words_no_stop_words']

        self.graph_generate(_vertex_source, _edge_source, window)

        self.keywords = self.sort_words(pagerank_config=pagerank_config)

    def graph_generate(self, vertex_source, edge_source, window=3):
        """将单词按关键程度从大到小排序

        Keyword arguments:
        vertex_source   --  二维列表，子列表代表句子，子列表的元素是单词，这些单词用来构造pagerank中的节点
        edge_source     --  二维列表，子列表代表句子，子列表的元素是单词，根据单词位置关系构造pagerank中的边
        window          --  一个句子中相邻的window个单词，两两之间认为有边
        pagerank_config --  pagerank的设置
        """
        sorted_words = []
        word_index = {}
        index_word = {}
        _vertex_source = vertex_source
        _edge_source = edge_source
        words_number = 0
        for word_list in _vertex_source:
            for word in word_list:
                if not word in word_index:
                    word_index[word] = words_number
                    index_word[words_number] = word
                    words_number += 1

        graph = np.zeros((words_number, words_number))

        for word_list in _edge_source:
            for w1, w2 in combine(word_list, self.window):
                if w1 in word_index and w2 in word_index:
                    index1 = word_index[w1]
                    index2 = word_index[w2]
                    graph[index1][index2] = 1.0
                    graph[index2][index1] = 1.0

        self.word_index = word_index
        self.graph = graph
        self.words_num = words_number
        self.index_word = index_word
        # debug('graph:\n', graph)

    def sort_words(self, pagerank_config={'alpha': 0.85, }):
        """将单词按关键程度从大到小排序

        Keyword arguments:
        vertex_source   --  二维列表，子列表代表句子，子列表的元素是单词，这些单词用来构造pagerank中的节点
        edge_source     --  二维列表，子列表代表句子，子列表的元素是单词，根据单词位置关系构造pagerank中的边
        window          --  一个句子中相邻的window个单词，两两之间认为有边
        pagerank_config --  pagerank的设置
        """
        sorted_words = []
        if self.algo_type=='advance':
            self.graph = self.get_weight_matrix()
        g= self.graph
        #g = self.graph
        nx_graph = nx.from_numpy_matrix(g)
        scores = nx.pagerank(nx_graph, **pagerank_config)  # this is a dict

        edge_labels = dict([((u, v,), d['weight']) for u, v, d in nx_graph.edges(data=True)])
        # pos = nx.spring_layout(nx_graph)
        # nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=edge_labels)
        # nx.draw_networkx(nx_graph)
        #
        # plt.show()

        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        for index, score in sorted_scores:
            item = AttrDict(word=self.index_word[index], weight=score)
            sorted_words.append(item)

        return sorted_words

    def get_keywords(self, num=6, word_min_len=1):
        """获取最重要的num个长度大于等于word_min_len的关键词。

        Return:
        关键词列表。
        """
        result = []
        count = 0
        for item in self.keywords:
            if count >= num:
                break
            if len(item.word) >= word_min_len:
                result.append(item)
                count += 1
        return result

    def get_keyphrases(self, keywords_num=12, min_occur_num=2):
        """获取关键短语。
        获取 keywords_num 个关键词构造的可能出现的短语，要求这个短语在原文本中至少出现的次数为min_occur_num。

        Return:
        关键短语的列表。
        """
        keywords_set = set([item.word for item in self.get_keywords(num=keywords_num, word_min_len=1)])
        keyphrases = set()
        for sentence in self.words_no_filter:
            one = []
            for word in sentence:
                if word in keywords_set:
                    one.append(word)
                else:
                    if len(one) > 1:
                        keyphrases.add(''.join(one))
                    # if len(one) == 0:
                    #     continue
                    # else:
                    one = []
            # 兜底
            if len(one) > 1:
                keyphrases.add(''.join(one))

        return [phrase for phrase in keyphrases
                if self.text.count(phrase) >= min_occur_num]

    def get_word_embedding(self):
        word_list = [word for word in self.word_index]
        encoding = self.tokenizer.encode_plus(self.text,
                                              add_special_tokens=True,
                                              max_length=512,
                                              return_token_type_ids=True,
                                              padding='max_length',
                                              truncation='longest_first',
                                              return_attention_mask=True,
                                              return_tensors='pt',
                                              )
        self.word_embedding, self.document_embedding = self.encoder.encode(encoding['input_ids'].to('cuda'),
                                                                           encoding['attention_mask'].to('cuda'),
                                                                           encoding['token_type_ids'].to('cuda'),
                                                                           word_list)
        # print(self.word_embedding)
        return self.word_embedding

    def get_sim_matrix(self):
        if self.word_embedding is None:
            self.get_word_embedding()
        self.sim_matrix = copy.deepcopy(self.graph)
        for i in range(self.words_num):
            for j in range(i + 1, self.words_num):
                sim = torch.cosine_similarity(self.word_embedding[self.index_word[i]],
                                              self.word_embedding[self.index_word[j]], dim=0)
                self.sim_matrix[i][j] = sim
                self.sim_matrix[j][i] = sim
        self.sim_matrix = np.multiply(self.sim_matrix, self.graph)
        return self.sim_matrix

    def get_position_matrix(self, title_wei=2, first_wei=1):
        # print(self.word_index)
        self.position_matrix = np.ones((self.words_num, self.words_num))
        for word in self.words_all_filters[1]:
            self.position_matrix[:, self.word_index[word]] = first_wei
        for word in self.words_all_filters[0]:
            self.position_matrix[:, self.word_index[word]] = title_wei
        self.position_matrix = np.multiply(self.position_matrix, self.graph)
        return self.position_matrix

    def get_weight_matrix(self, option=['position' ,'similarity']):
        #weight_matrix =  self.norm_matrix(np.ones((self.words_num, self.words_num)) * 0.5)
        weight_matrix = np.zeros((self.words_num, self.words_num))
        if 'position' in option:
            if self.position_matrix is None:
                self.get_position_matrix()
            weight_matrix += self.norm_matrix(self.position_matrix) * weight_matrix_param['position']
        if 'similarity' in option:
            if self.sim_matrix is None:
                self.get_sim_matrix()
            weight_matrix += self.norm_matrix(self.sim_matrix) * weight_matrix_param['similarity']
        return self.norm_matrix(np.multiply(weight_matrix, self.graph))

    def norm_matrix(self, x):
        return np.nan_to_num(x / x.sum(axis=1).reshape(-1, 1))

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--window", type=int, default=2)
        return parser

    def kpe(self, text, max_num=None, has_score=False):
        self.analyze(text=text, lower=True, window=self.window, pagerank_config=self.pagerank_config)
        exc = self.get_keywords(max_num, word_min_len=2)
        if not has_score:
            exc = [w['word'] for w in exc]
        #print(exc)
        return exc

    def clear(self):
        self.text = ''
        self.sentences = None
        self.words_no_filter = None  # 2维列表
        self.words_no_stop_words = None
        self.words_all_filters = None

        self.word_index = {}
        self.index_word = {}
        self.keywords = []
        self.graph = None

        self.sim_matrix = None
        self.position_matrix = None

        self.words_num = 0
        self.word_embedding = None
        self.document_embedding = None


if __name__ == '__main__':
    pass
