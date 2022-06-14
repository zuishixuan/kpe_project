import codecs
import os

import jieba
import torch
from transformers import BertTokenizer
from ..new_word_discovery.utils import *
from ..new_word_discovery.model import TrieNode
from ..new_word_discovery.config import *

from .JointMarkScore2AttenKpe import JointMarkScore2AttenKpe
from ..Segmentation import Segmentation


class JMS2AKper(object):
    def __init__(self, args, seg):
        self.args = args
        self.device = args.device
        self.tokenizer = BertTokenizer.from_pretrained(args.pretrained)
        self.model = JointMarkScore2AttenKpe(args=args).to(args.device)
        self.model.load_state_dict(torch.load(args.model_dir))
        self.model.eval()
        self.seg = seg
        self.stop_words = get_stopwords()
        self.stop_words_file = get_default_stop_words_file()
        # for word in codecs.open(self.stop_words_file, 'r', 'utf-8', 'ignore'):
        #     self.stop_words.add(word.strip())

    @staticmethod
    def add_model_specific_args(parent_parser):
        return JointMarkScore2AttenKpe.add_model_specific_args(parent_parser)

    def kpe(self, text, max_num=None, min_num=2, has_score=False, supplement=True, merge=False):
        encoding = self.tokenizer.encode_plus(
            str(text),
            add_special_tokens=True,
            max_length=512,
            return_token_type_ids=True,
            padding='max_length',
            truncation='longest_first',
            return_attention_mask=True,
            return_tensors='pt',
        )
        word_segments = self.seg.segment(text=text)['words_no_filter']
        word_li = []
        for li in word_segments:
            word_li.extend(li)

        word_segments, word_segments_len = padding(word_li, maxlen=self.args.max_word_count)

        b_input_ids = encoding['input_ids'].to(self.device)
        b_input_mask = encoding['attention_mask'].to(self.device)
        b_token_type_ids = encoding['token_type_ids'].to(self.device)

        mark_score, rank_score = self.model.get_contribute(b_input_ids,
                                                           attention_mask=b_input_mask,
                                                           token_type_ids=b_token_type_ids,
                                                           word_segments=word_segments)

        # mark_res = model.generateMarkAns(word_segments, mark_score)
        rank_res = self.model.generateScoreAns(word_segments, rank_score)
        joint_res = self.model.generateJointAns(word_segments, mark_score, rank_score,
                                                True)  # 'type_resort'   'type_remove')
        print("rank_res")
        print(rank_res)
        print("joint_res")
        print(joint_res)
        if max_num or min_num:
            num = max_num if max_num else min_num
            if num > len(joint_res) and supplement:
                sup = 0
                for w in rank_res:
                    if len(w[0]) == 1: continue
                    flag = True
                    for k in joint_res:
                        if sim(w[0], k[0]):
                            flag = False
                            break
                    if flag:
                        sup += 1
                        joint_res.append(w)
                        if num == len(joint_res):
                            break
        if merge:
            flag = True
            remo = set()
            ad = set()
            while flag:
                li = [i[0] for i in joint_res]
                flag = False
                #remo =set()
                ad = set()

                for i in joint_res:
                    for j in joint_res:
                        if i[0] in j[0] and j[0] not in i[0]:
                            if i not in remo:
                                remo.add(i)
                                flag = True
                        if i[0]+j[0] in text:
                            if i[0]+j[0] not in li:
                                ad.add((i[0]+j[0],max(i[1],j[1])))
                                remo.add(i)
                                remo.add(j)
                                flag = True
                        else:
                            if j[0] + i[0] in text:
                                if j[0] + i[0] not in li:
                                    ad.add((j[0] + i[0], max(j[1], i[1])))
                                    remo.add(i)
                                    remo.add(j)
                                    flag = True
                for a in ad:
                    if a not in joint_res:
                        joint_res.append(a)

            for r in remo:
                if r in joint_res:
                    joint_res.remove(r)
                # if len(remo) ==0 and len(ad) == 0:
                #     flag = False
            joint_res.sort(key=lambda x: (x[1]), reverse=True)



        # if max_num or min_num:
        #     num = max_num if max_num else min_num
        #     if num > len(joint_res) and supplement:
        #         sup = 0
        #         for w in rank_res:
        #             if len(w[0]) == 1: continue
        #             flag = True
        #             for k in joint_res:
        #                 if sim(w[0], k[0]):
        #                     flag = False
        #                     break
        #             if flag:
        #                 sup += 1
        #                 joint_res.append(w)
        #                 if num == len(joint_res):
        #                     break
        if max_num:
            joint_res = joint_res[0:max_num]
        if not has_score:
            joint_res = [r[0] for r in joint_res]


        # print(rank_res)
        return joint_res

    def kpe2(self, text, max_num=None, min_num=2, has_score=False, supplement=True):
        encoding = self.tokenizer.encode_plus(
            str(text),
            add_special_tokens=True,
            max_length=512,
            return_token_type_ids=True,
            padding='max_length',
            truncation='longest_first',
            return_attention_mask=True,
            return_tensors='pt',
        )
        word_segments = self.seg.segment(text=text)['words_no_filter']
        word_li = []
        for li in word_segments:
            word_li.extend(li)

        word_segments, word_segments_len = padding(word_li, maxlen=self.args.max_word_count)

        b_input_ids = encoding['input_ids'].to(self.device)
        b_input_mask = encoding['attention_mask'].to(self.device)
        b_token_type_ids = encoding['token_type_ids'].to(self.device)

        mark_score, rank_score = self.model.get_contribute(b_input_ids,
                                                           attention_mask=b_input_mask,
                                                           token_type_ids=b_token_type_ids,
                                                           word_segments=word_segments)

        # mark_res = model.generateMarkAns(word_segments, mark_score)
        rank_res = self.model.generateScoreAns(word_segments, rank_score)
        joint_res = self.model.generateJointAns(word_segments, mark_score, rank_score,
                                                True)  # 'type_resort'   'type_remove')

        if max_num or min_num:
            num = max_num if max_num else min_num
            if num > len(joint_res) and supplement:
                print("aaaaaaaaaaaaaa")
                sup = 0
                stopwords = self.stop_words
                if os.path.exists(root_dir):
                    print("bbbbbbbbbbbbbb")
                    root = load_model(root_dir)
                else:
                    print("aaaaaaaaaaaaaa")
                    word_freq = load_dictionary(dict_dir)
                    root = TrieNode('*', word_freq)
                    save_model(root, root_dir)
                print("ccccccccccccccc")
                # 加载新的文章
                # filename = basedir + '/data/demo2.txt'
                data = load_data(text, stopwords)
                # 将新的文章插入到Root中
                load_data_2_root(root, data)

                # 定义取TOP5个
                topN = 20
                result, add_word = root.find_word(topN)
                # 如果想要调试和选择其他的阈值，可以print result来调整
                # print("\n----\n", result)
                print("\n----\n", '增加了 %d 个新词, 词语和得分分别为: \n' % len(add_word))
                print('#############################')
                for word, score in add_word.items():
                    print(word + ' ---->  ', score)
                print('#############################')
                new_word = [word for word, score in add_word.items() if word in text]
                print(new_word)

                extra_dict = {}
                for word in new_word:
                    extra_dict[word] = -99999
                    for w in rank_res:
                        if w[0] in word:
                            extra_dict[word] = max(extra_dict[word], w[1])
                print(extra_dict)
                for w in extra_dict:
                    for r in reversed(rank_res):
                        if r[0] in w:
                            rank_res.remove(r)
                for w in extra_dict:
                    rank_res.append((w, extra_dict[w]))
                #rank_res =[r for r in rank_res if len(r[0])>1]
                rank_res.sort(key=lambda x: (x[1]), reverse=True)
                print(rank_res)

                sup = 0
                for w in rank_res:
                    if len(w[0]) == 1: continue
                    flag = True
                    for k in joint_res:
                        if sim(w[0], k[0]):
                            flag = False
                            break
                    if flag:
                        sup += 1
                        joint_res.append(w)
                        if num == len(joint_res):
                            break


        if max_num:
            joint_res = joint_res[0:max_num]
        if not has_score:
            joint_res = [r[0] for r in joint_res]

        # print(rank_res)
        return joint_res

    def kpe3(self, text, max_num=None, min_num=2, has_score=False, supplement=True):
        encoding = self.tokenizer.encode_plus(
            str(text),
            add_special_tokens=True,
            max_length=512,
            return_token_type_ids=True,
            padding='max_length',
            truncation='longest_first',
            return_attention_mask=True,
            return_tensors='pt',
        )
        word_segments = self.seg.segment(text=text)['words_no_filter']
        word_li = []
        for li in word_segments:
            word_li.extend(li)

        word_segments, word_segments_len = padding(word_li, maxlen=self.args.max_word_count)

        b_input_ids = encoding['input_ids'].to(self.device)
        b_input_mask = encoding['attention_mask'].to(self.device)
        b_token_type_ids = encoding['token_type_ids'].to(self.device)

        mark_score, rank_score = self.model.get_contribute(b_input_ids,
                                                           attention_mask=b_input_mask,
                                                           token_type_ids=b_token_type_ids,
                                                           word_segments=word_segments)

        # mark_res = model.generateMarkAns(word_segments, mark_score)
        rank_res = self.model.generateScoreAns(word_segments, rank_score)
        joint_res = self.model.generateJointAns(word_segments, mark_score, rank_score,
                                                True)  # 'type_resort'   'type_remove')

        if max_num or min_num:
            num = max_num if max_num else min_num
            if num > len(joint_res) and supplement:
                print("aaaaaaaaaaaaaa")
                sup = 0
                stopwords = self.stop_words
                if os.path.exists(root_dir):
                    print("aaaaaaaaaaaaaa")
                    root = load_model(root_dir)
                else:
                    print("bbbbbbbbbbbbbb")
                    word_freq = load_dictionary(dict_dir)
                    root = TrieNode('*', word_freq)
                    save_model(root, root_dir)

                # 加载新的文章
                # filename = basedir + '/data/demo2.txt'
                data = load_data(text, stopwords)
                # 将新的文章插入到Root中
                load_data_2_root(root, data)

                # 定义取TOP5个
                topN = 20
                result, add_word = root.find_word(topN)
                # 如果想要调试和选择其他的阈值，可以print result来调整
                # print("\n----\n", result)
                print("\n----\n", '增加了 %d 个新词, 词语和得分分别为: \n' % len(add_word))
                print('#############################')
                for word, score in add_word.items():
                    print(word + ' ---->  ', score)
                print('#############################')
                new_word = [word for word, score in add_word.items() if word in text]
                print(new_word)

                extra_dict = {}
                for word in new_word:
                    extra_dict[word] = -99999
                    for w in rank_res:
                        if w[0] in word:
                            extra_dict[word] = max(extra_dict[word], w[1])
                print(extra_dict)
                for w in extra_dict:
                    for r in reversed(rank_res):
                        if r[0] in w:
                            rank_res.remove(r)
                for w in extra_dict:
                    rank_res.append((w, extra_dict[w]))
                #rank_res =[r for r in rank_res if len(r[0])>1]
                rank_res.sort(key=lambda x: (x[1]), reverse=True)
                print(rank_res)

                sup = 0
                for w in rank_res:
                    if len(w[0]) == 1: continue
                    flag = True
                    for k in joint_res:
                        if sim(w[0], k[0]):
                            flag = False
                            break
                    if flag:
                        sup += 1
                        joint_res.append(w)
                        if num == len(joint_res):
                            break


        if max_num:
            joint_res = joint_res[0:max_num]
        if not has_score:
            joint_res = [r[0] for r in joint_res]

        # print(rank_res)
        return joint_res

    def filter_word(self, word):
        # word_list = [word for word in word_list if len(word) > 1]
        # word_list = [word.strip() for word in word_list if word.strip() not in self.stop_words]
        return word.strip() in self.stop_words


def sim(w, k, threshold=None):
    return w in k or k in w


def get_default_stop_words_file():
    d = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(d, '../../data/stopwords.txt')


def get_stopwords():
    with open(stopword_dir, 'r', encoding="utf_8") as f:
        stopword = [line.strip() for line in f]
    return set(stopword)


def padding(li, maxlen=300, pad_token=''):
    le = len(li)
    if len(li) < maxlen:
        li.extend([pad_token for i in range(maxlen - le)])
    else:
        if len(li) > maxlen:
            li = li[0:maxlen]
            le = maxlen
    return li, le


def exit(item, list):
    if item in list:
        return 1
    else:
        return 0


def load_data(text, stopwords):
    """

    :param filename:
    :param stopwords:
    :return: 二维数组,[[句子1分词list], [句子2分词list],...,[句子n分词list]]
    """
    data = []
    for line in [text]:
        seg = Segmentation(user_dict=None)
        res = seg.segment(line)['words_all_filters']
        word_list = []
        for li in res:
            word_list.extend(li)
        word_list = [x.strip() for x in word_list if x.strip() not in stopwords]
        data.append(word_list)
    return data


def load_data_2_root(root, data):
    print('------> 插入节点')
    for word_list in data:
        # tmp 表示每一行自由组合后的结果（n gram）
        # tmp: [['它'], ['是'], ['小'], ['狗'], ['它', '是'], ['是', '小'], ['小', '狗'], ['它', '是', '小'], ['是', '小', '狗']]
        ngrams = generate_ngram(word_list, 4)
        for d in ngrams:
            root.add(d)
    print('------> 插入成功')
