import random
import re
import sys

import pandas as pd
from transformers import Trainer, TrainingArguments, BertTokenizer, BertModel, BertPreTrainedModel, BertConfig
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput
from ..constant import tag_map
import jieba

# def read_data(data_dir=corpus_dir):
#     df = pd.read_excel(data_dir)
#
#     # df['len']=df['abst'].str.len()
#     df['len'] = df['abst'].map(len)
#     df['keywords'] = df['keyword'].str.split('\\;|；')
#     df['num'] = df['keywords'].map(len)
#     df['label'] = 1
#     return df
# sys.path.append("..")
from ..Segmentation import Segmentation


def read_data_for_bert(data_dir='./data/data.xlsx', per=0.3):
    df = pd.read_csv(data_dir)

    return df


class InputDataSet(Dataset):

    def __init__(self, data, tokenizer, max_len=500, user_dict=None, max_word_count=300):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.seg = Segmentation(user_dict=user_dict)
        self.max_word_count = max_word_count

    def __len__(self, ):
        return len(self.data)

    def __getitem__(self, item):  # item是索引 用来取数据
        text = str(self.data['seg'][item])
        word_li = text.split(';')
        mark = str(self.data['mark'][item])
        mark_li = mark.split(';')
        mark_label = [int(i) for i in mark_li]


        encoding = self.tokenizer.encode_plus(
            str(text),
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            truncation='longest_first',
            return_attention_mask=True,
            return_tensors='pt',
        )
        # print(str(self.data['text'][item]))
        # print(str(self.data['keyword'][item]))
        # print(word_li)
        # print(mark_li)

        word_list_len = len(word_li)
        word_segments, word_segments_len = self.padding(word_li, maxlen=self.max_word_count)
        mark_label.extend([-1] * (self.max_word_count - word_list_len))

        mark_label = torch.tensor(mark_label,dtype=torch.long)

        return {
            'texts': text,
            'input_ids': encoding['input_ids'].flatten(),  # 通过tokenizer做的编码 101 。。。。 102
            'attention_mask': encoding['attention_mask'].flatten(),  # 注意力编码111111000000
            "token_type_ids": encoding['token_type_ids'].flatten(),  # 分句编码11111000000
            # 'labels': labels,
            'word_segments': word_segments,
            'word_segments_len': word_segments_len,
            'mark_label': mark_label
        }

    # def __getitem__(self, item):  # item是索引 用来取数据
    #     text = str(self.data['text'][item])
    #     text = text[:min(len(text), self.max_len)]
    #     pattern = ';|；| '
    #     keywords = str(self.data['keyword'][item])
    #     res = re.split(pattern, keywords)
    #     keyword_list = [r.strip() for r in res]
    #
    #     # labels = torch.zeros(13)
    #
    #     encoding = self.tokenizer.encode_plus(
    #         str(text),
    #         add_special_tokens=True,
    #         max_length=self.max_len,
    #         return_token_type_ids=True,
    #         padding='max_length',
    #         truncation='longest_first',
    #         return_attention_mask=True,
    #         return_tensors='pt',
    #     )
    #     word_segments = self.seg.segment(text=text)['words_no_filter']
    #     keyword_list_seg = [self.seg.segment(text=w)['words_no_filter'][0] for w in keyword_list]
    #     # print(text)
    #     # print(word_segments)
    #     # print(keyword_list)
    #     # print(keyword_list_seg)
    #
    #     word_li = []
    #     for li in word_segments:
    #         word_li.extend(li)
    #
    #
    #     word_segments_len = 0
    #
    #     # word_dict={'':0}
    #     # word_set = set(word_li)
    #     # for w in word_set:
    #     #     if w not in word_dict:
    #     #         word_dict[w]= len(word_dict)
    #     # print(word_dict)
    #
    #     word_list_len = len(word_li)
    #     word_segments, word_segments_len = self.padding(word_li, maxlen=self.max_word_count)
    #
    #     label = torch.zeros(self.max_word_count, dtype=torch.long)
    #     for ws in keyword_list_seg:
    #         leng = len(ws)
    #         for i in range(word_list_len - leng + 1):
    #             if ws == word_li[i:i + leng]:
    #                 if leng == 1:
    #                     label[i] = tag_map['S-KEY']
    #                 if leng == 2:
    #                     label[i] = tag_map['B-KEY']
    #                     label[i + 1] = tag_map['E-KEY']
    #                 if leng > 2:
    #                     label[i] = tag_map['B-KEY']
    #                     label[i + leng-1] = tag_map['E-KEY']
    #                     label[i + 1:i + leng-1] = tag_map['I-KEY']
    #     # label = torch.zeros(self.max_word_count+2)
    #     # label[0] = tag_map['START']
    #     # label[word_list_len+1] = tag_map['END']
    #     # for ws in keyword_list_seg:
    #     #     leng = len(ws)
    #     #     for i in range(word_list_len - leng + 1):
    #     #         if ws == word_li[i:i + leng]:
    #     #             if leng == 1:
    #     #                 label[i + 1] = tag_map['S-KEY']
    #     #             if leng == 2:
    #     #                 label[i + 1] = tag_map['B-KEY']
    #     #                 label[i + 2] = tag_map['E-KEY']
    #     #             if leng > 2:
    #     #                 label[i + 1] = tag_map['B-KEY']
    #     #                 label[i + leng] = tag_map['E-KEY']
    #     #                 label[i + 2:i + leng] = tag_map['I-KEY']
    #     #print(label)
    #     mark_label = label
    #     return {
    #         'texts': text,
    #         'input_ids': encoding['input_ids'].flatten(),  # 通过tokenizer做的编码 101 。。。。 102
    #         'attention_mask': encoding['attention_mask'].flatten(),  # 注意力编码111111000000
    #         "token_type_ids": encoding['token_type_ids'].flatten(),  # 分句编码11111000000
    #         # 'labels': labels,
    #         'word_segments': word_segments,
    #         'word_segments_len': word_segments_len,
    #         'mark_label': mark_label
    #     }

    def padding(self, li, maxlen=300, pad_token=''):
        le = len(li)
        if len(li) < maxlen:
            li.extend([pad_token for i in range(maxlen - le)])
        else:
            if len(li) > maxlen:
                li = li[0:maxlen]
                le = maxlen
        return li, le


if __name__ == '__main__':
    data_dir = '../data/data_1m_2.csv'

    df = read_data_for_bert(data_dir)

    print(df)

    print(df.info())
    perc = [.70, .80, .90, .95]
    print(df.describe(percentiles=perc))

    model_dir = '../model/chinese-bert-wwm-ext'
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    train_dataset = InputDataSet(df, tokenizer=tokenizer, max_len=512)
    train_dataloader = DataLoader(train_dataset, batch_size=32)
    batch = next(iter(train_dataloader))
    print(batch)

    print(tokenizer.convert_ids_to_tokens(batch['input_ids'][1]))
