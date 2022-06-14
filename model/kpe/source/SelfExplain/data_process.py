import random
import sys

import pandas as pd
from transformers import Trainer, TrainingArguments, BertTokenizer, BertModel, BertPreTrainedModel, BertConfig
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput

# def read_data(data_dir=corpus_dir):
#     df = pd.read_excel(data_dir)
#
#     # df['len']=df['abst'].str.len()
#     df['len'] = df['abst'].map(len)
#     df['keywords'] = df['keyword'].str.split('\\;|；')
#     df['num'] = df['keywords'].map(len)
#     df['label'] = 1
#     return df
from ..Segmentation import Segmentation


def read_data_for_bert(data_dir='./data/data.xlsx', per=0.3):
    df = pd.read_csv(data_dir, index_col=0)

    return df


class InputDataSet(Dataset):

    def __init__(self, data, tokenizer, max_len, num_classes=13, max_word_count=300, max_word_len=10):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_classes = num_classes
        self.seg = Segmentation()
        self.max_word_count = max_word_count
        self.max_word_len = max_word_len

    def __len__(self, ):
        return len(self.data)

    def __getitem__(self, item):  # item是索引 用来取数据
        text = str(self.data['text'][item])
        subject_class = self.data['subject_class'][item]
        labels = torch.zeros(13)
        for i in subject_class.split(' '):
            labels[int(i)] = 1
        encoding = self.tokenizer.encode_plus(
            str(self.data['text'][item]),
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            truncation='longest_first',
            return_attention_mask=True,
            return_tensors='pt',
        )
        word_segments = self.seg.simple_segment(text=text)['words_all_filters']

        ## 手动构建
        # tokens = self.tokenizer.tokenize(text)
        # tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        # tokens_ids = [101] + tokens_ids + [102]
        # input_ids = fill_paddings(tokens_ids,self.max_len)
        #
        # attention_mask = [1 for _ in range(len(tokens_ids))]
        # attention_mask = fill_paddings(attention_mask,self.max_len)
        #
        # token_type_ids = [0 for _ in range(len(tokens_ids))]
        # token_type_ids = fill_paddings(token_type_ids,self.max_len)
        # print(text)
        # print(word_segments)
        word_segments_len = 0

        word_li = []
        for li in word_segments:
            word_li.extend(li)
        li_uni = list(set(word_li))
        li_uni.sort(key=word_li.index)


        #
        # for i,word in enumerate(li_uni):
        #     words = self.tokenizer.encode(word, add_special_tokens=False)
        #     length = len(words)
        #     word_segments[i][0:length] = words[0:length]



        word_segments, word_segments_len = self.padding(li_uni)
        return {
            'texts': text,
            'input_ids': encoding['input_ids'].flatten(),  # 通过tokenizer做的编码 101 。。。。 102
            'attention_mask': encoding['attention_mask'].flatten(),  # 注意力编码111111000000
            "token_type_ids": encoding['token_type_ids'].flatten(),  # 分句编码11111000000
            'labels': labels,
            'word_segments': word_segments,
            'word_segments_len': word_segments_len
        }

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
