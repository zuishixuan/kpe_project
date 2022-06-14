import random
from constant import *
import pandas as pd
from transformers import Trainer, TrainingArguments, BertTokenizer, BertModel, BertPreTrainedModel, BertConfig
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput


def read_data(data_dir=corpus_dir):
    df = pd.read_excel(data_dir)

    # df['len']=df['abst'].str.len()
    df['len'] = df['abst'].map(len)
    df['keywords'] = df['keyword'].str.split('\\;|；')
    df['num'] = df['keywords'].map(len)
    df['label'] = 1
    return df


def read_data_for_bert(data_dir='./data/data.xlsx', per=0.3):
    df = pd.read_excel(data_dir)
    # df['len']=df['abst'].str.len()
    df['len'] = df['abst'].map(len)
    df['keywords'] = df['keyword'].str.split('\\;|；')
    df['num'] = df['keywords'].map(len)
    df['label'] = 1

    count = len(df['abst'])
    indexs = random.sample(range(0, count-1), int(count * per))

    for i in indexs:
        j = random.randint(0, count-1)
        df['title'][i] = df['title'][j]
        if i != j:
            df['label'][i] = 0

    return df


class InputDataSet():

    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self, ):
        return len(self.data)

    def __getitem__(self, item):  # item是索引 用来取数据
        text = [str(self.data['title'][item]), str(self.data['abst'][item])]
        labels = self.data['label'][item]
        labels = torch.tensor(labels, dtype=torch.long)

        encoding = self.tokenizer.encode_plus(
            str(self.data['title'][item]),
            str(self.data['abst'][item]),
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            truncation='longest_first',
            return_attention_mask=True,
            return_tensors='pt',
        )

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

        return {
            'texts': text,
            'input_ids': encoding['input_ids'].flatten(),  # 通过tokenizer做的编码 101 。。。。 102
            'attention_mask': encoding['attention_mask'].flatten(),  # 注意力编码111111000000
            "token_type_ids": encoding['token_type_ids'].flatten(),  # 分句编码11111000000
            'labels': labels
        }


if __name__ == '__main__':
    data_dir = '../data/data.xlsx'

    df = read_data_for_bert(data_dir)

    print(df)

    print(df['keyword'])
    print(df['keywords'])
    # print(df['num'])

    # print(df['keyword'])
    print(df.info())
    perc = [.70, .80, .90, .95]
    print(df.describe(percentiles=perc))

    model_dir = '../chinese_wwm_pytorch'
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    train_dataset = InputDataSet(df, tokenizer=tokenizer, max_len=512)
    train_dataloader = DataLoader(train_dataset, batch_size=32)
    batch = next(iter(train_dataloader))
    print(batch)
    print(batch['input_ids'][1])
    print(batch['input_ids'].shape)
    print(batch['attention_mask'].shape)
    print(batch['token_type_ids'].shape)
    print(batch['labels'].shape)

    print(tokenizer.convert_ids_to_tokens(batch['input_ids'][1]))
