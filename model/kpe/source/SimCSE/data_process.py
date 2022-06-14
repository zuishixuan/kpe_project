import random
import pandas as pd
from transformers import Trainer, TrainingArguments, BertTokenizer, BertModel, BertPreTrainedModel, BertConfig
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput


def read_data(data_dir='../data/data4.csv'):
    df = pd.read_csv(data_dir)
    return df


class InputDataSet(Dataset):

    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self, ):
        return len(self.data)

    def __getitem__(self, item):  # item是索引 用来取数据
        text = str(self.data['text'][item])
        # new_batch = []
        encoding = self.tokenizer.batch_encode_plus(
            [text, text],
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            truncation='longest_first',
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'texts': text,
            'input_ids': encoding['input_ids'],  # 通过tokenizer做的编码 101 。。。。 102
            'attention_mask': encoding['attention_mask'],  # 注意力编码111111000000
            "token_type_ids": encoding['token_type_ids'],  # 分句编码11111000000
        }


if __name__ == '__main__':
    data_dir = 'data/data.xlsx'
