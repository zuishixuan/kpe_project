import pandas
from datasets import load_dataset
from transformers import BertTokenizer
from CSECollator import CSECollator
from torch.utils.data import DataLoader
import numpy as np
from transformers import BertConfig, BertModel
import torch

from SimCSE import SimCSE

MODEL_DIR = './model/chinese-bert-wwm-ext'


def load1():
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR, mirror="tuna")
    data_files = {"train": '../data/data3.csv'}
    ds = load_dataset('csv', data_files=data_files)
    ds_tokenized = ds.map(lambda example: tokenizer(example["name"] + '。' + example["describe"]), num_proc=5)
    collator = CSECollator(tokenizer, max_len=100)
    dl = DataLoader(ds_tokenized["train"],
                    batch_size=64,
                    collate_fn=collator.collate)
    print(ds_tokenized)


def load2():
    data_files = {"train": './data/news_title.txt'}
    ds = load_dataset("text", data_files=data_files)
    print(ds)


def load3():
    fname = '../data/data3.csv'
    df = pandas.read_csv(fname, encoding='utf-8', index_col=0)
    df['text'] = df.apply(lambda x: (x["name"] + '。' + x["describe"]), axis=1)
    for i in df['text']:
        print(i)
    print(df)


def load4():
    vecs_out_path = './model/out_vec/np_vecs.npy'
    ids_out_path = './model/out_vec/np_ids.npy'
    np_vec = np.load(vecs_out_path)
    print(np_vec.shape)
    np_id = np.load(ids_out_path)
    print(np_id.shape)
    id2text = {idx: text for idx, text in zip(np_id, np_vec)}
    print(id2text)


def getBertEmbed():
    simcse_model_path = "./model/out/epoch_1-batch_1250"
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR, mirror="tuna")
    conf = BertConfig.from_pretrained(MODEL_DIR)
    # conf.attention_probs_dropout_prob = dropout_prob
    # conf.hidden_dropout_prob = dropout_prob
    # encoder = BertModel.from_pretrained(MODEL_DIR, config=conf)
    sentence = "视频展示台。视频展台是一种视频输出设备，在教育系统得到广泛应用。本项目依据JY /T 0363-2002《视频展示台》可开展部分参数的检测，包括：外观和结构、功能、亮度分解力、亮度鉴别等级、变焦倍数、灯箱照度均匀度、标记和使用说明、绝缘电阻和抗电强度、保护接地连接电阻和电磁兼容性(注入电源骚扰电压)。"
    # encoder.load_state_dict(torch.load(simcse_model_path))
    encoder = SimCSE(pretrained=MODEL_DIR).to('cuda')
    encoder.load_state_dict(torch.load(simcse_model_path))
    encoding = tokenizer.encode_plus(sentence,
                                     add_special_tokens=True,
                                     max_length=100,
                                     return_token_type_ids=True,
                                     padding='max_length',
                                     truncation='longest_first',
                                     return_attention_mask=True,
                                     return_tensors='pt',
                                     )
    re = {
        'texts': sentence,
        'input_ids': encoding['input_ids'].flatten(),  # 通过tokenizer做的编码 101 。。。。 102
        'attention_mask': encoding['attention_mask'].flatten(),  # 注意力编码111111000000
        "token_type_ids": encoding['token_type_ids'].flatten(),  # 分句编码11111000000
    }
    # output = encoder.encode(input_ids=encoding['input_ids'].to('cuda'),
    #                  attention_mask=encoding['attention_mask'].to('cuda'),
    #                  token_type_ids=encoding['token_type_ids'].to('cuda'))
    output = encoder.encode(encoding['input_ids'].to('cuda'), encoding['attention_mask'].to('cuda'),
                            encoding['token_type_ids'].to('cuda'), ['视频展示台','视频展台'])
    print(output)


def main():
    getBertEmbed()


if __name__ == "__main__":
    main()
