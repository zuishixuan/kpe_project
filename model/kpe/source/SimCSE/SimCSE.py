# -*- coding: utf-8 -*-
# @Time    : 2021/6/10
# @Author  : kaka

import torch.nn as nn
from transformers import BertConfig, BertModel, BertTokenizer


class SimCSE(nn.Module):
    def __init__(self, pretrained="hfl/chinese-bert-wwm-ext", pool_type="cls", dropout_prob=0.3):
        super().__init__()
        conf = BertConfig.from_pretrained(pretrained)
        conf.attention_probs_dropout_prob = dropout_prob
        conf.hidden_dropout_prob = dropout_prob
        self.encoder = BertModel.from_pretrained(pretrained, config=conf)
        assert pool_type in ["cls", "pooler"], "invalid pool_type: %s" % pool_type
        self.pool_type = pool_type
        self.tokenizer = BertTokenizer.from_pretrained(pretrained, mirror="tuna")

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.encoder(input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids)
        if self.pool_type == "cls":
            output = output.last_hidden_state[:, 0]
        elif self.pool_type == "pooler":
            output = output.pooler_output
        return output

    def encode(self, input_ids, attention_mask, token_type_ids):
        output = self.encoder(input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids)
        # last_hidden_state = output.last_hidden_state
        # word_dict = {}
        # for word in word_list:
        #     if word not in word_dict:
        #         word_dict[word] = self.get_word_embedding(last_hidden_state, input_ids, word)
        # if self.pool_type == "cls":
        #     sen_embedding = output.last_hidden_state[:, 0]
        # elif self.pool_type == "pooler":
        #     sen_embedding = output.pooler_output

        # if self.pool_type == "cls":
        #     output = output.last_hidden_state[:, 0]
        # elif self.pool_type == "pooler":
        #     output = output.pooler_output
        return output

    def get_word_embedding(self, last_hidden_state, input_ids, word):
        words = self.tokenizer.encode(word,add_special_tokens=False)
        # print("words:")
        # print(words)
        # print("input_ids:")
        # print(input_ids[0])
        ind = 0
        for i in range(len(input_ids[0])):
            flag = True
            for j in range(len(words)):
                if input_ids[0][i + j] == words[j]:
                    continue
                else:
                    flag = False
                    break
            if flag:
                ind = i
                break
        # print(ind)
        # print(last_hidden_state[ind:ind + len(words)])
        embedding = sum(last_hidden_state[0][ind:ind + len(words)]) / len(words)
        return embedding

