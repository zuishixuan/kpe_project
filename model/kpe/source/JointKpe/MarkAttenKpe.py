from argparse import ArgumentParser

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertTokenizer
from transformers.modeling_utils import SequenceSummary
from torch.optim import AdamW
from torch.nn import MarginRankingLoss
from ..SimCSE.SimCSE import SimCSE
from TorchCRF import CRF
from d2l import torch as d2l
from torch.nn import functional as F


class MarkAttenKpe(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device
        config = BertConfig.from_pretrained(args.pretrained)
        self.model = SimCSE(pretrained=args.pretrained).to(self.device)
        self.model.load_state_dict(torch.load(args.simcse_model))
        # self.model = BertModel.from_pretrained(self.args.pretrained)
        self.pooler = SequenceSummary(config)
        self.pooler.summary_type = 'first'
        self.classifier = nn.Linear(args.atten_hidden_size, args.num_labels)
        self.activation = nn.ReLU()
        self.lamda = args.lamda
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.loss = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
        self.tokenizer = BertTokenizer.from_pretrained(args.pretrained)
        self.crf = CRF(args.num_labels)
        self.attention = d2l.MultiHeadAttention(key_size=config.hidden_size,
                                                value_size=config.hidden_size,
                                                query_size=config.hidden_size,
                                                num_heads=2,
                                                num_hiddens=args.atten_hidden_size,
                                                dropout=config.hidden_dropout_prob)

        # self.init_weights()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--min_lr", default=0, type=float,
                            help="Minimum learning rate.")
        parser.add_argument("--h_dim", type=int,
                            help="Size of the hidden dimension.", default=768)
        parser.add_argument("--num_classes", type=float,
                            help="Number of classes.", default=13)
        parser.add_argument("--lr", default=2e-5, type=float,
                            help="Initial learning rate.")
        parser.add_argument("--weight_decay", default=0.01, type=float,
                            help="Weight decay rate.")
        parser.add_argument("--lamda", default=0.01, type=float, help="Lamda Parameter")
        return parser

    def forward(self, input_ids, attention_mask, token_type_ids, word_segments, mark_label=None):
        output = self.model.encode(input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids)
        last_hidden_state = output.last_hidden_state

        batch_size = input_ids.shape[0]
        device = self.args.device
        batch_score_list = []
        word_segments = list(transpose(word_segments))
        batch_word_em_list = []
        batch_word_len_list = []
        for i in range(batch_size):
            word_list = word_segments[i]

            ind = 0
            word_em_list = []
            for word in word_list:
                if word == '':
                    break
                em, ind = self.get_word_embedding(last_hidden_state=last_hidden_state[i],
                                                  input_ids=input_ids[i],
                                                  word=word,
                                                  ind=ind)
                word_em_list.append(em)
            embedding = torch.stack(word_em_list)
            length = len(embedding)
            pad_embedding = F.pad(embedding, (0, 0, 0, self.args.max_word_count - length))
            batch_word_em_list.append(pad_embedding)
            batch_word_len_list.append(length)
            # self.attention(embedding,embedding,embedding,le)
            # score_list = self.classifier(self.dropout(embedding))
            # batch_score_list.append(score_list)
        batch_word_em = torch.stack(batch_word_em_list)
        batch_word_len = torch.tensor(batch_word_len_list, dtype=torch.long, device=self.device)
        atten = self.attention(batch_word_em, batch_word_em, batch_word_em, batch_word_len)
        score_list = self.classifier(self.dropout(atten))
        if mark_label is not None:
            mask = mark_label.gt(-1)
            loss = self.crf(score_list,mark_label,mask) * (-1)
            mark_loss = torch.mean(loss)
            return mark_loss
        else:
            return score_list

    def get_contribute(self, input_ids, attention_mask, token_type_ids, word_segments):
        output = self.model.encode(input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids)
        last_hidden_state = output.last_hidden_state

        batch_size = 1
        device = self.args.device
        word_list = word_segments

        ind = 0
        word_em_list = []
        for word in word_list:
            if word == '':
                break
            em, ind = self.get_word_embedding(last_hidden_state=last_hidden_state[0],
                                              input_ids=input_ids[0],
                                              word=word,
                                              ind=ind)
            word_em_list.append(em)
        # print(word_em_list)
        embedding = torch.stack(word_em_list)
        length = len(embedding)
        pad_embedding = F.pad(embedding, (0, 0, 0, self.args.max_word_count - length))
        batch_word_len = torch.tensor([length], dtype=torch.long, device=self.device)
        atten = self.attention(pad_embedding.unsqueeze(0), pad_embedding.unsqueeze(0), pad_embedding.unsqueeze(0), batch_word_len)
        score_list = self.classifier(atten)
        # mask_list = [1 if i<length else 0 for i in range(self.args.max_word_count)]
        # mask = torch.Tensor(mask_list).long()
        # res = self.crf.viterbi_decode(score_list, mask.unsqueeze(0))[0]

        return score_list

    def generateMarkAns(self, word_list, score_list):
        length = len(word_list)
        mask_list = [1 if i<length else 0 for i in range(self.args.max_word_count)]
        mask = torch.Tensor(mask_list).long()
        res_code = self.crf.viterbi_decode(score_list, mask.unsqueeze(0))[0]
        keywords_res = {}
        word = ''
        for i, w in zip(res_code, word_list):
            if i == 1:
                word = w
            if i == 2:
                word += w
            if i == 3:
                word += w
                if word in keywords_res:
                    keywords_res[word] += 1
                else:
                    keywords_res[word] = 1
            if i == 4:
                word = w
                if word in keywords_res:
                    keywords_res[word] += 1
                else:
                    keywords_res[word] = 1
        return keywords_res

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.args.lr, betas=(0.9, 0.99),
                     eps=1e-8)

    def get_word_embedding(self, last_hidden_state, input_ids, word, ind):
        words = self.tokenizer.encode(word, add_special_tokens=False)
        # ind = 0
        # words = torch.Tensor(words).to(self.device)
        for i in range(ind, len(input_ids) - len(words) + 1):
            flag = True

            for j in range(len(words)):
                if input_ids[i + j] == words[j]:
                    continue
                else:
                    flag = False
                    break
            if flag:
                ind = i
                break
        #     if words.equal(input_ids[i:i+len(words)]):
        #                 ind = i
        #                 break
        # words.to('cpu')
        embedding = sum(last_hidden_state[ind:ind + len(words)]) / len(words)
        return embedding, ind


def transpose(matrix):
    return zip(*matrix)
