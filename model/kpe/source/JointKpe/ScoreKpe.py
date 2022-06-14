from argparse import ArgumentParser

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertTokenizer
from transformers.modeling_utils import SequenceSummary
from torch.optim import AdamW
from torch.nn import MarginRankingLoss
from ..SimCSE.SimCSE import SimCSE
from torch.nn import functional as F


class ScoreKpe(nn.Module):
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
        self.score = nn.Linear(config.hidden_size, 1)
        self.activation = nn.ReLU()
        self.lamda = args.lamda
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.loss = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
        self.tokenizer = BertTokenizer.from_pretrained(args.pretrained)

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

    def forward(self, input_ids, attention_mask, token_type_ids, word_segments, kpe_label=None):
        output = self.model.encode(input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids)
        last_hidden_state = output.last_hidden_state

        batch_size = input_ids.shape[0]
        device = self.args.device
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
        # atten = self.attention(batch_word_em, batch_word_em, batch_word_em, batch_word_len)
        batch_score_list = self.score(self.dropout(batch_word_em))

        if kpe_label is not None:
            Rank_Loss_Fct = MarginRankingLoss(margin=1, reduction="mean")
            kpe_label = list(transpose(kpe_label))
            rank_losses = []
            for i in range(batch_size):
                label_list = kpe_label[i]
                score_list = batch_score_list[i]
                label_list = torch.stack(label_list[:score_list.shape[0]])
                flag = torch.FloatTensor([1]).to(device)
                true_score = score_list[label_list == 1]
                neg_score = score_list[label_list == 0]
                loss = Rank_Loss_Fct(true_score.unsqueeze(-1), neg_score.unsqueeze(0), flag)
                if loss != loss:
                    continue
                rank_losses.append(loss)
            rank_loss = torch.mean(torch.stack(rank_losses))
            return rank_loss
        else:
            return batch_score_list

    # def forward(self, input_ids, attention_mask, token_type_ids, word_segments, word_segments_len, kpe_label=None):
    #     output = self.model.encode(input_ids,
    #                                attention_mask=attention_mask,
    #                                token_type_ids=token_type_ids)
    #     last_hidden_state = output.last_hidden_state
    #
    #     batch_size = len(word_segments_len)
    #     device = self.args.device
    #     batch_score_list = []
    #     word_segments = list(transpose(word_segments))
    #     for i in range(batch_size):
    #         length = word_segments_len[i].item()
    #         word_list = word_segments[i]
    #
    #         ind = 0
    #         word_em_list = []
    #         for word in word_list:
    #             if word == '':
    #                 break
    #             em, ind = self.get_word_embedding(last_hidden_state=last_hidden_state[i],
    #                                               input_ids=input_ids[i],
    #                                               word=word,
    #                                               ind=ind)
    #             word_em_list.append(em)
    #         #print(word_em_list)
    #         embedding= []
    #         try:
    #             embedding = torch.stack(word_em_list)
    #         except Exception as E_results:
    #             print("捕捉有异常：", E_results)
    #             print('len',len(word_em_list))
    #             print('ind',ind)
    #             for ind,el in enumerate(word_em_list):
    #                 if isinstance(el,float):
    #                   print(el)
    #                   print(word_list[ind])
    #             #print(word_em_list)
    #             print(word_list)
    #
    #         score_list = self.score(self.dropout(embedding)).reshape(-1)
    #         batch_score_list.append(score_list)
    #
    #     if kpe_label is not None:
    #         Rank_Loss_Fct = MarginRankingLoss(margin=1, reduction="mean")
    #         kpe_label = list(transpose(kpe_label))
    #         rank_losses = []
    #         for i in range(batch_size):
    #             label_list = kpe_label[i]
    #             score_list = batch_score_list[i]
    #             label_list = torch.stack(label_list[:score_list.shape[0]])
    #             flag = torch.FloatTensor([1]).to(device)
    #             true_score = score_list[label_list == 1]
    #             neg_score = score_list[label_list == 0]
    #             loss = Rank_Loss_Fct(true_score.unsqueeze(-1), neg_score.unsqueeze(0), flag)
    #             if loss!=loss:
    #                 continue
    #             rank_losses.append(loss)
    #         rank_loss = torch.mean(torch.stack(rank_losses))
    #         return rank_loss
    #     else:
    #         return batch_score_list

    def get_contribute(self, input_ids, attention_mask, token_type_ids, word_segments, word_segments_len,
                       kpe_label=None):
        output = self.model.encode(input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids)
        last_hidden_state = output.last_hidden_state

        batch_size = 1
        device = self.args.device
        batch_score_list = []
        # word_segments = list(transpose(word_segments))
        for i in range(batch_size):
            word_list = word_segments

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
            # print(word_em_list)
            t = []
            try:
                t = torch.stack(word_em_list)
            except Exception as E_results:
                print("捕捉有异常：", E_results)
                print('len', len(word_em_list))
                print('ind', ind)
                for ind, el in enumerate(word_em_list):
                    if isinstance(el, float):
                        print(el)
                        print(word_list[ind])
                # print(word_em_list)
                print(word_list)

            score_list = self.score(t).reshape(-1)
            li_uni = list(set(word_list))
            li_uni.sort(key=word_list.index)
            word_dict = {}
            for w in li_uni:
                if w == '': continue
                ind_list = [ind for ind, wl in enumerate(word_list) if w == wl]
                word_dict[w] = max([score_list[ind] for ind in ind_list])

            # batch_score_list.append(score_list)
            # score_list = batch_score_list[0]

        return word_dict

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
