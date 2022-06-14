from argparse import ArgumentParser

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertTokenizer
from transformers.modeling_utils import SequenceSummary
from torch.optim import AdamW


class JointKpe(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        config = BertConfig.from_pretrained(self.args.pretrained)
        self.model = BertModel.from_pretrained(self.args.pretrained)
        self.pooler = SequenceSummary(config)
        self.pooler.summary_type='first'
        self.classifier = nn.Linear(config.hidden_size, self.args.num_classes)
        self.activation = nn.ReLU()
        self.lamda = self.args.lamda
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

    def forward(self, input_ids, attention_mask, token_type_ids,word_segments,word_segments_len,kpe_label):
        output = self.model(input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids)
        last_hidden_state = output.last_hidden_state
        h_cls = self.pooler(last_hidden_state)
        cls_hidden_state = self.dropout(h_cls)
        logit = self.classifier(cls_hidden_state)
        l_cls = self.sigmoid(logit)
        # lil_logits = self.lil(phrase_level_hidden=last_hidden_state)
        # lil_logits_mean = torch.mean(lil_logits, dim=1)
        # logits = cls_logit + self.lamda * lil_logits_mean
        # predicted_labels = torch.argmax(logits, -1)
        batch_size = self.args.batch_size
        l_mean_batch=[]
        device = self.args.device
        for i in range(len(word_segments_len)):
            length = word_segments_len[i]
            if length==0:
                print("exist length==0")
                l_mean = l_cls[i]
                l_mean_batch.append(l_mean.reshape(1, -1))
                continue

            word_list=[]
            for j in range(length):
                word_list.append(word_segments[j][i])
            #word_dict = {}
            ind = 0
            word_em_list = []
            for word in word_list:
                em , ind = self.get_word_embedding(last_hidden_state[i], input_ids[i], word, ind)
                word_em_list.append(em)
                # if word not in word_dict:
                #     word_dict[word] = self.get_word_embedding(last_hidden_state[i], input_ids[i], word)
        #     word_logit = {}
        #     for word in word_dict:
        #         h = word_dict[word]
        #         z = self.activation(h) - self.activation(h_cls[i])
        #         l = z
        #         word_logit[word] = l
        #     li = [word_logit[word].reshape(1,-1) for word in word_logit]
        #     res = torch.cat(li,dim=0)
        #     res = self.sigmoid( self.classifier(self.dropout(res)))
        #     l_mean = torch.mean(res,dim=0)
        #     l_mean_batch.append(l_mean.reshape(1,-1))
        # l_mean = torch.cat(l_mean_batch, dim=0)

        return l_cls #,l_mean

    def get_contribute(self, input_ids, attention_mask, token_type_ids,word_segments,word_segments_len):
        output = self.model(input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids)
        last_hidden_state = output.last_hidden_state
        h_cls = self.pooler(last_hidden_state)
        logit = self.classifier(h_cls)
        l_cls = self.sigmoid(logit)
        device = self.args.device

        length = word_segments_len
        # if length==0:
        #     print("exist length==0")
        #     l_mean = l_cls[i]
        #     l_mean_batch.append(l_mean.reshape(1, -1))
        #     continue

        word_list=[]
        for j in range(length):
            word_list.append(word_segments[j])
        word_dict = {}
        for word in word_list:
            if word not in word_dict:
                word_dict[word] = self.get_word_embedding(last_hidden_state, input_ids, word)
        word_logit = {}
        contri_dict ={}
        for word in word_dict:
            h = word_dict[word]
            z = self.activation(h) - self.activation(h_cls)
            word_logit[word] = z
            l = self.sigmoid(self.classifier(z))
            r= l_cls - l
            contri_dict[word] = r
        # li = [word_logit[word].reshape(1,-1) for word in word_logit]
        # res = torch.cat(li,dim=0)
        # l = self.sigmoid( self.classifier(res))
        # r = l_cls - res

        return l_cls,contri_dict

    def lil(self, phrase_level_hidden):
        phrase_level_activations = self.activation(phrase_level_hidden)
        pooled_seq_rep = self.pooler(phrase_level_hidden).unsqueeze(1)
        phrase_level_activations = phrase_level_activations - pooled_seq_rep
        phrase_level_logits = self.classifier(phrase_level_activations)
        return phrase_level_logits

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.args.lr, betas=(0.9, 0.99),
                     eps=1e-8)

    def get_word_embedding(self, last_hidden_state, input_ids, word, ind = 0):
        words = self.tokenizer.encode(word,add_special_tokens=False)
        #ind = 0
        for i in range(ind, len(input_ids)-len(words)+1):
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
        embedding = sum(last_hidden_state[ind:ind + len(words)]) / len(words)
        return embedding ,ind

    def get_word_embedding(self, last_hidden_state, input_ids, word):
        words = self.tokenizer.encode(word,add_special_tokens=False)
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
        embedding = sum(last_hidden_state[0][ind:ind + len(words)]) / len(words)
        return embedding
