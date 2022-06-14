# -*- coding: utf-8 -*-
# @Time    : 2021/6/10
# @Author  : kaka


import argparse
import logging
import os
from pathlib import Path

from datasets import load_dataset
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer
from SimCSE import SimCSE
from CSECollator import CSECollator
from data_process import *


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train_file", type=str, help="train text file")
    parser.add_argument("--pretrained", type=str, default="hfl/chinese-bert-wwm-ext",
                        help="huggingface pretrained model")
    parser.add_argument("--model_out", type=str, default="./model/out", help="model output path")
    parser.add_argument("--num_proc", type=int, default=5, help="dataset process thread num")
    parser.add_argument("--max_length", type=int, default=512, help="sentence max length")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--epochs", type=int, default=2, help="epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--tao", type=float, default=0.05, help="temperature")
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument("--display_interval", type=int, default=50, help="display interval")
    parser.add_argument("--save_interval", type=int, default=100, help="save interval")
    parser.add_argument("--pool_type", type=str, default="cls", help="pool_type")
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="dropout_rate")
    parser.add_argument("--gpu_no", type=str, default='1', help="device")
    args = parser.parse_args()
    return args


def load_data(args, tokenizer):
    data_files = {"train": args.train_file}
    ds = load_dataset("text", data_files=data_files)
    ds_tokenized = ds.map(lambda example: tokenizer(example["text"]), num_proc=args.num_proc)
    collator = CSECollator(tokenizer, max_len=args.max_length)
    dl = DataLoader(ds_tokenized["train"],
                    batch_size=args.batch_size,
                    collate_fn=collator.collate)
    return dl


def load_data_csv(args, tokenizer):

    df = read_data(args.train_file)
    dataset = InputDataSet(df, tokenizer=tokenizer, max_len=args.max_length)
    dl = DataLoader(dataset,
                    batch_size=args.batch_size)
    return dl


def compute_loss(y_pred, tao=0.05, device="cuda"):
    idxs = torch.arange(0, y_pred.shape[0], device=device)
    y_true = idxs + 1 - idxs % 2 * 2
    similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)
    similarities = similarities - torch.eye(y_pred.shape[0], device=device) * 1e12
    similarities = similarities / tao
    loss = F.cross_entropy(similarities, y_true)
    return torch.mean(loss)


def train(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_no
    tokenizer = BertTokenizer.from_pretrained(args.pretrained, mirror="tuna")
    dl = load_data_csv(args, tokenizer)
    model = SimCSE(args.pretrained, args.pool_type, args.dropout_rate).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model_out = Path(args.model_out)
    if not model_out.exists():
        os.mkdir(model_out)

    model.train()
    batch_idx = 0
    for epoch_idx in range(args.epochs):
        for data in tqdm(dl):
            batch_idx += 1
            # 维度转换 [batch, 2, seq_len] -> [batch * 2, sql_len]
            real_batch_num = data.get('input_ids').shape[0]
            input_ids = data.get('input_ids').view(real_batch_num * 2, -1).to(args.device)
            attention_mask = data.get('attention_mask').view(real_batch_num * 2, -1).to(args.device)
            token_type_ids = data.get('token_type_ids').view(real_batch_num * 2, -1).to(args.device)
            pred = model(input_ids=input_ids,
                         attention_mask=attention_mask,
                         token_type_ids=token_type_ids)
            loss = compute_loss(pred, args.tao, args.device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.item()
            if batch_idx % args.display_interval == 0:
                logging.info(f"batch_idx: {batch_idx}, loss: {loss:>10f}")
            # if batch_idx % args.save_interval == 0:
            #     torch.save(model.state_dict(),
            #                model_out / "epoch_{0}-batch_{1}-loss_{2:.6f}".format(epoch_idx, batch_idx, loss))
        torch.save(model.state_dict(),
                   model_out / "epoch_{0}-batch_{1}".format(epoch_idx, batch_idx))


def main():
    args = parse_args()
    train(args)
    # tokenizer = BertTokenizer.from_pretrained(args.pretrained, mirror="tuna")
    # print(args.pretrained)


if __name__ == "__main__":
    log_fmt = "%(asctime)s|%(name)s|%(levelname)s|%(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
