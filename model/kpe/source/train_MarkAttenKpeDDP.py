import argparse
import datetime
import logging
import os
import time
from pathlib import Path
from .JointKpe.MarkAttenKpe import MarkAttenKpe as Model
from .JointKpe.data_process_markkpe import *
from transformers import Trainer, TrainingArguments, BertTokenizer, BertModel, BertPreTrainedModel, BertConfig
from torch.utils.data import Dataset, DataLoader, RandomSampler, random_split, SequentialSampler
from transformers import get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm
from .Segmentation import Segmentation
import torch.distributed as dist
import torch.distributed.launch

import torch
import transformers
from .constant import get_real_dir


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train_file", type=str, default="../data/cnki_2/data_mark.csv", help="train text file")
    parser.add_argument("--pretrained", type=str, default="../model/chinese_roberta_wwm_ext_pytorch",
                        help="huggingface pretrained model")
    parser.add_argument("--simcse_model", type=str, default="../model/SimCSE/cnki_2/out/epoch_1-batch_12100")
    parser.add_argument("--model_out", type=str, default="../model/MarkAttenKpe/cnki_2/out", help="model output path")
    parser.add_argument("--markkpe_model", type=str, default="../model/MarkAttenKpe/cnki_2/out/epoch_2")
    parser.add_argument("--user_dict", type=str, default='../data/cnki_2/keywords_dict.txt')
    parser.add_argument("--num_proc", type=int, default=5, help="dataset process thread num")
    parser.add_argument("--max_length", type=int, default=500, help="sentence max length")
    parser.add_argument("--max_word_count", type=int, default=300, help="woed max num")
    parser.add_argument("--atten_hidden_size", type=int, default=400)
    parser.add_argument("--num_labels", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--epochs", type=int, default=4, help="epochs")
    parser.add_argument("--tao", type=float, default=0.05, help="temperature")
    parser.add_argument("--device", type=str, default="cuda:1", help="device")
    parser.add_argument("--display_interval", type=int, default=50, help="display interval")
    parser.add_argument("--save_interval", type=int, default=100, help="save interval")
    parser.add_argument("--pool_type", type=str, default="cls", help="pool_type")
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="dropout_rate")
    parser.add_argument("--gpu_no", type=str, default='0,1', help="device")
    # DDP ???1????????????`torch.distributed`?????????????????????`main.py(??????????????????py??????)`??????????????????????????????**???????????????`--local_rank`**
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser = Model.add_model_specific_args(parser)

    args = parser.parse_args()
    return args


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # ???????????????????????????
    elapsed_rounded = int(round((elapsed)))

    # ???????????? hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# ???????????????????????????????????????????????????
def flat_accuracy(preds, labels):
    # print('preds')
    # print(preds)
    # print('labels')
    # print(labels)
    # pred_flat = preds.ge(0.5)
    # labels_flat = labels
    # print('preds_flat')
    # print(pred_flat)
    # print('labels_flat')
    # print(labels_flat)
    return sum(row.all().int().item() for row in (preds.ge(0.5) == labels)) / preds.shape[0]


def train(args):
    transformers.logging.set_verbosity_error()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_no
    # DDP ???2????????? init_process_group ??????GPU ???????????????????????????????????????
    # dist.init_process_group(backend='nccl')
    # torch.cuda.set_device(args.local_rank)

    local_rank = args.local_rank
    if local_rank != -1:
        dist_backend = 'nccl'
        dist.init_process_group(backend=dist_backend)  # ???????????????????????????????????? distributed ???
        device = local_rank
        torch.cuda.set_device(local_rank)
    else:
        device = args.device
        torch.device(device) if torch.cuda.is_available() else torch.device('cpu')

    # device = local_rank if local_rank != -1 else (
    #     torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))
    # torch.cuda.set_device(local_rank)  # ?????????????????????gpu


    args.train_file = get_real_dir(args.train_file)
    args.model_out = get_real_dir(args.model_out)
    args.pretrained = get_real_dir(args.pretrained)
    args.user_dict = None
    args.simcse_model = get_real_dir(args.simcse_model)

    df = read_data_for_bert(args.train_file) #.head(200)

    tokenizer = BertTokenizer.from_pretrained(args.pretrained)

    dataset = InputDataSet(df, tokenizer=tokenizer, max_len=args.max_length, user_dict=args.user_dict)

    # ?????????????????????????????????
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    # ???????????????????????????????????????????????????
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))

    # ??? fine-tune ???????????????BERT ????????????????????????????????? 16 ??? 32
    batch_size = args.batch_size

    # DDP ???3????????? DistributedSampler ???????????????????????????
    if local_rank != -1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, rank=local_rank)
    else:
        train_sampler = RandomSampler(train_dataset)


    train_dataloader = DataLoader(
        train_dataset,  # ????????????
        # sampler=RandomSampler(train_dataset),  # ???????????????
        batch_size=batch_size,  # ????????????????????????
        sampler=train_sampler
    )

    # ??????????????????????????????????????????????????????
    # val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    validation_dataloader = DataLoader(
        val_dataset,  # ????????????
        sampler=SequentialSampler(val_dataset),  # ?????????????????????
        batch_size=batch_size
    )

    # ?????? Bert, ????????? BERT ?????? + ????????????????????????
    #DDP ???4?????????

    model = Model(args=args).to(device)
    if local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)

    # ??????????????????????????????????????????

    params = list(model.named_parameters())

    print('The BERT model has {:} different named parameters.\n'.format(len(params)))

    print('==== Embedding Layer ====\n')

    for p in params[0:5]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print('\n==== First Transformer ====\n')

    for p in params[5:21]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print('\n==== Output Layer ====\n')

    for p in params[-4:]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    # ???????????????16, 32
    # ????????????Adam??????5e-5, 3e-5, 2e-5
    # epochs
    # ????????????2, 3, 4
    # ????????????????????????
    #
    # Batch
    # size: 32????????????
    # DataLoaders
    # ????????????
    # Learning
    # rate???2e-5
    # Epochs??? 4???????????????????????????????????????????????????????????????

    # ????????? 'W' ?????? '??????????????????"
    # optimizer = model.configure_optimizers()
    #DDP ???5????????????????????????module
    if local_rank != -1:
        optimizer = model.module.configure_optimizers()
    else:
        optimizer = model.configure_optimizers()

    # ?????? epochs??? BERT ??????????????? 2 ??? 4 ?????????????????????????????????
    epochs = args.epochs

    # ?????????????????????
    total_steps = len(train_dataloader) * epochs

    # ????????????????????????
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)

    # ???????????????????????????????????????????????????
    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # ???????????????????????? loss??????????????????????????????????????????,
    training_stats = []

    # ????????????????????????
    total_t0 = time.time()

    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # ???????????? epoch ???????????????
        t0 = time.time()

        # ???????????? epoch ???????????? loss
        total_train_loss = 0
        loss_batch = 0

        # ???????????????????????????????????????????????????????????????????????????
        # dropout???batchnorm ??????????????????????????????????????????????????? (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # ????????????????????????
        for step, batch in tqdm(enumerate(train_dataloader)):

            # ?????????40?????????????????????????????????
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                loss_batch = loss_batch / 40
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.  Loss: {:}'.format(step, len(train_dataloader),
                                                                                       elapsed, loss_batch))
                loss_batch = 0

            # ??????????????????????????????????????? gpu ???
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_token_type_ids = batch['token_type_ids'].to(device)
            # b_labels = batch['labels'].to(device)
            word_segments = batch['word_segments']
            word_segments_len = batch['word_segments_len']
            mark_label = batch['mark_label'].to(device)

            # ????????????????????????????????????????????? 0????????? pytorch ?????????????????????
            model.zero_grad()

            # ????????????
            # ????????????:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # ???????????????????????????????????????????????????????????? ?????????, ????????? loss ??? logits -- ?????????????????????
            loss = model(b_input_ids,
                         attention_mask=b_input_mask,
                         token_type_ids=b_token_type_ids,
                         word_segments=word_segments,
                         mark_label=mark_label)
            # print(output)
            # loss = model.loss(l_cls,b_labels) + args.lamda*model.loss(l_mean,b_labels)

            # ?????? loss
            loss_batch += loss.item()
            total_train_loss += loss.item()

            # ????????????
            loss.backward()

            # ?????????????????????????????????????????????
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # ????????????
            optimizer.step()

            # ???????????????
            scheduler.step()

        # ??????????????????
        avg_train_loss = total_train_loss / len(train_dataloader)

        # ?????? epoch ???????????????
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))
        # ========================================
        #               Validation
        # ========================================
        # ???????????? epoch ????????????????????????????????????????????????

        print("")
        print("Running Validation...")

        t0 = time.time()

        # ???????????????????????????
        model.eval()

        # Tracking variables
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in tqdm(validation_dataloader):
            # ???????????????????????? gpu ???
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_token_type_ids = batch['token_type_ids'].to(device)
            # b_labels = batch['labels'].to(device)
            word_segments = batch['word_segments']
            word_segments_len = batch['word_segments_len']
            mark_label = batch['mark_label'].to(device)

            # ???????????????????????????????????????????????????
            with torch.no_grad():
                loss = model(b_input_ids,
                             attention_mask=b_input_mask,
                             token_type_ids=b_token_type_ids,
                             word_segments=word_segments,
                             mark_label=mark_label)
            # print(output)
            # loss = model.loss(l_cls, b_labels) + args.lamda * model.loss(l_mean, b_labels)
            # logits = l_cls

            # ?????? loss
            total_eval_loss += loss.item()

            # ?????????????????? labels ????????? cpu ?????????
            # logits = logits
            # label_ids = b_labels

            # ???????????????
            # total_eval_accuracy += flat_accuracy(logits, label_ids)

        # ???????????? epoch ????????????
        # avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        # print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        # ???????????? epoch ??? loss
        avg_val_loss = total_eval_loss / len(validation_dataloader)

        # ???????????????????????????
        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        # ???????????? epoch ?????????????????????
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                #                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )
        # ????????????????????????
        model_out = Path(args.model_out)
        if not model_out.exists():
            os.makedirs(model_out, exist_ok=True)

        # DDP ???6???????????????????????????module
        print("Saving model to %s" % model_out)
        if local_rank == -1:
            torch.save(model.state_dict(),
                       model_out / "epoch_{0}".format(epoch_i))
        if local_rank == 0:
            torch.save(model.module.state_dict(),
                       model_out / "epoch_{0}".format(epoch_i))

    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

    # ?????? 2 ?????????
    pd.set_option('precision', 2)

    # ????????????????????? DataFrame ???
    df_stats = pd.DataFrame(data=training_stats)

    # ?????? epoch ????????????????????????
    df_stats = df_stats.set_index('epoch')

    # ??????????????????
    print(df_stats)



def tesk(args):
    transformers.logging.set_verbosity_error()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_no

    def get_seg(text):
        max_len = 500
        max_word_count = 300
        seg = Segmentation(user_dict=None)

        text = text
        text = text[:min(len(text), max_len)]
        pattern = ';|???| '

        word_segments = seg.segment(text=text)['words_no_filter']
        # print(text)
        # print(word_segments)
        # print(keyword_list)
        # print(keyword_list_seg)

        word_li = []
        for li in word_segments:
            word_li.extend(li)

        word_segments_len = 0

        # word_dict={'':0}
        # word_set = set(word_li)
        # for w in word_set:
        #     if w not in word_dict:
        #         word_dict[w]= len(word_dict)
        # print(word_dict)

        word_list_len = len(word_li)
        res = ';'.join(word_li)
        # print(res)
        return res

    def get_mark(text, keywords):
        max_len = 500
        max_word_count = 300
        seg = Segmentation(user_dict=None)

        text = text
        text = text[:min(len(text), max_len)]
        pattern = ';|???| '
        keywords = keywords
        res = re.split(pattern, keywords)
        keyword_list = [r.strip() for r in res]

        word_segments = seg.segment(text=text)['words_no_filter']
        keyword_list_seg = [seg.segment(text=w)['words_no_filter'][0] for w in keyword_list]

        word_li = []
        for li in word_segments:
            word_li.extend(li)

        word_list_len = len(word_li)

        label = [0] * word_list_len  # torch.zeros(max_word_count, dtype=torch.long)
        for ws in keyword_list_seg:
            leng = len(ws)
            for i in range(word_list_len - leng + 1):
                if ws == word_li[i:i + leng]:
                    if leng == 1:
                        label[i] = tag_map['S-KEY']
                    if leng == 2:
                        label[i] = tag_map['B-KEY']
                        label[i + 1] = tag_map['E-KEY']
                    if leng > 2:
                        label[i] = tag_map['B-KEY']
                        label[i + leng - 1] = tag_map['E-KEY']
                        label[i + 1:i + leng - 1] = [tag_map['I-KEY'] for k in range(i + 1, i + leng - 1)]

        mark_label = [str(l) for l in label]
        res = ';'.join(mark_label)
        # print(res)
        return res

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = args.device

    args.train_file = get_real_dir(args.train_file)
    args.model_out = get_real_dir(args.model_out)
    args.pretrained = get_real_dir(args.pretrained)
    args.user_dict = None
    args.simcse_model = get_real_dir(args.simcse_model)
    args.markkpe_model = get_real_dir(args.markkpe_model)

    tokenizer = BertTokenizer.from_pretrained(args.pretrained)

    model = Model(args=args).to(args.device)
    model.load_state_dict(torch.load(args.markkpe_model))

    text = '???????????????????????????????????????????????????????????????????????????"????????????",???????????????????????????????????????,?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????,??????????????????????????????????????????,??????????????????????????????,????????????????????????????????????????????????????????????,??????????????????????????????????????????????????????????????????????????????????????????????????????????????????,????????????????????????????????????????????????????????????????????????,????????????????????????????????????????????????,?????????????????????????????????????????????,?????????????????????????????????????????????????????????????????????,??????????????????????????????????????????,?????????????????????;???????????????????????????????????????,?????????????????????????????????,??????????????????,???????????????????????????????????????????????????????????????'



    print(text)
    text = get_seg(text)
    word_li = text.split(';')
    print(text)
    print(word_li)

    pattern = ';|???| '
    keywords = '????????????;PORTAAL????????????;??????STEM??????'
    res = re.split(pattern, keywords)
    keyword_list = [r.strip() for r in res]

    # labels = torch.zeros(13)
    seg = Segmentation(user_dict=args.user_dict)

    encoding = tokenizer.encode_plus(
        str(text),
        add_special_tokens=True,
        max_length=512,
        return_token_type_ids=True,
        padding='max_length',
        truncation='longest_first',
        return_attention_mask=True,
        return_tensors='pt',
    )
    word_segments = seg.simple_segment(text=text)['words_all_filters']
    # print(tokenizer.decode(encoding['input_ids']))
    # print(word_segments)

    word_segments_len = 0

    # word_dict={'':0}
    # word_set = set(word_li)
    # for w in word_set:
    #     if w not in word_dict:
    #         word_dict[w]= len(word_dict)
    # print(word_dict)
    def padding(li, maxlen=300, pad_token=''):
        le = len(li)
        if len(li) < maxlen:
            li.extend([pad_token for i in range(maxlen - le)])
        else:
            if len(li) > maxlen:
                li = li[0:maxlen]
                le = maxlen
        return li, le

    word_segments, word_segments_len = padding(word_li, maxlen=300)

    def exit(item, list):
        if item in list:
            return 1
        else:
            return 0

    # kpe_label = [exit(word, keyword_list) for word in word_segments]
    # for word in word_segments:
    #     if word in keyword_list:
    #         label = 1
    #     else:
    #         label = 0

    batch = {
        'texts': text,
        'input_ids': encoding['input_ids'],  # ??????tokenizer???????????? 101 ???????????? 102
        'attention_mask': encoding['attention_mask'],  # ???????????????111111000000
        "token_type_ids": encoding['token_type_ids'],  # ????????????11111000000
        # 'labels': labels,
        'word_segments': word_segments,
        'word_segments_len': word_segments_len
    }

    b_input_ids = batch['input_ids'].to(device)
    b_input_mask = batch['attention_mask'].to(device)
    b_token_type_ids = batch['token_type_ids'].to(device)
    # b_labels = batch['labels'].to(device)
    word_segments = batch['word_segments']
    word_segments_len = batch['word_segments_len']
    # kpe_label = batch['kpe_label']

    # ????????????????????????????????????????????? 0????????? pytorch ?????????????????????
    model.zero_grad()

    # ????????????
    # ????????????:
    # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
    # ???????????????????????????????????????????????????????????? ?????????, ????????? loss ??? logits -- ?????????????????????
    score = model.get_contribute(b_input_ids,
                                 attention_mask=b_input_mask,
                                 token_type_ids=b_token_type_ids,
                                 word_segments=word_segments)
    res = model.generateMarkAns(word_li,score)
    print(res)
    # num_list = [1, 8, 2, 3, 10, 4, 5]
    # ordered_list = sorted(range(len(score)), key=lambda k: score[k],reverse=True)
    # key = [batch['word_segments'][ordered_list[i]] for i in range(len(score))]
    # print(key)
    # print(ordered_list)  # [0, 2, 3, 5, 6, 1, 4]

    # print(output)
    # loss = model.loss(l_cls,b_labels) + args.lamda*model.loss(l_mean,b_labels)
    # print(score)
    # word = ''
    # keywords_res = {}
    # for i, w in zip(score, word_li):
    #     if i == 1:
    #         word = w
    #     if i == 2:
    #         word += w
    #     if i == 3:
    #         word += w
    #         if word in keywords_res:
    #             keywords_res[word]+=1
    #         else:
    #             keywords_res[word] = 1
    #     if i == 4:
    #         word = w
    #         if word in keywords_res:
    #             keywords_res[word] += 1
    #         else:
    #             keywords_res[word] = 1
    # print(keywords_res)
    # category_sort = sorted(score.items(), key=lambda x: -x[1])
    # for c in category_sort:
    #     print(c)


def main():
    args = parse_args()
    print(__name__)
    train(args)

def test():
    args = parse_args()
    print(__name__)
    tesk(args)

if __name__ == "__main__":
    log_fmt = "%(asctime)s|%(name)s|%(levelname)s|%(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
