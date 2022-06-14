import argparse
import datetime
import logging
import os
import time
from enum import Enum
from pathlib import Path
from .JointKpe.JointMarkScore2AttenKpe import JointMarkScore2AttenKpe as Model
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
from .constant import *


seg = Segmentation(user_dict=None)

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--file_dir", type=str, default=Train_dataset_file_dir[Dataset_na])
    parser.add_argument("--pretrained", type=str, default=Pretrained_model_dir[Pretrained_na])
    parser.add_argument("--pretrained_model_path", type=str, default=Model_file_read_dir[Pretrained_na])
    parser.add_argument("--model_out", type=str, default=Model_file_out_dir[Model_na],
                        help="model output path")
    parser.add_argument("--model_dir", type=str, default=Model_file_read_dir[Model_na])

    parser.add_argument("--user_dict", type=str, default='../data/cnki_2/keywords_dict.txt')
    parser.add_argument("--num_proc", type=int, default=5, help="dataset process thread num")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--epochs", type=int, default=epoch_num, help="epochs")
    parser.add_argument("--tao", type=float, default=0.05, help="temperature")
    parser.add_argument("--device", type=str, default=device, help="device")
    parser.add_argument("--display_interval", type=int, default=50, help="display interval")
    parser.add_argument("--save_interval", type=int, default=100, help="save interval")
    parser.add_argument("--pool_type", type=str, default="cls", help="pool_type")
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="dropout_rate")
    parser.add_argument("--gpu_no", type=str, default='0,1', help="device")
    # DDP （1）要使用`torch.distributed`，你需要在你的`main.py(也就是你的主py脚本)`中的主函数中加入一个**参数接口：`--local_rank`**
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser = Model.add_model_specific_args(parser)

    args = parser.parse_args()
    return args


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # 四舍五入到最近的秒
    elapsed_rounded = int(round((elapsed)))

    # 格式化为 hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# 根据预测结果和标签数据来计算准确率
def flat_accuracy(preds, labels):
    # pred_flat = preds.ge(0.5)
    # labels_flat = labels

    return sum(row.all().int().item() for row in (preds.ge(0.5) == labels)) / preds.shape[0]


def train(args):
    transformers.logging.set_verbosity_error()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_no

    # DDP （2）使用 init_process_group 设置GPU 之间通信使用的后端和端口：
    # dist.init_process_group(backend='nccl')
    # torch.cuda.set_device(args.local_rank)

    local_rank = args.local_rank
    if local_rank != -1:
        dist_backend = 'nccl'
        dist.init_process_group(backend=dist_backend)  # 初始化进程组，同时初始化 distributed 包
        args.device = local_rank
        device = local_rank
        torch.cuda.set_device(local_rank)
    else:
        device = args.device
        torch.device(device) if torch.cuda.is_available() else torch.device('cpu')


    args.user_dict = None


    if test_100:
        df = read_data_for_bert(args.file_dir).head(100)
    else:
        df = read_data_for_bert(args.file_dir)

    tokenizer = BertTokenizer.from_pretrained(args.pretrained)

    dataset = InputDataSet(df, tokenizer=tokenizer, max_len=args.max_length, user_dict=args.user_dict)

    # 计算训练集和验证集大小
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size

    # 按照数据大小随机拆分训练集和测试集
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))

    # 在 fine-tune 的训练中，BERT 作者建议小批量大小设为 16 或 32
    batch_size = args.batch_size

    # DDP （3）使用 DistributedSampler 对数据集进行划分：
    if local_rank != -1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, rank=local_rank)
    else:
        train_sampler = None

    train_dataloader = DataLoader(
        train_dataset,  # 训练样本
        # sampler=RandomSampler(train_dataset),  # 随机小批量
        batch_size=batch_size,  # 以小批量进行训练
        sampler=train_sampler
    )

    # 验证集不需要随机化，这里顺序读取就好
    # val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    validation_dataloader = DataLoader(
        val_dataset,  # 验证样本
        # sampler=SequentialSampler(val_dataset),  # 顺序选取小批量
        batch_size=batch_size
    )

    # 加载 Bert, 预训练 BERT 模型 + 顶层的线性分类层
    # DDP （4）模型

    model = Model(args=args).to(device)
    if local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          find_unused_parameters=True)
    if deep_train:
        if local_rank != -1:
            model.module.load_state_dict(torch.load(args.model_dir))
        else:
            model.load_state_dict(torch.load(args.model_dir))
    # 将所有模型参数转换为一个列表

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

    # 批量大小：16, 32
    # 学习率（Adam）：5e-5, 3e-5, 2e-5
    # epochs
    # 的次数：2, 3, 4
    # 我们的选择如下：
    #
    # Batch
    # size: 32（在构建
    # DataLoaders
    # 时设置）
    # Learning
    # rate：2e-5
    # Epochs： 4（我们将看到这个值对于本任务来说有点大了）

    # 我认为 'W' 代表 '权重衰减修复"
    # DDP （5）优化器多加一个module
    if local_rank != -1:
        optimizer = model.module.configure_optimizers()
    else:
        optimizer = model.configure_optimizers()

    # 训练 epochs。 BERT 作者建议在 2 和 4 之间，设大了容易过拟合
    epochs = args.epochs

    # 总的训练样本数
    total_steps = len(train_dataloader) * epochs

    # 创建学习率调度器
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)

    # 设定随机种子值，以确保输出是确定的
    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # 存储训练和评估的 loss、准确率、训练时长等统计指标,
    training_stats = []

    # 统计整个训练时长
    total_t0 = time.time()

    basic = 0
    if deep_train:
        basic = last_epoch_num

    for epoch_i in range(basic, basic + epochs):

        # ========================================
        #               Training
        # ========================================

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # 统计单次 epoch 的训练时间
        t0 = time.time()

        # 重置每次 epoch 的训练总 loss
        loss_batch = {
            'loss': 0,
            'mark_loss': 0,
            'rank_loss': 0
        }
        loss_epoch = {
            'loss': 0,
            'mark_loss': 0,
            'rank_loss': 0
        }
        # 将模型设置为训练模式。这里并不是调用训练接口的意思
        # dropout、batchnorm 层在训练和测试模式下的表现是不同的 (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # 训练集小批量迭代
        for step, batch in tqdm(enumerate(train_dataloader)):

            # 每经过40次迭代，就输出进度信息
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                for k in loss_batch:
                    loss_batch[k] = loss_batch[k] / 40
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.  Loss: {:.2f} mark_loss {:.2f} rank_loss{:.2f}'.format(step, len(train_dataloader),
                                                                                       elapsed, loss_batch['loss'],
                                                                                                loss_batch['mark_loss'],
                                                                                                loss_batch['rank_loss']))
                for k in loss_batch:
                    loss_batch[k] = 0

            # 准备输入数据，并将其拷贝到 gpu 中
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_token_type_ids = batch['token_type_ids'].to(device)
            word_segments = batch['word_segments']
            kpe_label = batch['kpe_label'].to(device)

            # 每次计算梯度前，都需要将梯度清 0，因为 pytorch 的梯度是累加的
            model.zero_grad()

            # 前向传播
            # 文档参见:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # 该函数会根据不同的参数，会返回不同的值。 本例中, 会返回 loss 和 logits -- 模型的预测结果
            loss, mark_loss, rank_loss = model(b_input_ids,
                                               attention_mask=b_input_mask,
                                               token_type_ids=b_token_type_ids,
                                               word_segments=word_segments,
                                               kpe_label=kpe_label)
            # print(output)
            # loss = model.loss(l_cls,b_labels) + args.lamda*model.loss(l_mean,b_labels)

            # 累加 loss
            loss_cur = {
                'loss': loss.item(),
                'mark_loss': mark_loss.item(),
                'rank_loss': rank_loss.item()
            }
            for k in loss_cur:
                loss_batch[k] += loss_cur[k]
                loss_epoch[k] += loss_cur[k]

            # 反向传播
            loss.backward()

            # 梯度裁剪，避免出现梯度爆炸情况
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # 更新参数
            optimizer.step()

            # 更新学习率
            scheduler.step()

        # 平均训练误差
        for k in loss_epoch:
            loss_epoch[k] = loss_epoch[k] / len(train_dataloader)


        # 单次 epoch 的训练时长
        training_time = format_time(time.time() - t0)

        print("")
        for k in loss_epoch:
            print("  Average training {0}: {1:.2f}".format(k, loss_epoch[k]))
        print("  Training epcoh took: {:}".format(training_time))

        # ========================================
        #               Validation
        # ========================================
        # 完成一次 epoch 训练后，就对该模型的性能进行验证

        print("")
        print("Running Validation...")

        t0 = time.time()

        # 设置模型为评估模式
        model.eval()

        # Tracking variables
        loss_valid_epoch = {
            'loss_val': 0,
            'mark_loss_val': 0,
            'rank_loss_val': 0
        }
        total_eval_accuracy = 0

        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in tqdm(validation_dataloader):
            # 将输入数据加载到 gpu 中
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_token_type_ids = batch['token_type_ids'].to(device)
            # b_labels = batch['labels'].to(device)
            word_segments = batch['word_segments']
            kpe_label = batch['kpe_label'].to(device)

            # 评估的时候不需要更新参数、计算梯度
            with torch.no_grad():
                loss, mark_loss, rank_loss = model(b_input_ids,
                                                   attention_mask=b_input_mask,
                                                   token_type_ids=b_token_type_ids,
                                                   word_segments=word_segments,
                                                   kpe_label=kpe_label)
            # print(output)
            # loss = model.loss(l_cls, b_labels) + args.lamda * model.loss(l_mean, b_labels)
            # logits = l_cls

            # 累加 loss
            loss_val_cur = {
                'loss_val': loss.item(),
                'mark_loss_val': mark_loss.item(),
                'rank_loss_val': rank_loss.item()
            }
            for k in loss_val_cur:
                loss_valid_epoch[k] += loss_val_cur[k]

            # 将预测结果和 labels 加载到 cpu 中计算
            # logits = logits
            # label_ids = b_labels

            # 计算准确率
            # total_eval_accuracy += flat_accuracy(logits, label_ids)

        # 打印本次 epoch 的准确率
        # avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        # print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        # 统计本次 epoch 的 loss
        for k in loss_valid_epoch:
            loss_valid_epoch[k] = loss_valid_epoch[k] / len(validation_dataloader)


        # 统计本次评估的时长
        validation_time = format_time(time.time() - t0)
        for k in loss_valid_epoch:
            print("  Validation Loss {0}: {1:.2f}".format(k, loss_valid_epoch[k]))
        print("  Validation took: {:}".format(validation_time))

        # 记录本次 epoch 的所有统计信息
        info ={
                'epoch': epoch_i + 1,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        info.update(loss_epoch)
        info.update(loss_valid_epoch)
        training_stats.append(
            info
        )

        # 目录不存在则创建
        model_out = Path(args.model_out)
        if not model_out.exists():
            os.makedirs(model_out, exist_ok=True)

        # DDP （6）模型保存多加一个module
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

    # 保留 2 位小数
    pd.set_option('precision', 2)

    # 加载训练统计到 DataFrame 中
    df_stats = pd.DataFrame(data=training_stats)

    # 使用 epoch 值作为每行的索引
    df_stats = df_stats.set_index('epoch')

    # 展示表格数据
    print(df_stats)
    save_dir = args.model_out + '/train_log.csv'
    df_stats.to_csv(save_dir)


def tesk(args):
    transformers.logging.set_verbosity_error()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_no

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = args.device

    args.user_dict = None


    tokenizer = BertTokenizer.from_pretrained(args.pretrained)

    model = Model(args=args).to(args.device)
    model.load_state_dict(torch.load(args.model_dir))

    text = '车速里程表标准装置检定装置。车速里程表标准装置检定装置用于车速里程表标准装置的检定或校准。车速里程表标准装置用于检定、校准机动车辆转速表、里程表和车速里程表。广泛应用于汽车、摩托车、机动车、机械行业。我院于2008年建立车速里程表标准装置检定装置计量标准。该标准装置主要由0.01级高精度智能测速系统组成，装置标准不确定度为： Uerl=0.04% k=3；。该项计量标准经历年计量标准考核，以及重复性、稳定性核查，其计量性能稳定可靠，目前用于浙江省内各相关行业的车速里程表标准装置的检定及校准。'

    print(len(text))
    text = text[:min(len(text), args.max_length)]
    print(len(text))
    pattern = ';|；| '
    keywords = '主动学习;PORTAAL评价系统;大学STEM课堂'
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

    word_li = []
    for li in word_segments:
        word_li.extend(li)

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

    kpe_label = [exit(word, keyword_list) for word in word_segments]
    # for word in word_segments:
    #     if word in keyword_list:
    #         label = 1
    #     else:
    #         label = 0

    batch = {
        'texts': text,
        'input_ids': encoding['input_ids'],  # 通过tokenizer做的编码 101 。。。。 102
        'attention_mask': encoding['attention_mask'],  # 注意力编码111111000000
        "token_type_ids": encoding['token_type_ids'],  # 分句编码11111000000
        # 'labels': labels,
        'word_segments': word_segments,
        'word_segments_len': word_segments_len,
        'kpe_label': kpe_label
    }

    b_input_ids = batch['input_ids'].to(device)
    b_input_mask = batch['attention_mask'].to(device)
    b_token_type_ids = batch['token_type_ids'].to(device)
    # b_labels = batch['labels'].to(device)
    word_segments = batch['word_segments']
    word_segments_len = batch['word_segments_len']
    kpe_label = batch['kpe_label']

    # 每次计算梯度前，都需要将梯度清 0，因为 pytorch 的梯度是累加的
    model.zero_grad()

    # 前向传播
    # 文档参见:
    # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
    # 该函数会根据不同的参数，会返回不同的值。 本例中, 会返回 loss 和 logits -- 模型的预测结果
    mark_score, rank_score = model.get_contribute(b_input_ids,
                                                  attention_mask=b_input_mask,
                                                  token_type_ids=b_token_type_ids,
                                                  word_segments=word_segments)

    mark_res = model.generateMarkAns(word_segments, mark_score)
    rank_res = model.generateScoreAns(word_segments, rank_score)
    joint_res = model.generateJointAns(word_segments,mark_score,rank_score,'type_remove')
    print(mark_res)
    for res in rank_res:
        print(res, ' ', rank_res[res])
    print('-'*80)
    for res in joint_res:
        print(res)


def main():
    args = parse_args()
    print(__name__)
    train(args)
    # tesk(args)


def test():
    args = parse_args()
    print(__name__)
    tesk(args)
    # tesk(args)


if __name__ == "__main__":
    log_fmt = "%(asctime)s|%(name)s|%(levelname)s|%(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
