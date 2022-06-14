import argparse
import datetime
import logging
import os
import time
from pathlib import Path
from .JointKpe.ScoreKpe import ScoreKpe
from .JointKpe.data_process_scorekpe import *
from transformers import Trainer, TrainingArguments, BertTokenizer, BertModel, BertPreTrainedModel, BertConfig
from torch.utils.data import Dataset, DataLoader, RandomSampler, random_split, SequentialSampler
from transformers import get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm
from .Segmentation import Segmentation

import torch
import transformers
from .constant import get_real_dir


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train_file", type=str, default="../data/cnki_2/data_filter.csv" ,help="train text file")
    parser.add_argument("--pretrained", type=str, default="../model/chinese_roberta_wwm_ext_pytorch",
                        help="huggingface pretrained model")
    parser.add_argument("--simcse_model", type=str, default="../model/SimCSE/cnki_2/out/epoch_1-batch_12100")
    parser.add_argument("--model_out", type=str, default="../model/ScoreKpe/cnki_2/out", help="model output path")
    parser.add_argument("--scorekpe_model", type=str, default="../model/ScoreKpe/cnki_2/out/epoch_3")
    parser.add_argument("--user_dict", type=str, default='../data/cnki_2/keywords_dict.txt')
    parser.add_argument("--num_proc", type=int, default=5, help="dataset process thread num")
    parser.add_argument("--max_length", type=int, default=500, help="sentence max length")
    parser.add_argument("--max_word_count", type=int, default=300, help="woed max num")
    parser.add_argument("--atten_hidden_size", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--epochs", type=int, default=4, help="epochs")
    parser.add_argument("--tao", type=float, default=0.05, help="temperature")
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument("--display_interval", type=int, default=50, help="display interval")
    parser.add_argument("--save_interval", type=int, default=100, help="save interval")
    parser.add_argument("--pool_type", type=str, default="cls", help="pool_type")
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="dropout_rate")
    parser.add_argument("--gpu_no", type=str, default='0', help="device")
    parser = ScoreKpe.add_model_specific_args(parser)

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
    return sum(row.all().int().item() for row in (preds.ge(0.5) == labels))/preds.shape[0]


def train(args):
    transformers.logging.set_verbosity_error()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_no

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = args.device

    args.train_file = get_real_dir(args.train_file)
    args.model_out = get_real_dir(args.model_out)
    args.pretrained = get_real_dir(args.pretrained)
    args.user_dict = get_real_dir(args.user_dict)
    args.simcse_model = get_real_dir(args.simcse_model)

    df = read_data_for_bert(args.train_file)

    tokenizer = BertTokenizer.from_pretrained(args.pretrained)

    dataset = InputDataSet(df, tokenizer=tokenizer, max_len=args.max_length,user_dict=args.user_dict)

    # 计算训练集和验证集大小
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    # 按照数据大小随机拆分训练集和测试集
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))

    # 在 fine-tune 的训练中，BERT 作者建议小批量大小设为 16 或 32
    batch_size = args.batch_size
    train_dataloader = DataLoader(
        train_dataset,  # 训练样本
        sampler=RandomSampler(train_dataset),  # 随机小批量
        batch_size=batch_size  # 以小批量进行训练
    )

    # 验证集不需要随机化，这里顺序读取就好
    validation_dataloader = DataLoader(
        val_dataset,  # 验证样本
        sampler=SequentialSampler(val_dataset),  # 顺序选取小批量
        batch_size=batch_size
    )
    batch = next(iter(train_dataloader))

    # 加载 Bert, 预训练 BERT 模型 + 顶层的线性分类层
    #model = SENet(args=args).to(args.device)
    model = ScoreKpe(args=args).to(args.device)

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

    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # 统计单次 epoch 的训练时间
        t0 = time.time()

        # 重置每次 epoch 的训练总 loss
        total_train_loss = 0
        loss_batch = 0

        # 将模型设置为训练模式。这里并不是调用训练接口的意思
        # dropout、batchnorm 层在训练和测试模式下的表现是不同的 (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # 训练集小批量迭代
        for step, batch in tqdm(enumerate(train_dataloader)):

            # 每经过40次迭代，就输出进度信息
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                loss_batch= loss_batch/40
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.  Loss: {:}'.format(step, len(train_dataloader), elapsed, loss_batch))
                loss_batch = 0

            # 准备输入数据，并将其拷贝到 gpu 中
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_token_type_ids = batch['token_type_ids'].to(device)
            #b_labels = batch['labels'].to(device)
            word_segments = batch['word_segments']
            word_segments_len = batch['word_segments_len']
            kpe_label = batch['kpe_label']

            # 每次计算梯度前，都需要将梯度清 0，因为 pytorch 的梯度是累加的
            model.zero_grad()

            # 前向传播
            # 文档参见:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # 该函数会根据不同的参数，会返回不同的值。 本例中, 会返回 loss 和 logits -- 模型的预测结果
            loss = model(b_input_ids,
                           attention_mask=b_input_mask,
                           token_type_ids=b_token_type_ids,
                           word_segments=word_segments,
                            kpe_label = kpe_label)
            # print(output)
            #loss = model.loss(l_cls,b_labels) + args.lamda*model.loss(l_mean,b_labels)

            # 累加 loss
            loss_batch += loss.item()
            total_train_loss += loss.item()

            # 反向传播
            loss.backward()

            # 梯度裁剪，避免出现梯度爆炸情况
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # 更新参数
            optimizer.step()

            # 更新学习率
            scheduler.step()

        # 平均训练误差
        avg_train_loss = total_train_loss / len(train_dataloader)

        # 单次 epoch 的训练时长
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
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
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in tqdm(validation_dataloader):
            # 将输入数据加载到 gpu 中
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_token_type_ids = batch['token_type_ids'].to(device)
            #b_labels = batch['labels'].to(device)
            word_segments = batch['word_segments']
            word_segments_len = batch['word_segments_len']
            kpe_label = batch['kpe_label']

            # 评估的时候不需要更新参数、计算梯度
            with torch.no_grad():
                loss = model(b_input_ids,
                           attention_mask=b_input_mask,
                           token_type_ids=b_token_type_ids,
                           word_segments=word_segments,
                           word_segments_len=word_segments_len,
                            kpe_label = kpe_label)
            # print(output)
            # loss = model.loss(l_cls, b_labels) + args.lamda * model.loss(l_mean, b_labels)
            # logits = l_cls

            # 累加 loss
            total_eval_loss += loss.item()

            # 将预测结果和 labels 加载到 cpu 中计算
            # logits = logits
            # label_ids = b_labels

            # 计算准确率
            # total_eval_accuracy += flat_accuracy(logits, label_ids)

        # 打印本次 epoch 的准确率
        # avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        # print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        # 统计本次 epoch 的 loss
        avg_val_loss = total_eval_loss / len(validation_dataloader)

        # 统计本次评估的时长
        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        # 记录本次 epoch 的所有统计信息
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
        # 目录不存在则创建
        model_out = Path(args.model_out)
        if not model_out.exists():
            os.mkdir(model_out)

        print("Saving model to %s" % model_out)
        torch.save(model.state_dict(),
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

    # 目录不存在则创建
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    #
    # print("Saving model to %s" % output_dir)

    # 使用 `save_pretrained()` 来保存已训练的模型，模型配置和分词器
    # 它们后续可以通过 `from_pretrained()` 加载
    # model_to_save = model.module if hasattr(model, 'module') else model  # 考虑到分布式/并行（distributed/parallel）训练
    # model_to_save.save_pretrained(output_dir)
    # tokenizer.save_pretrained(output_dir)

    # Good practice: save your training arguments together with the trained model
    # torch.save(args, os.path.join(output_dir, 'training_args.bin'))




def tesk(args):
    transformers.logging.set_verbosity_error()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_no

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = args.device

    args.train_file = get_real_dir(args.train_file)
    args.model_out = get_real_dir(args.model_out)
    args.pretrained = get_real_dir(args.pretrained)
    args.user_dict = get_real_dir(args.user_dict)
    args.simcse_model = get_real_dir(args.simcse_model)
    args.scorekpe_model = get_real_dir(args.scorekpe_model)

    tokenizer = BertTokenizer.from_pretrained(args.pretrained)


    model = ScoreKpe(args=args).to(args.device)
    model.load_state_dict(torch.load(args.scorekpe_model))

    text = '人工智能及机器学习技术的快速发展,为负荷预测问题提供了崭新的解决思路。该文结合人工智能的前沿理论研究,提出一种基于多模型融合Stacking集成学习方式的负荷预测方法。考虑不同算法的数据观测与训练原理差异,充分发挥各个模型优势,构建多个机器学习算法嵌入的Stacking集成学习的负荷预测模型,模型的基学习器包含XGBoost树集成算法和长短记忆网络算法。算例使用ENTSO中瑞士负荷数据对算法有效性进行了验证。预测结果表明,XGBoost、梯度决策树、随机森林模型能够通过自身模型的增益情况对输入数据的特征贡献度进行量化分析;Stacking中各个基学习器的学习能力越强,关联程度越低,模型预测效果越好;与传统单模型预测相比,基于多模型融合的Stacking集成学习方式的负荷预测方法有着较高的预测精度。'

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
    #b_labels = batch['labels'].to(device)
    word_segments = batch['word_segments']
    word_segments_len = batch['word_segments_len']
    kpe_label = batch['kpe_label']

    # 每次计算梯度前，都需要将梯度清 0，因为 pytorch 的梯度是累加的
    model.zero_grad()

    # 前向传播
    # 文档参见:
    # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
    # 该函数会根据不同的参数，会返回不同的值。 本例中, 会返回 loss 和 logits -- 模型的预测结果
    score = model.get_contribute(b_input_ids,
                   attention_mask=b_input_mask,
                   token_type_ids=b_token_type_ids,
                   word_segments=word_segments,
                   word_segments_len=word_segments_len)
    #num_list = [1, 8, 2, 3, 10, 4, 5]
    # ordered_list = sorted(range(len(score)), key=lambda k: score[k],reverse=True)
    # key = [batch['word_segments'][ordered_list[i]] for i in range(len(score))]
    # print(key)
    # print(ordered_list)  # [0, 2, 3, 5, 6, 1, 4]

    # print(output)
    #loss = model.loss(l_cls,b_labels) + args.lamda*model.loss(l_mean,b_labels)
    category_sort = sorted(score.items(), key=lambda x: -x[1])
    for c in category_sort:
        print(c)


def main():
    args = parse_args()
    print(__name__)
    #train(args)
    train(args)
if __name__ == "__main__":
    log_fmt = "%(asctime)s|%(name)s|%(levelname)s|%(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()