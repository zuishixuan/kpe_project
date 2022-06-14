import transformers
from transformers import BertForSequenceClassification, AdamW, BertConfig

from transformers import get_linear_schedule_with_warmup

from data_process import read_data, read_data_for_bert, InputDataSet

from transformers import Trainer, TrainingArguments, BertTokenizer, BertModel, BertPreTrainedModel, BertConfig
from torch.utils.data import Dataset, DataLoader, RandomSampler, random_split, SequentialSampler

from constant import *

import torch

if __name__ == '__main__':

    transformers.logging.set_verbosity_error()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = read_data_for_bert()

    tokenizer = BertTokenizer.from_pretrained(model_dir)

    dataset = InputDataSet(df, tokenizer=tokenizer, max_len=300)

    # 计算训练集和验证集大小
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    # 按照数据大小随机拆分训练集和测试集
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))

    # 在 fine-tune 的训练中，BERT 作者建议小批量大小设为 16 或 32
    batch_size = 16
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

    # 加载 BertForSequenceClassification, 预训练 BERT 模型 + 顶层的线性分类层
    model = BertForSequenceClassification.from_pretrained(
        model_dir,  # 小写的 12 层预训练模型
        num_labels=2,  # 分类数 --2 表示二分类
        # 你可以改变这个数字，用于多分类任务
        output_attentions=False,  # 模型是否返回 attentions weights.
        output_hidden_states=False,  # 模型是否返回所有隐层状态.
    )

    # 在 gpu 中运行该模型
    model.cuda()

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
    optimizer = AdamW(model.parameters(),
                      lr=2e-5,  # args.learning_rate - default is 5e-5
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8
                      )

    # 训练 epochs。 BERT 作者建议在 2 和 4 之间，设大了容易过拟合
    epochs = 2

    # 总的训练样本数
    total_steps = len(train_dataloader) * epochs

    # 创建学习率调度器
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)
    import numpy as np


    # 根据预测结果和标签数据来计算准确率
    def flat_accuracy(preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)


    import time
    import datetime


    def format_time(elapsed):
        '''
        Takes a time in seconds and returns a string hh:mm:ss
        '''
        # 四舍五入到最近的秒
        elapsed_rounded = int(round((elapsed)))

        # 格式化为 hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))


    import random
    import numpy as np

    # 以下训练代码是基于 `run_glue.py` 脚本:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

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

        # 将模型设置为训练模式。这里并不是调用训练接口的意思
        # dropout、batchnorm 层在训练和测试模式下的表现是不同的 (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # 训练集小批量迭代
        for step, batch in enumerate(train_dataloader):

            # 每经过40次迭代，就输出进度信息
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # 准备输入数据，并将其拷贝到 gpu 中
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['labels'].to(device)

            # 每次计算梯度前，都需要将梯度清 0，因为 pytorch 的梯度是累加的
            model.zero_grad()

            # 前向传播
            # 文档参见:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # 该函数会根据不同的参数，会返回不同的值。 本例中, 会返回 loss 和 logits -- 模型的预测结果
            output = model(b_input_ids,
                           token_type_ids=None,
                           attention_mask=b_input_mask,
                           labels=b_labels)
            # print(output)
            loss = output['loss']
            logits = output['logits']

            # 累加 loss
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
        for batch in validation_dataloader:
            # 将输入数据加载到 gpu 中
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['labels'].to(device)

            # 评估的时候不需要更新参数、计算梯度
            with torch.no_grad():
                output = model(b_input_ids,
                               token_type_ids=None,
                               attention_mask=b_input_mask,
                               labels=b_labels)

            loss = output['loss']
            logits = output['logits']

            # 累加 loss
            total_eval_loss += loss.item()

            # 将预测结果和 labels 加载到 cpu 中计算
            logits = logits
            label_ids = b_labels

            # 计算准确率
            total_eval_accuracy += flat_accuracy(logits, label_ids)

        # 打印本次 epoch 的准确率
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

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
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

    import pandas as pd

    # 保留 2 位小数
    pd.set_option('precision', 2)

    # 加载训练统计到 DataFrame 中
    df_stats = pd.DataFrame(data=training_stats)

    # 使用 epoch 值作为每行的索引
    df_stats = df_stats.set_index('epoch')

    # 展示表格数据
    print(df_stats)

    import os

    # 目录不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Saving model to %s" % output_dir)

    # 使用 `save_pretrained()` 来保存已训练的模型，模型配置和分词器
    # 它们后续可以通过 `from_pretrained()` 加载
    model_to_save = model.module if hasattr(model, 'module') else model  # 考虑到分布式/并行（distributed/parallel）训练
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Good practice: save your training arguments together with the trained model
    # torch.save(args, os.path.join(output_dir, 'training_args.bin'))
