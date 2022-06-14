import argparse
import datetime
import re
import time

import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertTokenizer

# import TextRank4Keyword
from .constant import *
from .Segmentation import Segmentation
from .JointKpe.JointMarkScore2AttenKpe import JointMarkScore2AttenKpe
from .JointKpe.JMS2AKper import JMS2AKper
from .TextRank4Keyword import TextRank4Keyword

seg = Segmentation(user_dict=keyword_dict_dir)
#seg = Segmentation(user_dict=None)


def parse_args(Model=None):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--file_dir", type=str, default=Valid_dataset_file_dir[Valid_dataset_na])
    parser.add_argument("--pretrained", type=str, default=Pretrained_model_dir[Pretrained_na])
    parser.add_argument("--user_dict", type=str, default='../data/cnki_2/keywords_dict.txt')
    parser.add_argument("--pretrained_model_path", type=str, default=Model_file_read_dir[Pretrained_na])
    parser.add_argument("--model_out", type=str, default=Model_file_out_dir[Model_na],
                        help="model output path")
    parser.add_argument("--model_dir", type=str, default=Model_file_read_dir[Model_na])
    parser.add_argument("--valid_data_num", type=int, default=valid_data_num)
    parser.add_argument("--model_name", type=str, default=Model_na)
    parser.add_argument("--device", type=str, default=device, help="device")
    parser.add_argument("--gpu_no", type=str, default='0,1', help="device")
    parser.add_argument("--host", type=str, default='0.0.0.0')
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--threaded", action='store_true')
    parser.add_argument("runserver", type=bool, default=True)
    if Model:
        parser = Model.add_model_specific_args(parser)
    args = parser.parse_args()
    return args


def get_valid_data(args, shuffle=None):
    df = pd.read_csv(args.file_dir)
    if args.valid_data_num != None:
        if shuffle:
            df = df.sample(args.valid_data_num, random_state=random_seed)
        else:
            df = df.head(args.valid_data_num)
    return df


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # 四舍五入到最近的秒
    elapsed_rounded = int(round((elapsed)))

    # 格式化为 hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def get_f1(dict):
    if dict['Acc'] + dict['Rec'] != 0:
        dict['F1'] = 2 * dict['Acc'] * dict['Rec'] / (dict['Acc'] + dict['Rec'])
    else:
        dict['F1'] = 0


def eval():
    args = None
    Model = None
    model = None

    if Model_na == Model_name.JointMarkScore2AttenKpe.value:
        Model = JMS2AKper
        args = parse_args(Model=Model)
        model = Model(args=args, seg=seg)

    if Model_na == Model_name.TextRank.value:
        Model = TextRank4Keyword
        args = parse_args(Model=Model)
        user_dict = args.user_dict
        model = Model(simcse_model_path=args.pretrained_model_path, MODEL_DIR=args.pretrained, user_dict=None,
                      algo_type='basic')

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_no
    df = get_valid_data(args=args, shuffle=shuffle)

    # 存储训练和评估的 loss、准确率、训练时长等统计指标,
    training_stats = []
    # 统计单次 validate 的训练时间
    t0 = time.time()

    for topk in range(1, args.max_kpe_num + 1):

        print('Evaluation...  TopK {}'.format(topk))
        res_batch = {
            'Acc': 0,
            'Rec': 0,
            'F1': 0,
            'Hit': 0
        }
        res_all = {
            'Acc': 0,
            'Rec': 0,
            'F1': 0,
            'Hit': 0
        }
        j = 0
        i = 0
        for item in tqdm(df.itertuples()):
            text = item.text
            keyword = item.keyword
            pattern = r',|;|；| '
            res = re.split(pattern, keyword)
            keyword_list = [r.strip() for r in res]

            j += 1
            if len(keyword_list) == 0:
                continue
            keyword_kpe = model.kpe(text=text, max_num=topk)
            if len(keyword_kpe) == 0:
                continue
            if len(keyword_kpe) < topk and skip:
                continue
            i += 1
            # print(keyword_kpe)

            label_num = len(keyword_list)
            kpe_num = min(topk, len(keyword_kpe))
            # hit_kpe_list = keyword_kpe[:min(label_num, kpe_num)]
            kpe_list = keyword_kpe[:kpe_num]
            correct = [r for r in kpe_list if r in keyword_list]
            cor_num = len(correct)
            acc = cor_num / kpe_num
            rec = cor_num / label_num

            # correct_hit = [r for r in hit_kpe_list if r in keyword_list]
            # hit = len(correct_hit) / label_num
            hit = 0

            res_cur = {
                'Acc': acc,
                'Rec': rec,
                'F1': 0,
                'Hit': hit
            }
            for k in res_cur:
                res_batch[k] += res_cur[k]
                res_all[k] += res_cur[k]

            # 每经过40次迭代，就输出进度信息
            if i % 40 == 0 and not i == 0:
                elapsed = format_time(time.time() - t0)
                for k in res_batch:
                    res_batch[k] = res_batch[k] / 40
                get_f1(res_batch)
                print(
                    '  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.  Acc: {:.2f}  Rec {:.2f}  F1 {:.2f}  Skip {:}'.format(
                        i, len(df),
                        elapsed, res_batch['Acc'],
                        res_batch['Rec'],
                        res_batch['F1'],
                        j - i))
                for k in res_batch:
                    res_batch[k] = 0

        # 平均训练误差
        for k in res_all:
            res_all[k] = res_all[k] / i
        get_f1(res_all)

        # 单次时长
        eval_time = format_time(time.time() - t0)

        for k in res_all:
            print("  Validation Info {0}: {1:.2f}".format(k, res_all[k]))
        print("  Validation took: {:}".format(res_all))

        # 记录本次 epoch 的所有统计信息
        info = {
            'topK': topk,
            'data_size': len(df),
            'Eval Time': eval_time,
        }
        info.update(res_all)
        info.update({'skip_num': j - i})
        training_stats.append(
            info
        )

        # 保留 2 位小数
        pd.set_option('precision', 2)
        # 加载训练统计到 DataFrame 中
        df_stats = pd.DataFrame(data=training_stats)
        # 使用 topK 值作为每行的索引
        df_stats = df_stats.set_index('topK')

    # 展示表格数据
    file_name = format_save_log_file_name()
    print('-' * 5, file_name, '-' * 30)
    print(df_stats)
    create_dir(args.model_out)
    df_stats.to_csv(args.model_out + '/' + file_name)


def eval_kpe_file():
    args = None
    Model = None
    model = None

    if Model_na == Model_name.JointMarkScore2AttenKpe.value:
        Model = JMS2AKper
        args = parse_args(Model=Model)
        model = Model(args=args, seg=seg)

    # if Model_na == Model_name.TextRank.value:
    #     Model = TextRank4Keyword
    #     args = parse_args(Model=Model)
    #     user_dict = args.user_dict
    #     model = Model(simcse_model_path=args.pretrained_model_path, MODEL_DIR=args.pretrained, user_dict=None,
    #                   algo_type='basic')

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_no
    df = get_valid_data(args=args, shuffle=shuffle)

    # 存储训练和评估的 loss、准确率、训练时长等统计指标,
    training_stats = []
    # 统计单次 validate 的训练时间
    t0 = time.time()

    res_batch = {
        'Acc': 0,
        'Rec': 0,
        'F1': 0,
        'Num': 0
    }
    res_all = {
        'Acc': 0,
        'Rec': 0,
        'F1': 0,
        'Num': 0
    }

    i = 0
    for item in tqdm(df.itertuples()):
        text = item.text
        keyword = item.keyword
        pattern = r',|;|；| '
        res = re.split(pattern, keyword)
        keyword_list = [r.strip() for r in res]
        i += 1

        keyword_kpe = model.kpe(text=text, max_num=None)
        res_kpe = {
            'text': text,
            'label': keyword,
            'kpe': ';'.join(keyword_kpe)
        }

        # print(keyword_kpe)

        label_num = len(keyword_list)
        kpe_num = len(keyword_kpe)
        kpe_list = keyword_kpe[:kpe_num]
        correct = [r for r in kpe_list if r in keyword_list]
        cor_num = len(correct)
        if kpe_num != 0 and label_num != 0:
            acc = cor_num / kpe_num
            rec = cor_num / label_num
        else:
            acc = 0
            rec = 0

        hit = 0

        res_cur = {
            'Acc': acc,
            'Rec': rec,
            'F1': 0,
            'Num': kpe_num
        }
        get_f1(res_cur)

        for k in res_cur:
            res_batch[k] += res_cur[k]
            res_all[k] += res_cur[k]

        # 记录本次的所有统计信息
        info = {
            'id': i
        }
        info.update(res_kpe)
        info.update(res_cur)
        training_stats.append(info)

        # 每经过40次迭代，就输出进度信息
        if i % 40 == 0 and not i == 0:
            elapsed = format_time(time.time() - t0)
            for k in res_batch:
                res_batch[k] = res_batch[k] / 40
            get_f1(res_batch)
            print(
                '  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.  Acc: {:.2f}  Rec {:.2f}  F1 {:.2f}  Num{:.2f} '.format(
                    i, len(df),
                    elapsed, res_batch['Acc'],
                    res_batch['Rec'],
                    res_batch['F1'],
                    res_batch['Num']))
            info = {
                'id': i
            }
            info.update(res_batch)
            training_stats.append(info)
            for k in res_batch:
                res_batch[k] = 0

    # 平均训练误差
    for k in res_all:
        res_all[k] = res_all[k] / i
    get_f1(res_all)

    # 单次时长
    eval_time = format_time(time.time() - t0)

    for k in res_all:
        print("  Validation Info {0}: {1:.2f}".format(k, res_all[k]))
    print("  Validation took: {:}".format(res_all))

    # 记录本次 epoch 的所有统计信息
    info = {
        'id': i
    }
    info.update(res_all)
    training_stats.append(info)

    # 保留 2 位小数
    pd.set_option('precision', 2)
    # 加载训练统计到 DataFrame 中
    df_stats = pd.DataFrame(data=training_stats)
    # 使用 topK 值作为每行的索引
    df_stats = df_stats.set_index('id')

    # 展示表格数据
    file_name = format_save_log_file_name('_kpe_res.csv')
    print('-' * 5, file_name, '-' * 30)
    print(df_stats)
    create_dir(args.model_out)
    df_stats.to_csv(args.model_out + '/' + file_name)


def format_save_log_file_name(suffix='_eval_log.csv'):
    if Model_na == Model_name.TextRank.value:
        file_name = '{}_{}_{}'.format(Model_na, Valid_dataset_na, valid_data_num)
    else:
        file_name = '{}_{}_{}_{}_{}'.format(Model_na, Valid_dataset_na, out, epoch, valid_data_num)
        if shuffle:
            file_name = file_name + '_shuffle_{}'.format(random_seed)
        if skip:
            file_name = file_name + '_skip'
    return file_name + suffix


def test():
    args = None
    Model = None
    model = None

    if Model_na == Model_name.JointMarkScore2AttenKpe.value:
        Model = JMS2AKper
        args = parse_args(Model=Model)
        model = Model(args=args, seg=seg)

    # if Model_na == Model_name.TextRank.value:
    #     Model = TextRank4Keyword
    #     args = parse_args(Model=Model)
    #     user_dict = args.user_dict
    #     model = Model(simcse_model_path=args.pretrained_model_path, MODEL_DIR=args.pretrained, user_dict=None)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_no
    # df = get_valid_data(args=args)
    data_list = test_data_list
    if not isinstance(data_list, list):
        data_list = [data_list]
    for data in data_list:
        text = data['text']
        keyword_label = data['keyword']
        print(text)
        pattern = ';|；| '
        res = re.split(pattern, keyword_label)
        res = [r.strip() for r in res]
        print(res)
        keyword_kpe = model.kpe(text=text, has_score=True, supplement=False)
        print("basic:")
        for key in keyword_kpe:
            print(key)
        keyword_kpe = model.kpe(text=text, has_score=True, min_num=5, supplement=True)
        print('suppl:')
        for key in keyword_kpe:
            print(key)
        keyword_kpe = model.kpe(text=text, has_score=True, min_num=5, supplement=True,merge=True)
        print('merge:')
        for key in keyword_kpe:
            print(key)
        keyword_kpe = model.kpe2(text=text, has_score=True, min_num=5, supplement=True)
        print('rec:')
        for key in keyword_kpe:
            print(key)


def getmodel_metadata():
    Model_na = Model_name.JointMarkScore2AttenKpe.value
    Dataset_na = Dataset_name.combine_2.value
    out = 'out'
    epoch = 'epoch_5'
    device = "cuda:1"

    args = None
    Model = None
    model = None
    print("-"*20+'get model'+"-"*20)

    if Model_na == Model_name.JointMarkScore2AttenKpe.value:
        Model = JMS2AKper
        args = parse_args(Model=Model)
        model = Model(args=args, seg=seg)

    if Model_na == Model_name.TextRank.value:
        Model = TextRank4Keyword
        args = parse_args(Model=Model)
        user_dict = args.user_dict
        model = Model(simcse_model_path=args.pretrained_model_path, MODEL_DIR=args.pretrained, user_dict=None)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_no
    return model


def getmodel_cnki():
    Model_na = Model_name.JointMarkScore2AttenKpe.value
    Dataset_na = Dataset_name.cnki_2.value
    out = 'out7'
    epoch = 'epoch_6'
    device = "cuda:1"

    args = None
    Model = None
    model = None
    print("-"*20+'get model'+"-"*20)

    if Model_na == Model_name.JointMarkScore2AttenKpe.value:
        Model = JMS2AKper
        args = parse_args(Model=Model)
        model = Model(args=args, seg=seg)

    if Model_na == Model_name.TextRank.value:
        Model = TextRank4Keyword
        args = parse_args(Model=Model)
        user_dict = args.user_dict
        model = Model(simcse_model_path=args.pretrained_model_path, MODEL_DIR=args.pretrained, user_dict=None)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_no
    return model

# def kpe(text):
#     # text = codecs.open('./doc/02.txt', 'r', 'utf-8').read()
#     # text = "世界的美好。世界美国英国。 世界和平。"
#
#     tr4w = TextRank4Keyword.TextRank4Keyword(simcse_model_path="./model/SimCSE/cnki/out/epoch_1-batch_500")
#     tr4w.analyze(text=text, lower=True, window=3, pagerank_config={'alpha': 0.85})
#
#     print('关键词前五：')
#     for item in tr4w.get_keywords(5, word_min_len=2):
#         print(item.word, item.weight, type(item.word))
#
#     print('关键词短语：')
#
#     for phrase in tr4w.get_keyphrases(keywords_num=5, min_occur_num=0):
#         print(phrase, type(phrase))


def main():
    eval()


if __name__ == "__main__":
    main()
