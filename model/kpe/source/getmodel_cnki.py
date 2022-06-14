import argparse
import datetime
import re
import time

import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertTokenizer

# import TextRank4Keyword
from .constant_cnki import *
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



def getmodel_cnki():
    args = None
    Model = None
    model = None
    print("-"*20+'get model_cnki'+"-"*20)

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



def main():
    print("a")


if __name__ == "__main__":
    main()
