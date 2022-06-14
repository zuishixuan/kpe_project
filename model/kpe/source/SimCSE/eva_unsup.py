# -*- coding: utf-8 -*-
# @Time    : 2021/6/10
# @Author  : kaka

import logging
import time

from sklearn.cluster import DBSCAN

import argparse

from SimCSERetrieval import SimCSERetrieval
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import MinMaxScaler

from sklearn.cluster import AgglomerativeClustering



def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train_file", type=str, default="../../data/cnki_2/data.csv" ,help="train text file")
    parser.add_argument("--pretrained", type=str, default="../../model/chinese_roberta_wwm_ext_pytorch",
                        help="huggingface pretrained model")
    parser.add_argument("--simcse_model", type=str, default="../../model/SimCSE/cnki_2/out/epoch_1-batch_12100")
    parser.add_argument("--vector_out", type=str, default="../../model/SimCSE/cnki_2/out_vec_title_keyword/", help="vector output path")
    parser.add_argument("--num_proc", type=int, default=5, help="dataset process thread num")
    parser.add_argument("--max_length", type=int, default=512, help="sentence max length")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument("--gpu_no", type=str, default='1', help="device")
    args = parser.parse_args()
    return args

def main(args):
    fname = args.train_file
    pretrained = args.pretrained  # huggingface modelhub 下载的预训练模型
    simcse_model = args.simcse_model
    batch_size = args.batch_size
    max_length = args.max_length
    vector_out = args.vector_out
    device = "cuda"

    logging.info("Load model")
    simcse = SimCSERetrieval(fname, pretrained, simcse_model, batch_size, max_length, device, vector_out)

    logging.info("Sentences to vectors")
    simcse.encode_file_csv()

    logging.info("Build faiss index")
    simcse.build_index()
    simcse.index.nprob = 20

    starttime = time.time()


    query_sentence = '准确解读《共产党宣言》阐明的一般原理——评《宣言》导读中提出的若干观点,共产党宣言;无产阶级政党'
    print("\nquery title:{0}".format(query_sentence))
    print("\nsimilar titles:")
    print(u"\n".join(simcse.sim_query_value(query_sentence, topK=10)[1]))
    print(simcse.sim_query_ids(query_sentence, topK=10)[1])
    print(simcse.sim_query_ids(query_sentence, topK=10)[0])

    endtime = time.time()
    print("循环运行时间:%.2f秒" % (endtime - starttime))

def test(args):
    fname = args.train_file
    pretrained = args.pretrained  # huggingface modelhub 下载的预训练模型
    simcse_model = args.simcse_model
    batch_size = args.batch_size
    max_length = args.max_length
    vector_out = args.vector_out
    device = "cuda"

    logging.info("Load model")
    simcse = SimCSERetrieval(fname, pretrained, simcse_model, batch_size, max_length, device, vector_out)

    logging.info("Sentences to vectors")
    simcse.encode_file_csv()

    print(simcse.vecs)
    X = simcse.vecs
    # model = DBSCAN(eps=1.5, min_samples=10, metric='euclidean', algorithm='auto')

    # model.fit(X)
    # plt.figure()
    # plt.scatter(X[:, 0], X[:, 1], c=model.labels_)
    # plt.show()
    model = AgglomerativeClustering(n_clusters=10, affinity='euclidean', memory=None, connectivity=None,
                                    compute_full_tree='auto', linkage='ward')

    min_max_scalar = MinMaxScaler()
    data_scalar = min_max_scalar.fit_transform(X)
    model.fit(X)

    from scipy.cluster.hierarchy import linkage, dendrogram
    plt.figure(figsize=(20, 6))
    Z = linkage(data_scalar, method='ward', metric='euclidean')
    p = dendrogram(Z, 0)
    plt.show()



if __name__ == "__main__":
    log_fmt = "%(asctime)s|%(name)s|%(levelname)s|%(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    args = parse_args()
    test(args)

