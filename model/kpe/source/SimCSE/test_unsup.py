# -*- coding: utf-8 -*-
# @Time    : 2021/6/10
# @Author  : kaka

import logging
from SimCSERetrieval import SimCSERetrieval


def main():
    fname = "./data/news_title.txt"
    pretrained = "./model/chinese-bert-wwm-ext"  # huggingface modelhub 下载的预训练模型
    simcse_model = "./model/out/epoch_1-batch_2800-loss_0.005823"
    batch_size = 64
    max_length = 100
    device = "cuda"

    logging.info("Load model")
    simcse = SimCSERetrieval(fname, pretrained, simcse_model, batch_size, max_length, device)

    logging.info("Sentences to vectors")
    simcse.encode_file()

    logging.info("Build faiss index")
    simcse.build_index(n_list=1024)
    simcse.index.nprob = 20

    query_sentence = "辽宁省1km水稻光合资源利用率数据集"
    print("\nquery title:{0}".format(query_sentence))
    print("\nsimilar titles:")
    print(u"\n".join(simcse.sim_query(query_sentence, topK=10)))


if __name__ == "__main__":
    log_fmt = "%(asctime)s|%(name)s|%(levelname)s|%(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()

