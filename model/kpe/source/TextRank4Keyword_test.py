#-*- encoding:utf-8 -*-
from __future__ import print_function

import TextRank4Keyword

def kpe(text):
    # text = codecs.open('./doc/02.txt', 'r', 'utf-8').read()
    # text = "世界的美好。世界美国英国。 世界和平。"

    tr4w = TextRank4Keyword.TextRank4Keyword()
    tr4w.analyze(text=text,lower=True, window=3, pagerank_config={'alpha':0.85})

    print('关键词前五：')
    for item in tr4w.get_keywords(5, word_min_len=2):
        print(item.word, item.weight, type(item.word))

    print('关键词短语：')

    for phrase in tr4w.get_keyphrases(keywords_num=5, min_occur_num = 0):
        print(phrase, type(phrase))


def get_em(text):
    # text = codecs.open('./doc/02.txt', 'r', 'utf-8').read()
    # text = "世界的美好。世界美国英国。 世界和平。"

    tr4w = TextRank4Keyword.TextRank4Keyword(simcse_model_path="./model/SimCSE/cnki/out/epoch_1-batch_500")
    tr4w.analyze(text=text, lower=True, window=2, pagerank_config={'alpha': 0.85})

    print('关键词前五：')
    for item in tr4w.get_keywords(20, word_min_len=2):
        print(item.word, item.weight, type(item.word))

    print('关键词短语：')

    for phrase in tr4w.get_keyphrases(keywords_num=10, min_occur_num=2):
        print(phrase, type(phrase))
    # k = tr4w.get_weight_matrix()
    # #k = tr4w.get_sim_matrix()
    # print(k[0].sum())
    # # res = tr4w.norm_matrix(k)
    # # print(res)

def main():
    text = '康德哲学视点下人工智能生成物的著作权问题探讨。康德"主客体统一认识论"和"人是目的"哲学视点下,无论人工智能发展到何种阶段,都只能作为人利用的客体和工具处理,而不能将其拟制为与人享有平等地位的法律主体。以此为前提,人工智能生成物应当作为人利用人工智能创作的作品并按照现行著作权法关于作品的构成要件判断其独创性。在人工智能生成物构成作品的情况下,应按照现行著作权法关于著作权归属的原则处理其权利归属,即人工智能生成物的著作权原则上归属于利用人工智能进行作品创作的作者(自然人或者法人或者非法人单位),例外情况下属于雇主或者委托人。'
    print(text)
    print(len(text))

    get_em(text)



if __name__ == "__main__":
    main()
