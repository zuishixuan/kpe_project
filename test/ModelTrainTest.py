from model.ModelTrain import dict_text_train_model
from config.constant import work_path
import time, gensim
from model.Visualisation import plot


def text_train_model(text_path=None):
    begin = time.time()
    # 引入数据集
    sentences = [['woman', 'queue'], ['man', 'king']]
    # train word2vec on the two sentences
    model = gensim.models.Word2Vec(sentences, min_count=1)
    end = time.time()
    print('耗时：' + str(end - begin))
    return model


def test_text_model():
    model = text_train_model()
    # r_1 = model.wv['first']
    # print(r_1)
    # r_2 = model.wv.distance('first', 'second')
    # print(r_2)
    r_3 = model.wv.similarity('woman', 'man')
    r_4 = model.wv.most_similar(positive=['woman', 'queue'], negative=['man'], topn=1)
    print(r_3)
    print(r_4)


def test_train_model():
    model = dict_text_train_model(work_path + '/model/data/train')
    r_1 = model.wv['半导体']
    print(r_1)
    r_2 = model.wv.similarity('半导体', '比亚迪')
    print(r_2)
    r_3 = model.wv.most_similar(positive=['比亚迪', '汽车'], negative=['半导体'], topn=3)
    print(r_3)
    model_path = work_path + '/model/test.bin'
    model.save(model_path)


def test_load_model():
    model_path = work_path + '/model/test.bin'
    new_model = gensim.models.Word2Vec.load(model_path, mmap='r')
    print(new_model.wv)
    print(new_model.vector_size)
    words = new_model.wv.index_to_key
    print("================len(words)================")
    print(len(words))
    for word in words:
        print(word)


def test_visual():
    model_path = work_path + '/model/test.bin'
    new_model = gensim.models.Word2Vec.load(model_path)
    plot(new_model)
    # model_path = work_path + '/model/vectors_dic.bin'
    # model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)
    # # words = model.index_to_key
    # # for word in words:
    # #     print(word)
    # plot(model)
