import os
from enum import Enum
from pathlib import Path

model_dir = '../chinese_wwm_pytorch'
output_dir = './model_save/1'
# model_dir = './RoBERTa_zh_L12_PyTorch'
# 模型存储到的路径
# output_dir = './PoBERTa_model_save/'

corpus_dir = '../data/data.xlsx'
stopwords_path = '../data/stopWord.txt'

subject_category_correct = {
    "畜牧学": "畜牧、兽医科学",
    "信息与系统科学相关工程与技术": "信息科学与系统科学",
    "电子与通信技术": "电子、通信与自动控制技术",
    "环境科学技术及资源科学技术": "环境科学技术",
    "预防医学与公共卫生学": "预防医学与卫生学",
    "工程与技术科学基础学": "工程与技术科学基础学科"
}
subject_category_tag = {
    '工程与技术科学基础学科': 0,
    '地球科学': 1,
    '生物学': 2,
    '自然科学相关工程与技术': 3,
    '物理学': 4,
    '林学': 5,
    '环境科学技术': 6,
    '测绘科学技术': 7,
    '农学': 8,
    '天文学': 9,
    '统计学': 10,
    '社会学': 11,
    '其他': 12
}

tag_map = {"O": 0, "B-KEY": 1, "I-KEY": 2, "E-KEY": 3, "S-KEY": 4, "START": 5, "END": 6}
num_tag = 7


# ('工程与技术科学基础学科', 27698), ('地球科学', 27518), ('生物学', 26738), ('自然科学相关工程与技术', 19993), ('物理学', 19975), ('林学', 12236), ('环境科学技术', 7923), ('测绘科学技术', 7630), ('农学', 2807), ('天文学', 643), ('统计学', 614), ('社会学', 431), ('预防医学与卫生学', 140), ('水利工程', 116), ('临床医学', 102), ('基础医学', 83), ('药学', 56), ('化学工程', 50), ('畜牧、兽医科学', 46), ('计算机科学技术', 37), ('交通运输工程', 36), ('信息科学与系统科学', 34), ('中医学与中药学', 33), ('经济学', 30), ('水产学', 27), ('能源科学技术', 18), ('历史学', 16), ('安全科学技术', 11), ('材料科学', 10), ('力学', 9), ('动力与电气工程', 5), ('军事医学与特种医学', 5), ('电子、通信与自动控制技术', 3), ('心理学', 3)
# 34

def get_real_dir(rel_dir):
    d = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(d, rel_dir)


def create_dir(dir):
    # 目录不存在则创建
    dir = Path(dir)
    if not dir.exists():
        os.makedirs(dir, exist_ok=True)


class Model_name(Enum):
    JointMarkScoreKpe = "JointMarkScoreKpe"
    JointMarkScore2AttenKpe = "JointMarkScore2AttenKpe"
    SimCSE = 'SimCSE'
    TextRank = 'TextRank'


class Dataset_name(Enum):
    cnki_1 = "cnki_1"
    cnki_2 = "cnki_2"
    metadata_1 = "metadata_1"
    metadata_2 = "metadata_2"
    metadata_all = "metadata_all"
    metadata_test = "metadata_test"
    combine = "combine"
    combine_2 = "combine_2"


Pretrained_na = Model_name.SimCSE.value

# val
Model_na = Model_name.JointMarkScore2AttenKpe.value
Valid_dataset_na = Dataset_name.cnki_1.value

Dataset_na = Dataset_name.cnki_2.value
out = 'out7'
epoch = 'epoch_6'
device = "cuda:1"
# Dataset_na = Dataset_name.combine_2.value
# out = 'out'
# epoch = 'epoch_5'
# device = "cuda:1"
# ----------------------
valid_data_num = 300
random_seed = 14
shuffle = True
skip = True
test_data_list = [
    {
        "text":
            "算法权力的兴起、异化及法律规制。在人工智能时代,具有自主学习与决策功能的智能算法与人的行为分离并超越了工具化范畴。算法基于海量数据运算配置社会资源,直接作为行为规范影响人的行为,辅助甚至取代公权力决策,从而发展为一支新兴的技术权力。算法权力以处理海量数据的机器优势、基于复杂算法生态的架构优势与对社会权力运行的嵌入优势为基础,形成了跨越性与隔离性的特征。由于缺乏有效规制,算法权力在商业领域形成与消费者的不公平交易,催生了监视资本主义;在公权力领域嵌入公权力运行的各个环节,甚至成为独立的决策者,严重挑战正当程序制度和权力专属原则。防范算法权力异化,应建立限权与赋权的制度体系。限权制度包括明确算法的应用范围与限制条件,建立正当程序制度和算法问责机制;赋权制度包括赋予公民个人数据权利,建立事后救济制度,加强行业自律以及引入第三方参与合作治理。",
        "keyword":
            "人工智能,算法,权力异化,法律规制"
    },
    {
        "text":
            '马克思历史认识模式的复杂性及实践解读。在马克思历史认识模式中,资本主义与社会主义关系的普遍和特殊是联结在一起的,但是由于时代和相关理论表述的局限性,特别是马克思晚年为俄国农村公社和整个俄国缩短向社会主义发展历程所设定的"历史环境"没有出现,这就给后人留下了把"普遍"和"特殊"分离开来进而对立起来的"空间"。由于种种原因,1905年以后,在普列汉诺夫和列宁那里,资本主义与社会主义关系的普遍和特殊开始变为两种独立的认识范式。虽然列宁在反外国武装干涉和国内战争结束后不久就意识到了这方面问题的严重性,故把"普遍"和"特殊"统一起来的趋向在他当时的相关思想中以不太稳定的形式显示出来,但他过早的去世又中断了这种趋向。斯大林社会主义模式把资本主义与社会主义关系的普遍和特殊彻底割裂开来,这对落后国家社会主义理论和实践产生了一定的影响。自改革开放以来形成和发展起来的中国特色社会主义,在把资本主义与社会主义关系的普遍和特殊有机结合起来、超越马克思历史认识模式所产生悖论的方面,提供了一条可供选择的现实性路径。',
        "keyword":
            "历史认识模式;社会主义;资本主义;中国特色社会主义"
    },
    {
        "text":
            '车辆牌照识别系统综述。基于图像和字符识别技术的智能化交通管理系统——车辆牌照识别系统 ,一般要先对原始图像进行转换、压缩、增强、水平校正等预处理 ,再用边缘检测法对牌照进行定位与分割 ,而字符识别多采用特征提取与模式匹配等方法 .从中可以看出 :多种预处理与识别技术有机结合以提高系统识别能力 ,在有效、实用的原则下将神经网络与人工智能技术相结合将成为模式识别研究的两个重要发展趋势',
        "keyword":
            "模式识别,字符,人工智能,车辆牌照识别系统"
    },
    {
        "text":
            '20世纪特别是40年代以来，生物学吸收了数学、物理学和化学等的成就，逐渐发展成一门精确的、定量的、深入到分子层次的科学，人们已经认识到生命是物质的一种运动形态。生命的基本单位是细胞（由蛋白质、核酸、脂质等生物大分子组成的物质系统）。生命现象就是这一复杂系统中物质、能量和信息三个量综合运动与传递的表现。生命有许多为无生命物质所不具备的特性。例如，生命能够在常温、常压下合成多种有机化合物，包括复杂的生物大分子；能够以远远超出机器的生产效率来利用环境中的物质和能制造体内的各种物质，而很少排放污染环境的有害物质；能以极高的效率储存信息和传递信息；具有自我调节功能和自我复制能力；以不可逆的方式进行着个体发育和物种的演化等等，揭露生命过程中的机制具有巨大的理论和实践意义。',
        "keyword":
            ""
    },
    {
        "text":
            '从生物的基本结构单位──细胞的水平来考察，有的生物尚不具备细胞形态，在已具有细胞形态的生物中，有的由原核细胞构成，有的由真核细胞构成。从组织结构水平来看，有的是单生的或群体的单细胞生物，有的是多细胞生物，而多细胞生物又可根据组织器官的分化和发展而分为多种类型。从营养方式来看，有的是光合自养，有的是吸收异养或腐食性异养，有的是吞食异养。从生物在生态系统中的作用来看，有的是有机食物的生产者，有的是消费者，有的是分解者，等等。',
        "keyword":
            ""
    },
    {
        "text":
            '生物学（Biology）(简称生物或生命科学)，是研究生物（包括植物、动物和微生物）的结构、功能、发生和发展规律的科学，是自然科学的一个部分。目的在于阐明和控制生命活动，改造自然，为农业、工业和医学等实践服务。几千年来，人类在农、林、牧、副、渔和医药等实践中，积累了有关植物、动物、微生物和人体的丰富知识。1859年，英国博物学家达尔文《物种起源》的发表，确立了唯物主义生物进化观点，推动了生物学的迅速发展。',
        "keyword":
            ""
    },
    {
        "text":
            '新型冠状病毒（2019-nCoV）假病毒阴性对照hRNaseP基因标准物质,编号：GBW(E)091117 名称：新型冠状病毒（2019-nCoV）假病毒阴性对照hRNaseP基因标准物质 应用领域：临床\卫生及法医/临床检验 保存条件： 注意事项：',
        "keyword":
            ""
    },
    {
        "text":
            '辽宁省1km水稻光合资源利用率数据集（1960-2010年）,粮食生产对农业自然资源的利用率是评价农业生产水平高低的科学指标，该数据集反映了辽宁省水稻光合资源利用率的空间分布情况，包含水稻光合资源利用率一个指标，可为东北地区农业资源管理等研究提供数据支撑服务。',
        "keyword":""
    },
    {
        "text":
            '中国地面气象站逐小时观测资料,中国国家级地面站小时值数据,包括气温、气压、相对湿度、水汽压、风、降水量等要素小时观测值。实时数据经过质量控制。 各要素数据的实有率超过99.9%，数据的正确率均接近100%。',
        "keyword":
            ""
    },
    {
        "text":
            '北京市提供新冠病毒核酸检测服务的医疗卫生机构数据集,2020年6月13日，北京市卫生健康委员会公布了98所具备核酸检测能力的医疗卫生机构，可面向团体和个人提供新冠病毒核酸检测服务。本数据集在北京市卫健委公布名单的基础上，由国家人口健康科学数据中心数据资源建设团队加工整理。数据集包括检测机构的名称、所属辖区、详细地址、联系电话、预约方式、面向对象等，对电话预约、网站预约、微信公众号预约、其他预约等方式进行了划分，便于群众浏览、查询和预约。',
        "keyword":
            ""
    },
    {
        "text":
            '基于信息增益与相似度的专利关键词抽取算法评价模型,针对目前专利关键词抽取算法评价中主要采用抽取的关键词与专家人工标注关键词进行匹配存在的问题,提出一种基于信息增益与相似度的专利关键词抽取算法评价模型。[方法/过程]提出的评价模型从内部和外部两个层面评估专利关键词抽取算法的准确性。其中,内部评价模型度量待评价算法抽取的每个关键词的信息增益,以评估被抽取的关键词的新颖性与创造性;外部评价模型使用待评价算法抽取的关键词集表示专利,计算相关专利的相似度,衡量算法抽取的关键词描述专利主题的有效性。[结果/结论]通过评价模型有效性验证实验与评价模型应用实证研究,结果表明提出的基于信息增益与相似度的评价模型具有可行性与有效性。 ',
        "keyword":
            "专利;关键词抽取;评价;信息增益;相似度"
    },

]

keyword_dict_dir = '../data/keyword_dict/keyword_dict_filter.txt'
keyword_dict_dir = get_real_dir(keyword_dict_dir)


class Eval_type(Enum):
    eval = "eval"
    batch_kpe = "batch_kpe"
    test = 'test'


eval_type = Eval_type.test.value
# train
# Model_na = Model_name.JointMarkScore2AttenKpe.value
# Dataset_na = Dataset_name.combine_2.value
# out ='out'
# epoch = 'epoch_3'
# device = "cuda:0"
# epoch_num = 8
# test_100 = False
#
# deep_train = False
# last_epoch_num = 4

Pretrained_model_dir = {
    'SimCSE': "../model/chinese_roberta_wwm_ext_pytorch"
}

Model_file_out_dir = {
    'SimCSE': "../model/SimCSE/cnki_2" + "/out",
    'JointMarkScoreKpe': "../model/JointMarkScoreKpe/" + Dataset_na + "/" + out,
    'JointMarkScore2AttenKpe': "../model/JointMarkScore2AttenKpe/" + Dataset_na + "/" + out,
    'TextRank': "../model/TextRank/cnki_2" + "/out"
}

Model_file_read_dir = {
    'SimCSE': "../model/SimCSE/cnki_2" + "/out" + "/epoch_1-batch_12100",
    'JointMarkScoreKpe': "../model/JointMarkScoreKpe/" + Dataset_na + "/" + out + '/' + epoch,
    'JointMarkScore2AttenKpe': "../model/JointMarkScore2AttenKpe/" + Dataset_na + "/" + out + '/' + epoch,
    'TextRank': ''
}

Train_dataset_file_dir = {
    'cnki_1': "../data/cnki/data.csv",
    'cnki_2': "../data/cnki_2/data_mark.csv",
    'metadata_1': "../data/metadata_1/data_mark.csv",
    'metadata_2': "../data/metadata_2/data_mark.csv",
    'metadata_all': "../data/metadata_all/data_mark.csv",
    'metadata_test': "../data/metadata_test/data_mark.csv",
    'combine': "../data/combine/data_mark.csv",
    'combine_2': "../data/combine_2/data_mark.csv"
}

Valid_dataset_file_dir = {
    'cnki_1': "../data/cnki/data.csv",
    'cnki_2': "../data/cnki_2/data_mark.csv",
    'metadata_1': "../data/metadata_1/data_mark.csv",
    'metadata_2': "../data/metadata_2/data_mark.csv",
    'metadata_all': "../data/metadata_all/data_mark.csv",
    'metadata_test': "../data/metadata_test/data_mark.csv",
    'combine': "../data/combine/data_mark.csv",
    'combine_2': "../data/combine_2/data_mark.csv"
}

for key in Train_dataset_file_dir:
    Train_dataset_file_dir[key] = get_real_dir(Train_dataset_file_dir[key])

for key in Valid_dataset_file_dir:
    Valid_dataset_file_dir[key] = get_real_dir(Valid_dataset_file_dir[key])

for key in Pretrained_model_dir:
    Pretrained_model_dir[key] = get_real_dir(Pretrained_model_dir[key])

for key in Model_file_out_dir:
    Model_file_out_dir[key] = get_real_dir(Model_file_out_dir[key])

for key in Model_file_read_dir:
    Model_file_read_dir[key] = get_real_dir(Model_file_read_dir[key])
