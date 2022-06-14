from app.utils.FileUtil import *
from app.source.KalmanFilter import KalmanFilter
from config.constant import work_path


class Prediction:
    def __init__(self, filename):

        path = work_path + '/app/data/{}'.format(filename)
        self.data = []
        for d in read_data_format(path):
            if len(d) >= 14:
                self.data.append(d)
        self.technology_init()
        self.year_dict_init()
        # 初始化 Kalman 滤波器
        self.kf = KalmanFilter()
        self.technology_year_init()
        self.hotspot_prediction_init()
        self.expand = 10000

    def technology_init(self):
        self.technology_dict = {}
        self.technology_year_dict = {}

        for item in self.data:
            item[9] = item[9].split('.')[0]
            for t in item[13].split(','):
                if self.technology_dict.get(t) is None:
                    self.technology_dict[t] = 1
                else:
                    self.technology_dict[t] += 1
                # if self.technology_year_dict.get(t) is None:
                #     for self
                #     self.technology_year_dict[t] = 1
                # else:
                #     self.technology_dict[t] += 1

    def year_dict_init(self):
        self.year_to_count = {}
        for item in self.data:
            item[9] = item[9].split('.')[0]
            if self.year_to_count.get(item[9]) is None:
                self.year_to_count[item[9]] = 1
            else:
                self.year_to_count[item[9]] += 1

    def technology_year_init(self):
        self.technology_year_dict = {}
        for item in self.data:
            for t in item[13].split(','):
                if self.technology_year_dict.get(t) is None:
                    self.technology_year_dict[t] = {}
                    for y in self.year_to_count.keys():
                        self.technology_year_dict[t][y] = 0
                self.technology_year_dict[t][item[9]] += 1

    def hotspot_prediction_init(self):
        self.tenchnology_year_to_per = dict()
        for t, ys in self.technology_year_dict.items():
            self.tenchnology_year_to_per[t] = {}
            for y, v1 in self.year_to_count.items():
                self.tenchnology_year_to_per[t][y] = ys[y] / v1
