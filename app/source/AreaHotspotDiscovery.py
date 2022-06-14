from app.utils.FileUtil import *
import math
import multiprocessing
from app.source import BaseModel
from config.constant import work_path, LABEL_DICT, LABEL_LEAF, LABEL_ID_TO_LEAF


def processed_by_multi_thread(function, multi_range):
    num_thread = int(multiprocessing.cpu_count() / 2)
    pool = multiprocessing.Pool(num_thread)
    res = pool.map(function, multi_range)
    pool.close()
    pool.join()
    return res


class AreaHotspotDiscovery(BaseModel):
    def __init__(self, filename, years=['2020']):
        super().__init__()
        path = work_path + '/app/data/{}'.format(filename)
        self.data = []
        self.region_year_patent_num = {}
        for d in read_data_format(path):
            if len(d) >= 14 and d[9].split('.')[0] in years:
                self.data.append(d)
        self.init()
        pass

    def init(self):
        self.data_init()
        self.all_regions_and_fields_init()
        self.all_region_patents_nums_init()

    def data_init(self):
        for o, item in enumerate(self.data):
            if len(item) < 14:
                continue
            item[9] = item[9].split('.')[0]
            item[13] = item[13].split(',')

    def all_regions_and_fields_init(self):
        fields = set()
        self.regions = set()
        self.tech_to_area = dict()
        n = 0
        for i, item in enumerate(self.data):
            if len(item) != 14:
                n += 1
                continue
            if item[10] == 'nan':
                continue
            if self.region_year_patent_num.get(item[10]) is None:
                self.region_year_patent_num[item[10]] = {}
            for t in item[13]:
                if t not in self.stopwords:
                    if self.region_year_patent_num[item[10]].get(t) is None:
                        self.region_year_patent_num[item[10]][t] = 1
                    else:
                        self.region_year_patent_num[item[10]][t] += 1
                    if not self.tech_to_area.__contains__(t):
                        self.tech_to_area[t] = {}
                    if not self.tech_to_area[t].__contains__(item[10]):
                        self.tech_to_area[t][item[10]] = 1
                    else:
                        self.tech_to_area[t][item[10]] += 1
                    fields.add(t)
            self.regions.add(item[10])
        print("no keyword count: {}".format(n))
        self.fields = list(fields)

    def all_region_patents_nums_init(self):
        self.region_patents_nums_dict = {}
        for r in self.regions:
            self.region_patents_nums_dict[r] = self.get_all_patents_nums_by_regions(r)

    # 某一地区所有专利说量
    def get_all_patents_nums_by_regions(self, r):
        n = 0
        for i, item in enumerate(self.data):
            if r == item[10]:
                n += 1
        return n

    def get_area_research_field_hot(self, reverse=True, top=5):
        result = []
        for r in self.regions:
            p_n = self.region_year_patent_num[r]
            children = [{'name': '{}相关技术'.format(LABEL_DICT[item[0]]), 'value': item[1]} for item in sorted(p_n.items(), key=lambda x: x[1], reverse=reverse)[:top]]
            result.append({'name': r, 'value': sum([c['value'] for c in children]), 'children': children})
        return result

    def get_hot_ares(self, t_id, reverse=True, top=5):
        if t_id in LABEL_LEAF:
            result = []
            for a, n in self.tech_to_area.get(t_id, {}).items():
                if a != 'None':
                    result.append({'name': '{}'.format(a), 'value': n})
        else:
            dic = {}
            for _id in LABEL_ID_TO_LEAF.get(t_id, []):
                for a, n in self.tech_to_area.get(_id, {}).items():
                    if dic.get(a) is None:
                        dic[a] = n
                    else:
                        dic[a] += n
            result = [{'name': '{}'.format(a), 'value': n} for a, n in dic.items()]

        return sorted(result, key=lambda x: x['value'], reverse=reverse)[:top]
