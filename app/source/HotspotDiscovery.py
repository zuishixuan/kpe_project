from app.utils.FileUtil import *
import math
import multiprocessing
from app.source import BaseModel
from config.constant import work_path, LABEL_DICT


def processed_by_multi_thread(function, multi_range):
    num_thread = int(multiprocessing.cpu_count() / 2)
    pool = multiprocessing.Pool(num_thread)
    res = pool.map(function, multi_range)
    pool.close()
    pool.join()
    return res


class HotspotDiscovery(BaseModel):
    def __init__(self, filename, years=['2020']):
        super().__init__()
        path = work_path + '/app/data/{}'.format(filename)
        self.data = []
        for d in read_data_format(path):
            if len(d) >= 14 and d[9].split('.')[0] in years:
                self.data.append(d)
        self.init()
        pass

    def init(self):
        self.data_init()
        self.all_regions_and_fields_init()
        self.all_patents_nums_sqr_region_init()
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
        n = 0
        for i, item in enumerate(self.data):
            if len(item) != 14:
                n += 1
                continue
            if item[10] == 'nan':
                continue
            for t in item[13]:
                if t not in self.stopwords:
                    fields.add(t)
            self.regions.add(item[10])
        print("no keyword count: {}".format(n))
        self.fields = list(fields)

    def calc(self, i):
        print('i={}:{}'.format(i, self.regions[i]))
        fields = self.get_all_patents_field_by_region(self.regions[i])
        sum = 0
        for f in fields:
            n, a = self.get_research_field_nums_by_region(self.regions[i], f)
            sum += math.pow(n, 2)
        sum = sum if sum != 0 else 1
        self.fields_sqrt_region[self.regions[i]] = math.sqrt(sum)

    def all_patents_nums_sqr_region_init(self):
        self.fields_sqrt_region = {}
        self.region_every_patent_nums_dict = {}

        # processed_by_multi_thread(self.calc, self.regions)

        for r in self.regions:
            if r == 'nan':
                continue
            fields = self.get_all_patents_field_by_region(r)
            sum = 0
            self.region_every_patent_nums_dict[r] = {}
            for f in fields:
                n, a = self.get_research_field_nums_by_region(r, f)
                self.region_every_patent_nums_dict[r][f] = (n, a)
                sum += math.pow(n, 2)
            sum = sum if sum != 0 else 1
            self.fields_sqrt_region[r] = math.sqrt(sum)

    def all_region_patents_nums_init(self):
        self.region_patents_nums_dict = {}
        for r in self.regions:
            self.region_patents_nums_dict[r] = self.get_all_patents_nums_by_regions(r)

    def all_region_every_patent_nums_init(self):
        self.region_every_patent_nums_dict = {}
        for r in self.regions:
            for f in self.fields:
                c = self.get_research_field_nums_by_region(r, f)
                self.region_every_patent_nums_dict[r][f] = {}
                self.region_every_patent_nums_dict[r][f] = c

    # 得到地区的某一研究领域数量
    def get_research_field_nums_by_region(self, region, field):
        n = 0
        a = 0
        for i, item in enumerate(self.data):
            if item[10] == region:
                if field in item[13]:
                    n += 1
                a += 1
        return n, a

    def get_all_patents_field_by_region(self, region):
        fields = set()
        for i, item in enumerate(self.data):
            if item[10] == region:
                for t in item[13]:
                    fields.add(t)
        return fields

    def get_all_patents_nums_sqr_region(self, region):
        fields = self.get_all_patents_field_by_region(region)
        sum = 0
        for f in fields:
            n, a = self.get_research_field_nums_by_region(region, f)
            sum += math.pow(n, 2)
        sum = sum if sum != 0 else 1
        return math.sqrt(sum)

    # 某一地区所有专利说量
    def get_all_patents_nums_by_regions(self, r):
        n = 0
        for i, item in enumerate(self.data):
            if r == item[10]:
                n += 1
        return n

    # 地区某一研究领域数量标准化
    def research_field_standardization(self, region, field):
        n, a = self.region_every_patent_nums_dict[region].get(field, (0, 0))
        # return n / self.get_all_patents_nums_sqr_region(region)
        return n / self.fields_sqrt_region[region]

    # 某一研究领域的所有地区数量
    def get_regions_nums_by_fields(self, field):
        reg = set()
        for i, item in enumerate(self.data):
            if field in item[13]:
                reg.add(item[10])
        return reg

    def get_research_field_hot(self, i):
        field = self.fields[i]

        # if i % 1000 == 0:
        #     print('当前数量{}'.format(i))
        Nfa = 0
        for r in self.regions:
            # n, a = self.get_research_field_nums_by_region(r, field)
            n, a = self.region_every_patent_nums_dict[r].get(field, (0, 0))
            patent_nums = self.region_patents_nums_dict.get(r, 1)
            if patent_nums == 0:
                patent_nums = 1
            Nfa += self.research_field_standardization(r, field) * math.exp(n / patent_nums)

        return (LABEL_DICT.get(field), Nfa * len(self.get_regions_nums_by_fields(field)) / len(self.regions))

    def get_all_research_field_hot(self, reverse=True, top=5):
        result = []

        # result = processed_by_multi_thread(self.get_research_field_hot, range(len(self.fields)))
        for i, f in enumerate(self.fields):
            # result = processed_by_multi_thread(self.get_research_field_hot, f)
            result.append(self.get_research_field_hot(i))
        return sorted(result, key=lambda x: x[1], reverse=reverse)[:top]
