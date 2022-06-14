from app.utils.FileUtil import *
import matplotlib.pyplot as pltfrom
from app.dao.mysql import PatentStopwordsDAO
from app.source import BaseModel
from config.constant import work_path, LABEL_LEAF, LABEL_ID_TO_LEAF
import random


class PathEvolution(BaseModel):
    def __init__(self, filename):
        # stopwords = PatentStopwordsDAO.do_query_by_sql_filter('status', '1')
        # self.stopwords = [word[1] for word in list(stopwords)]
        super().__init__()
        self.data = read_data_format(work_path+'/app/data/{}'.format(filename))
        self.init()

    def init(self):
        # super().init()
        # self.init_data()
        self.technology_init()

    # def init_data(self):
    #     result = self.get_all_data()
    #     print('ssss')
    #     pass

    def technology_init(self):
        self.year_to_tech = {}
        self.area_to_tech = {}
        for o, item in enumerate(self.data):
            if len(item) < 14:
                continue
            year = item[9].split('.')[0]
            tech = item[13].split(',')
            # if not item[13].endswith(','):
            #     tech = tech[:-1]
            if not self.year_to_tech.__contains__(year):
                self.year_to_tech[year] = {}
            if not self.area_to_tech.__contains__(item[10]):
                self.area_to_tech[item[10]] = {}
            if not self.area_to_tech[item[10]].__contains__(year):
                self.area_to_tech[item[10]][year] = {}
            for i, t in enumerate(tech):
                t = t.replace(' ', '', -1)
                c = 1
                if t == '' or t == 'nan' or t in self.stopwords:
                    continue
                if t in LABEL_LEAF:
                    if self.year_to_tech[year].get(t) is None:
                        self.year_to_tech[year][t] = c
                    else:
                        self.year_to_tech[year][t] += c
                    if self.area_to_tech[item[10]][year].get(t) is None:
                        self.area_to_tech[item[10]][year][t] = c
                    else:
                        self.area_to_tech[item[10]][year][t] += c

    def technology_path(self, path_top=3):
        r = {}
        for year, techs in self.year_to_tech.items():
            sum = 0
            tech_s = sorted(techs.items(), key=lambda x: x[1], reverse=True)
            for t, c in techs.items():
                sum+=c
            top = path_top if len(tech_s) >= path_top else len(tech_s)
            r[year] = {'tech': [tech_s[i][0] for i in range(top)], "per": [tech_s[i][1] / sum for i in range(top)],
                       "count": [tech_s[i][1] for i in range(top)], 'total': sum}
        return sorted(r.items(), key=lambda x:x[0], reverse=True)

    def technology_path_by_id(self, t_id):
        r = {}
        if t_id in LABEL_LEAF:
            for year, techs in self.year_to_tech.items():
                sum = 0
                tech_num = techs.get(t_id, None)
                for t, c in techs.items():
                    sum+=c

                if tech_num is None:
                    _nums = r.get(str(int(year)+1), {'tech': t_id, "per": 0, "count": 3, 'total': sum})['count'] + 1
                    sum+=_nums
                    r[year] = {'tech': t_id, "per": _nums / sum, "count": _nums, 'total': sum}
                else:
                    r[year] = {'tech': t_id, "per": tech_num / sum, "count": tech_num, 'total': sum}
                if r[year]['per'] < 0.002:
                    r[year]['per'] = 0.002
        else:
            for _id in LABEL_ID_TO_LEAF.get(t_id, []):
                for year, techs in self.year_to_tech.items():
                    tech_num = techs.get(_id, None)
                    if r.get(year) is None:
                        sum = 0
                        for t, c in techs.items():
                            sum += c
                        r[year] = {'tech': t_id, "per": 0, "count": 0, 'total': sum}
                    if tech_num is not None:
                        r[year]["count"] += tech_num
                        if r[year]["total"] == 0:
                            print(r[year]["count"])
                            r[year]["total"] = 1
                        r[year]["per"] = r[year]["count"] / r[year]["total"]

            for y, item in r.items():
                # if item['per'] < 0.001:
                #     item['per'] = r.get(str(int(y)+2), {'tech': t_id, "per": 0.0012})['per'] + max((
                #                         r.get(str(int(y) + 2), {'tech': t_id, "per": 0.0012})[
                #                             'per'] - r.get(str(int(y) + 1), {'tech': t_id, "per": 0.0011})['per']
                #                 ), 0.0001)
                if item['per'] < 0.001:
                    item['per'] = r.get(str(int(y)+2), {'tech': t_id, "per": 0.0012})['per'] + 0.0001

        return r

    def technology_path_by_area(self, t_area, path_top=3):
        r = {}
        if self.area_to_tech.get(t_area) is None:
            return None
        for year, techs in self.area_to_tech[t_area].items():
            sum = 0
            tech_s = sorted(techs.items(), key=lambda x: x[1], reverse=True)
            for t, c in techs.items():
                sum+=c
            top = path_top if len(tech_s) >= path_top else len(tech_s)
            r[year] = {'tech': [tech_s[i][0] for i in range(top)], "per": [tech_s[i][1] / sum for i in range(top)],
                       "count": [tech_s[i][1] for i in range(top)], 'total': sum}
        return sorted(r.items(), key=lambda x: x[0], reverse=True)


    def technology_ids_by_area(self, t_area, path_top=3):
        r = {}
        if self.area_to_tech.get(t_area) is None:
            return None
        for year, techs in self.area_to_tech[t_area].items():
            sum = 0
            tech_s = sorted(techs.items(), key=lambda x: x[1], reverse=True)
            for t, c in techs.items():
                sum+=c
            top = path_top if len(tech_s) >= path_top else len(tech_s)
            r[year] = {'tech': [tech_s[i][0] for i in range(top)], "per": [tech_s[i][1] / sum for i in range(top)],
                       "count": [tech_s[i][1] for i in range(top)], 'total': sum}
        return sorted(r.items(), key=lambda x: x[0], reverse=True)

    def technology_path_by_name(self, name):
        r = {}
        for year, techs in self.year_to_tech.items():
            sum = 0
            max = 1
            for t, c in techs.items():
                sum += c
                if t == name:
                    max += c
            r[year] = {'tech': name, "per": max / sum, "count": max, 'total': sum}
        return r

    def change_data(self, filename):
        self.data = read_data_format(work_path+'/app/data/{}'.format(filename))
        self.init()



