# -*- coding: utf-8 -*-
"""
常数类
"""
import os
import csv

# 工作目录
work_path = os.path.dirname(os.path.dirname(__file__))
# 分类
# label_path = work_path + "/config/data/label.csv"
label_path = work_path + "/config/data/all.csv"
# 关键词
# keywords_path = work_path + "/config/data/label_keywords.csv"

LABEL_LIST = []
with open(label_path, 'r', encoding="UTF-8") as f:
    reader = csv.DictReader(f)
    for item in reader:
        LABEL_LIST.append(item)

LABEL_DICT = dict()

for label in LABEL_LIST:
    LABEL_DICT[label.get('id')] = label.get('name')


def get_label_leaf():
    parent_ids = set()
    parent_ids.add('-1')
    for label in LABEL_LIST:
        parent_ids.add(label.get('pid'))
    leaf = set()
    for label in LABEL_LIST:
        _id = label.get('id')
        if _id not in parent_ids:
            leaf.add(label.get('id'))
    return list(leaf)


LABEL_LEAF = get_label_leaf()

LABEL_ID_TO_LEAF = dict()


def find_leaf(_id):
    result = []
    if _id in LABEL_LEAF:
        result.append(_id)
        return result
    for label in LABEL_LIST:
        if label.get('pid') == _id:
            result += find_leaf(label.get('id'))
    return result


for label in LABEL_LIST:
    LABEL_ID_TO_LEAF[label.get('id')] = find_leaf(label.get('id'))


# 关键词
# with open(keywords_path, 'r', encoding="UTF-8") as f:
#     KEYWORD_LIST = []
#     reader = csv.DictReader(f)
#     for item in reader:
#         KEYWORD_LIST.append(item)

if __name__ == '__main__':
    print(work_path)
    print(LABEL_LIST)
    print(LABEL_DICT)
    for item in LABEL_LIST:
        print(item.get('name'))
