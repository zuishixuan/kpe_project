def read_data_format(path):
    r = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            r.append(line.strip().split('\t'))
    return r
