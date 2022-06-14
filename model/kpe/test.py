# from source.test import main
from fuzzywuzzy import fuzz, process

if __name__ == '__main__':
    keyword = ['恒温恒湿', '实验设备', '试验箱']
    print(fuzz.ratio("恒温", "恒温恒湿"))
    matchname = process.extract("恒温", keyword, scorer=fuzz.token_set_ratio)
    print(matchname)

    # main()
