
def get_lists_by_txt(path):
    with open(path, 'r',encoding='utf-8') as f:
        l=f.readlines()
        word_list = [i.rstrip().split(" ") for i in l]
    return word_list

def get_list_by_txt(path):
    with open(path, 'r',encoding='utf-8') as f:
        l=f.readlines()
        word_list = [i.rstrip() for i in l]
    return word_list

def save_lists_as_text(lists,path):
    foutput = open(path, 'w',encoding='utf-8')
    for list in lists:
        foutput.write(" ".join(list))
        foutput.write("\n")
    foutput.close()
    return

def save_list_as_text(list,path,sep=" "):
    foutput = open(path, 'w',encoding='utf-8')
    foutput.write(sep.join(list))
    foutput.close()
    return