from source.kpe_eval import main,test,eval_kpe_file
from source.constant import *

if __name__ == '__main__':
    if eval_type == Eval_type.eval.value:
        main()
    if eval_type == Eval_type.test.value:
        test()
    if eval_type == Eval_type.batch_kpe.value:
        eval_kpe_file()