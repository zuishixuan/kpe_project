#from source.new_word_discovery.new_word_discovery import main
import torch
import jieba

# from source.train_MarkKpe import main
#from source.test import main
#from source.train_ScoreKpe import main
#from source.train_MarkAttenKpeDDP import main,test
#from source.train_ScoreAttenKpeDDP import main,test
#from source.train_ScoreAttenKpe2DDP import main,test
from source.train_JointMarkScoreKpeDDP import main,test

if __name__ == '__main__':
    # i = torch.Tensor([1,2,3,4])
    # label= torch.Tensor([1,0,1,0,1,1])
    # print(i[label==1])

   main()
   #test()
   #  res = jieba.cut('中国共产党领导社会主义现代化的五维向度——基于新中国70年历程的思考。社会主义现代化在新中国70年的历史进程中取得了举世瞩目的伟大成就,实现了巨大而深度的跃升。回顾这一探索历程,可沿着中国共产党领导社会主义现代化建设的基本逻辑展开,这就同其他现代化模式相区别,彰显现代化理论的"中国特色现代性"和实践探索的中国逻辑,并可从五个维度表征中国共产党领导社会主义现代化建设的逻辑内涵,即中国共产党作为伟大使命型政党,明确设定社会主义现代化的伟大目标;作为善于学习型政党,不断深化借鉴现代化模式的合理内核;作为自我革命型政党,通过政党的革命化全方位保证现代化;作为实干奋斗型政党,用超强行动力交出现代化的历史答卷;作为服务人民型政党,致力将现代化成果充分惠及全体人民。')
   #  for w in res:
   #      print(w)