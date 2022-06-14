from app.dao.mysql import MetadataDAO
import datetime
from manager import app

if __name__ == '__main__':
    with app.app_context():
        #detail = PatentDAO.count_by_filter()
        #print(detail)
        #entity = {'id': 2, 'name': '英汉常用生物学词汇数据库', 'identifier': 'CSTR:13913.11.micro.dic.biological-vocabulary', 'subject_category': '生物学', 'theme_category': '微生物资源', 'describe': '英汉常用生物学词汇数据库收集并整合了生物类常用的英汉对照词汇内容，包括中英文名称，缩写，类别等内容。', 'keyword': '生物学;英汉;词汇', 'keyword_deny': None, 'keyword_score': '0.89;0.99;0.78', 'keyword_deny_score': None, 'generate_date': datetime.date(2020, 9, 16), 'submit_date': datetime.date(2021, 5, 6), 'submit_user': 'admin'}
        b= MetadataDAO.MetadataDAO.get_detail_by_identifier('CSTR:13913.11.micro.dic.biological-vocabulary')
        print("bbbbbbbbbbbb")
        print(b)
        a= MetadataDAO.MetadataDAO.count()
        print(a)

