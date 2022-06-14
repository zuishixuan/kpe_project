from .BaseApi import BaseApi


class PatentApi(BaseApi):

    def get_list(self, pageno, pagesize):
        page = self.get_page(pageno, pagesize)
        print(page)
