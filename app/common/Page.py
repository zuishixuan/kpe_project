from app.common.BaseEntity import BaseEntity


class Page(BaseEntity):
    def __init__(self, total=0, records=[]):
        self.total = total
        self.records = records
