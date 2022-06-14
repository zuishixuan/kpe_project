from app.common import BaseEntity


class PatentEntity(BaseEntity):

    def __init__(self):
        self.id = None
        self.title = None
        self.summary = None
        self.signory = None
        self.year = None
        self.area = None
