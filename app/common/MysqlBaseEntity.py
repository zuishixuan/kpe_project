from flask_sqlalchemy import SQLAlchemy
from .BaseEntity import BaseEntity

db = SQLAlchemy()


class MySqlBaseEntity(BaseEntity):
    pass
