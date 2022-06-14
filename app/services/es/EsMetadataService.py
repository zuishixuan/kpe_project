import json

from app.dao.es.EsMetadataRepository import EsMetadataRepository


class EsMetadataService:
    _client = EsMetadataRepository()

    @classmethod
    def count(cls):
        return cls._client.count()

    @classmethod
    def get_page(cls, page, size):
        return cls._client.get_page(page, size)

    @classmethod
    def get_by_id(cls, id):
        return cls._client.get_by_id(id)

    @classmethod
    def save(cls, source):
        return cls._client.save(source)

    @classmethod
    def get_query_page(cls, page, size, field, field_value):
        return cls._client.get_query_page(page, size, field, field_value)
