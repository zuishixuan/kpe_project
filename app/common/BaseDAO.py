from app.common.MysqlBaseEntity import db
from app.common.Page import Page
from datetime import datetime
from sqlalchemy import desc, func, text, asc
from config.config import app

# 直接写sql会有数据库绑定的问题，需要如下操作做适配
from logger import logger


def get_session(entity):
    if hasattr(entity, '__bind_key__'):
        bind_key = entity.__bind_key__
        session = db.create_scoped_session(
            options=dict(bind=db.get_engine(app, bind_key), binds={})
        )
    else:
        session = db.session
    return session


class BaseDAO(object):
    @classmethod
    def query_all_by_ids(cls, entity, ids):
        return entity.query.filter(entity.id.in_(ids)).all()

    @classmethod
    def do_batch_save(cls, entity, data_list):
        if data_list is not None and len(data_list) > 0:
            session = get_session(entity)
            # 下面两个方法均可
            # session.execute(entity.__table__.insert(), data_list)
            session.bulk_insert_mappings(entity, data_list)
            session.commit()
            session.close()
            return True
        return False

    @classmethod
    def do_group_length_by_sql(cls, entity, group_column):
        print(hasattr(entity, group_column))
        if group_column is None or not hasattr(entity, group_column):
            return []

        table_name = entity.__tablename__
        sql = text("SELECT t.name, COUNT(t.name) value FROM (" \
                   "SELECT LENGTH( " + group_column + " ) name " \
                                                      "FROM " + table_name + "" \
                                                                             ") t " \
                                                                             "GROUP BY t.name " \
                                                                             "ORDER BY value DESC")
        session = get_session(entity)
        result = session.execute(sql)
        session.commit()
        session.close()
        return result

    @classmethod
    def do_query_by_sort(cls, entity, order_by, top_n):
        res = entity.query.order_by(desc(order_by)).limit(top_n)

        return [i.to_json() for i in res]

    @classmethod
    def do_query_by_sql(cls, entity, columns):
        if isinstance(columns, str):
            column = columns
            if column is None or not hasattr(entity, column):
                return []

            table_name = entity.__tablename__
        else:
            for column in columns:
                if column is None or not hasattr(entity, column):
                    return []
            table_name = entity.__tablename__
            column = ','.join(columns, )
        sql = text("SELECT " + column + " FROM " + table_name + " WHERE filter>1")

        session = get_session(entity)
        result = session.execute(sql)
        session.commit()
        session.close()
        return result

    @classmethod
    def do_group_by_column(cls, entity, id_column, group_column):
        session = get_session(entity)
        id_field = getattr(entity, id_column)
        group_field = getattr(entity, group_column)
        if group_field is None or id_field is None:
            return []
        result = session.query(group_field, func.count(id_field)) \
            .group_by(group_field).all()
        return result

    @classmethod
    def set_page_size(cls, pageno=1, pagesize=10):
        if isinstance(pageno, str):
            pageno = int(pageno)
        if pageno < 1:
            pageno = 1

        if isinstance(pagesize, str):
            pagesize = int(pagesize)
        if pagesize < 1:
            pagesize = 10
        return pageno, pagesize

    @classmethod
    def do_count(cls, entity):
        return entity.query.count()

    @classmethod
    def do_count_by_filter(cls, entity):
        return entity.query.filter(entity.filter > 1).count()

    @classmethod
    def do_get_list(cls, entity, pageno=1, pagesize=10, order_by='update_time'):
        pageno, pagesize = cls.set_page_size(pageno, pagesize)
        results = entity.query \
            .order_by(asc(order_by)).limit(pagesize) \
            .offset((pageno - 1) * pagesize)
        return [i.to_json() for i in results]

    @classmethod
    def do_query_all(cls, entity, pageno=1, pagesize=10, order_by='update_time',
                     select_name=None, select_target=None):
        pageno, pagesize = cls.set_page_size(pageno, pagesize)
        results = entity.query.filter(select_name == select_target).order_by(asc(order_by)).limit(pagesize) \
            .offset((pageno - 1) * pagesize)
        # 返回数据给控制器的时候需要转换数据，要不然会有编码问题
        outputs = [result.to_json() for result in results]
        total = entity.query.filter(select_name == select_target).count()
        page = Page(total, outputs)
        return page.to_json()

    @classmethod
    def do_get_detail(cls, entity, id):
        result = entity.query.get(id)
        return result.to_json()

    @classmethod
    def do_save_one(cls, data):
        if data:
            session = db.session
            try:
                session.add(data)
                session.commit()
                session.close()
                return True
            except Exception:
                logger.error('数据插入异常')
                session.rollback()
                return False
        return False

    # 未测试
    @classmethod
    def do_save_one_with_id_return(cls, data):
        if data:
            session = db.session
            try:
                session.add(data)
                session.flush()
                session.commit()
                id = data.id
                session.close()
                return id
            except Exception:
                logger.error('数据插入异常')
                session.rollback()
                return -1
        return -1
        # if data is not None:
        #     session = db.session
        #     session.add(data)
        #     session.flush()
        #     session.commit()
        #     session.close()
        #     return data.id
        # return -1

    @classmethod
    def do_update_one(cls, entity, source):
        if 'create_date' in source and source['create_date'] is not None:
            del source["create_date"]

        if 'update_time' in source and source['update_time'] is not None:
            source['update_time'] = datetime.now
        if source is not None:
            session = get_session(entity)
            query = session.query(entity).filter_by(id=source['id']).update(source)
            session.commit()
            session.close()
            return query
        return False

    @classmethod
    def do_delete_one(cls, entity, table_id):
        if table_id is not None:
            session = get_session(entity)
            query = session.query(entity).filter_by(id=table_id).delete()
            session.commit()
            session.close()
            return query
        return False

    @classmethod
    def do_query_by_sql_filter(cls, entity, columns, values):
        if isinstance(columns, str):
            column = columns

            if column is None:
                return []

            table_name = entity.__tablename__
            sql = text("SELECT * FROM " + table_name + " where " + column + "='" + values + "'")
        else:
            _if = []
            for column in columns:
                if column is None or not hasattr(entity, column):
                    return []
                _if.append(column + "='" + values + "'")
            table_name = entity.__tablename__
            sele = ' and '.join(_if)
            sql = text("SELECT * FROM " + table_name + " where " + sele)

        print(sql)

        session = get_session(entity)
        result = session.execute(sql)
        session.commit()
        session.close()
        return result

    @classmethod
    def do_count_by_group_column(cls, entity, group_column):
        session = get_session(entity)
        group_field = getattr(entity, group_column)
        if group_field is None is None:
            return []
        result = session.query(group_field, func.count('*')) \
            .group_by(group_field).all()
        return result

    @classmethod
    def do_count_by_group_column_time_scope(cls, entity, group_column, be, end):
        session = get_session(entity)
        group_field = getattr(entity, group_column)
        if group_field is None is None:
            return []
        result = session.query(group_field, func.count('*')) \
            .group_by(group_field).filter(entity.search_date.between(be, end)).all()
        return result
