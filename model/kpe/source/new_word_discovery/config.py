#!/usr/bin/env python
# encoding: utf-8

import os

basedir = os.path.abspath(os.path.dirname(__file__))
dict_dir = basedir + '/data/dict.txt'
root_dir = basedir + "/data/root.pkl"
stopword_dir = basedir+'/data/stopword.txt'


class Config(object):
    DEBUG = False
    TESTING = False
    REQUEST_STATS_WINDOW = 15


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False


class TestingConfig(Config):
    TESTING = True


config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig
}
