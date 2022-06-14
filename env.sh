#!/usr/bin/bash

ENV=$1

if [ ! ${ENV} ]
then
    ENV=local
fi
# 项目根目录
export PYTHONPATH=./
# 当前环境 可选 dev pro
export FLASK_CONFIG=${ENV}