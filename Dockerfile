FROM python:3.8
WORKDIR /tpidentify

COPY requirements.txt ./
RUN apt-get install libsm6
RUN apt-get install libxrender1
RUN apt-get install libxext-dev
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install -r requirements.txt

COPY . .
RUN export FLASK_CONFIG=pro
CMD ["gunicorn", "manager:app", "-c", "./gunicorn.conf.py"]
