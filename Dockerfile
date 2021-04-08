# 指定所创建镜像的基础镜像
FROM python:3.8-slim-buster as base

COPY requirements.txt ./

# 安装依赖
RUN apt-get update && \
    apt-get install -y apt-utils libgomp1 libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/* && \
    pip3 install -i https://pypi.doubanio.com/simple/ --upgrade pip --no-cache-dir && \
    pip3 install -i https://pypi.doubanio.com/simple/ -r requirements.txt --no-cache-dir && \
    rm requirements.txt

# 阶段构建，节省时间
from base

WORKDIR /var/www/ocr
COPY . .

ENV LANG C.UTF-8

# 声明镜像内服务监听的端口
EXPOSE 8888
