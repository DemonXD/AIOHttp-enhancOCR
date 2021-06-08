#################################
# Date: 2021/06/08
# Author: Miles Xu
# Email: kanonxmm@163.com
# Desc.: 读取数据
#################################
# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import cv2
from .const import OUTPUT_SHAPE, data_dir
from .label_vec import sparse_tuple_from


def get_file_text_array():
    """[summary]
        将文件和标签读取到内存，减少磁盘IO
    """
    file_name_array = []
    text_array = []
    for parent, dirname, filenames in os.walk(data_dir):
        file_name_array = filenames
    
    for f in file_name_array:
        text = f.split("_")[1]
        text_array.append(text)
    
    return file_name_array, text_array


def get_next_batch(file_name_array, text_array, batch_size=64):
    """[summary]
        获取训练的批量数据
    """
    inputs = np.zeros([batch_size, OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]])
    codes = []

    # 获取训练样本
    for idx in range(batch_size):
        index = random.randint(0, len(file_name_array) - 1)
        image = cv2.imread(data_dir + file_name_array[index])
        image = cv2.resize(image, (OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]), 3)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        text = text_array[index]
        #矩阵转置
        inputs[idx, :] = np.transpose(image.reshape((OUTPUT_SHAPE[0], OUTPUT_SHAPE[1])))
        codes.append(list(text))

    targets = [np.asarray(i) for i in codes]
    sparse_targets = sparse_tuple_from(targets)
    seq_len = np.ones(inputs.shape[0]) * OUTPUT_SHAPE[1]

    return inputs, sparse_targets, seq_len