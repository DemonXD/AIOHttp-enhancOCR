#################################
# Date: 2021/06/07
# Author: Miles Xu
# Email: kanonxmm@163.com
# Desc.: 标签向量化(稀疏矩阵)
# pic_1 : 29
# pic_2 : 6836
# pic_3 : 682
# pic_4 : 586825289
# pic_5 : 1
# pic_6 : 71147
# 向量化后：
# indices = [
#   [0, 0], [0, 1],
#   [1, 0], [1, 1], [1, 2], [1, 3],
#   [2, 0], [2, 1], [2, 2],
#   [3, 0], [3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7], [3, 8]
#   [4, 0], 
#   [5, 0], [5, 1],[5, 2], [5, 3], [5, 4],
# ]
# values = [2, 9, 6, 8, 3, 6, 6, 8, 2, 5, 8, 5, 8, 2, 5, 2, 8, 9, 1, 7, 1, 1, 4, 7]
# shape = [6, 9]
#################################
# -*- coding: utf-8 -*-
import numpy as np
from .const import SETS

def sparse_tuple_from(sequences, dtype=np.int32):
    indices = []
    values = []

    for idx, seq in enumerate(sequences):
        indices.extend(zip([idx] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape


def decode_sparse_tensor(sparse_tensor):
    decoded_indexes = []
    current_idx = 0
    current_seq = []

    for offset, i_and_idx in enumerate(sparse_tensor[0]):
        i = i_and_idx
        if i != current_idx:
            decoded_indexes.append(current_seq)
            current_idx = i
            current_seq = []
        current_seq.append(offset)
    decoded_indexes.append(current_seq)
    result = []
    for idx in decoded_indexes:
        result.append(decode_a_seq(idx, sparse_tensor))
    return result


def decode_a_seq(idx, spars_tensor):
    decoded = []
    for each in idx:
        strs = SETS[spars_tensor[1][each]]
        decoded.append(strs)
    return decoded