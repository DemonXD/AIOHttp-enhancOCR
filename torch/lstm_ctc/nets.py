#################################
# Date: 2021/06/08
# Author: Miles Xu
# Email: kanonxmm@163.com
# Desc.: 构建lstm-ctc 网络
#################################
# -*- coding: utf-8 -*-
import tensorflow as tf
from .const import OUTPUT_SHAPE, num_hidden, num_layers, num_classes


def get_train_model():
    # 输入
    inputs = tf.placeholder(tf.float32, [None, None, OUTPUT_SHAPE[0]]) 

    # 稀疏矩阵
    targets = tf.sparse_placeholder(tf.int32)

    # 序列长度 [batch_size,]
    seq_len = tf.placeholder(tf.int32, [None])

    # 定义LSTM网络
    cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
    stack = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)      # old
    outputs, _ = tf.nn.dynamic_rnn(cell, inputs, seq_len, dtype=tf.float32)
    shape = tf.shape(inputs)
    batch_s, max_timesteps = shape[0], shape[1]

    outputs = tf.reshape(outputs, [-1, num_hidden])
    W = tf.Variable(tf.truncated_normal([num_hidden,
                                        num_classes],
                                        stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0., shape=[num_classes]), name="b")
    logits = tf.matmul(outputs, W) + b
    logits = tf.reshape(logits, [batch_s, -1, num_classes])

    # 转置矩阵
    logits = tf.transpose(logits, (1, 0, 2))

    return logits, inputs, targets, seq_len, W, b 