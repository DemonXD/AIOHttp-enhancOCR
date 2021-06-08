#################################
# Date: 2021/06/08
# Author: Miles Xu
# Email: kanonxmm@163.com
# Desc.: 项目封装
#################################
# -*- coding: utf-8 -*-
import tensorflow as tf
import cv2
import numpy as np
from .nets import get_train_model
from .const import model_dir, OUTPUT_SHAPE
from .label_vec import decode_sparse_tensor


# LSTM+CTC 文字识别能力封装
# 输入：图片
# 输出：识别结果文字
def predict(image):

    # 获取网络结构
    logits, inputs, targets, seq_len, W, b = get_train_model()
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 加载模型
        saver.restore(sess, tf.train.latest_checkpoint(model_dir))
        # 图像预处理
        image = cv2.resize(image, (OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]), 3)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        pred_inputs = np.zeros([1, OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]])
        pred_inputs[0, :] = np.transpose(image.reshape((OUTPUT_SHAPE[0], OUTPUT_SHAPE[1])))
        pred_seq_len = np.ones(1) * OUTPUT_SHAPE[1]
        # 模型预测
        pred_feed = {inputs: pred_inputs,seq_len: pred_seq_len}
        dd, log_probs = sess.run([decoded[0], log_prob], pred_feed)
        # 识别结果转换
        detected_list = decode_sparse_tensor(dd)[0]
        detected_text = ''
        for d in detected_list:
            detected_text = detected_text + d

    return detected_text