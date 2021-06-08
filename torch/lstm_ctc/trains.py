#################################
# Date: 2021/06/08
# Author: Miles Xu
# Email: kanonxmm@163.com
# Desc.: 模型训练
#################################
# -*- coding: utf-8 -*-
import tensorflow as tf
from .label_vec import decode_sparse_tensor
from .const import (
    INITIAL_LEARNING_RATE, DECAY_STEPS,
    LEARNING_RATE_DECAY_FACTOR, BATCHES,
    BATCH_SIZE, REPORT_STEPS, TRAIN_SIZE,
    model_dir, num_epochs
)
from .load_data import get_file_text_array, get_next_batch
from .nets import get_train_model

# 准确性评估
# 输入：预测结果序列 decoded_list ,目标序列 test_targets
# 返回：准确率
def report_accuracy(decoded_list, test_targets):
    original_list = decode_sparse_tensor(test_targets)
    detected_list = decode_sparse_tensor(decoded_list)

    # 正确数量
    true_numer = 0

    # 预测序列与目标序列的维度不一致，说明有些预测失败，直接返回
    if len(original_list) != len(detected_list):
        print("len(original_list)", len(original_list), "len(detected_list)", len(detected_list),
              " test and detect length desn't match")
        return

    # 比较预测序列与结果序列是否一致，并统计准确率        
    print("T/F: original(length) <-------> detectcted(length)")
    for idx, number in enumerate(original_list):
        detect_number = detected_list[idx]
        hit = (number == detect_number)
        print(hit, number, "(", len(number), ") <-------> ", detect_number, "(", len(detect_number), ")")
        if hit:
            true_numer = true_numer + 1
    accuracy = true_numer * 1.0 / len(original_list)
    print("Test Accuracy:", accuracy)

    return accuracy

def train():
    # 获取训练样本数据
    file_name_array, text_array = get_file_text_array()

    # 定义学习率
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                               global_step,
                                               DECAY_STEPS,
                                               LEARNING_RATE_DECAY_FACTOR,
                                               staircase=True)
    # 获取网络结构
    logits, inputs, targets, seq_len, W, b = get_train_model()

    # 设置损失函数
    loss = tf.nn.ctc_loss(labels=targets, inputs=logits, sequence_length=seq_len)
    cost = tf.reduce_mean(loss)

    # 设置优化器
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)
    acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))

    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session() as session:
        session.run(init)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

        for curr_epoch in range(num_epochs):
            train_cost = 0
            train_ler = 0
            for batch in range(BATCHES):
                # 训练模型
                train_inputs, train_targets, train_seq_len = get_next_batch(file_name_array, text_array, BATCH_SIZE)
                feed = {inputs: train_inputs, targets: train_targets, seq_len: train_seq_len}
                b_loss, b_targets, b_logits, b_seq_len, b_cost, steps, _ = session.run(
                    [loss, targets, logits, seq_len, cost, global_step, optimizer], feed)

        # 评估模型
        if steps > 0 and steps % REPORT_STEPS == 0:
            test_inputs, test_targets, test_seq_len = get_next_batch(file_name_array, text_array, BATCH_SIZE)
            test_feed = {inputs: test_inputs,targets: test_targets,seq_len: test_seq_len}
            dd, log_probs, accuracy = session.run([decoded[0], log_prob, acc], test_feed)
        
        report_accuracy(dd, test_targets)

        # 保存识别模型
        save_path = saver.save(session, model_dir + "lstm_ctc_model.ctpk",global_step=steps)

        c = b_cost
        train_cost += c * BATCH_SIZE

        train_cost /= TRAIN_SIZE
        # 计算 loss
        train_inputs, train_targets, train_seq_len = get_next_batch(file_name_array, text_array, BATCH_SIZE)
        val_feed = {inputs: train_inputs,targets: train_targets,seq_len: train_seq_len}
        val_cost, val_ler, lr, steps = session.run([cost, acc, learning_rate, global_step], feed_dict=val_feed)

        log = "{} Epoch {}/{}, steps = {}, train_cost = {:.3f}, val_cost = {:.3f}"
        print(log.format(curr_epoch + 1, num_epochs, steps, train_cost, val_cost))