# 数据集，可根据需要增加英文或其它字符

SETS = [str(i) for i in range(10)] + \
       [chr(ord('a')+i) for i in range(0, 26)] + \
       [chr(ord('A')+i) for i in range(0, 26)] + \
       [',', '.', '?', ';', '"', '[', ']', '{', '}',
        '`', '~', '!', '@', '#', '$', '%', '^', '&',
        '*', '(', ')', '-', '=', '_', '+', '|', '\\',
        '/', '<', '>', ':', "'"]

# 分类数量
num_classes = len(SETS) + 1     # 数据集字符数+特殊标识符

# 图片大小，32 x 256
OUTPUT_SHAPE = (32, 256)

# 学习率
INITIAL_LEARNING_RATE = 1e-3
DECAY_STEPS = 5000
REPORT_STEPS = 100
LEARNING_RATE_DECAY_FACTOR = 0.9
MOMENTUM = 0.9

# LSTM网络层次
num_hidden = 128
num_layers = 2

# 训练轮次、批量大小
num_epochs = 50000
BATCHES = 10
BATCH_SIZE = 32
TRAIN_SIZE = BATCHES * BATCH_SIZE

# 数据集目录、模型目录
data_dir = "/tmp/lstm_ctc_data/"
model_dir = "/tmp/lstm_ctc_model/"