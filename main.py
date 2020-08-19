# 导包
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import jieba
from gensim.models import KeyedVectors

import re
import os

# 使用gensim加载预训练中文分词，需要等待一段时间
cn_model = KeyedVectors.load_word2vec_format('vectors/sgns.zhihu.bigram',
                                            binary=False, unicode_errors='ignore')
# 读取训练数据
pos_file_list = os.listdir('data/pos')
neg_file_list = os.listdir('data/neg')
pos_file_list = [f'data/pos/{x}' for x in pos_file_list]
neg_file_list = [f'data/neg/{x}' for x in neg_file_list]
pos_neg_file_list = pos_file_list + neg_file_list

# 读取所有的文本，放入到x_train,前3000是正向样本，后3000负向样本
x_train = []
for file in pos_neg_file_list:
    with open(file, 'r', encoding='utf-8') as f:
        text = f.read().strip()
        pass
    x_train.append(text)
    pass


x_train = np.array(x_train)
y_train = np.concatenate((np.ones(3000), np.zeros(3000)))  # 生成标签

# 打乱训练样本和标签的顺序
np.random.seed(116)
np.random.shuffle(x_train)

np.random.seed(116)
np.random.shuffle(y_train)
# 进行分词操作
x_train_tokens = []
for text in x_train:
    # 去掉标点
    # text = re.sub("[\s+.!/_,$%^(+\"']+|[+——！，。？、~@#￥%……&（）]+", "", text)
    cut = jieba.cut(text)
    cut_list = [x for x in cut]
    for i,word in enumerate(cut_list):
        try:
            # 将词转换为索引index
            cut_list[i] = cn_model.vocab[word].index
            pass
        except KeyError:
            # 如果词不在字典中，则输出0
            cut_list[i] = 0
            pass
        pass
    x_train_tokens.append(cut_list)
    pass

# 索引长度标准化
# 因为每段评语的长度是不一样的，我们如果单纯取最长的一个评语，并把其他评语填充成同样的长度，
# 这样十分浪费计算资源，所以我们去一个折衷的长度
tokens_count = [len(tokens) for tokens in x_train_tokens]
tokens_count.sort(reverse=True)

# 画图查看词的长度分布
plt.plot(tokens_count)
plt.ylabel('tokens count')
plt.xlabel('tokens length')
plt.show()
# 可以看出大部分词的长度都是在500以下的

# 当tokens长度分布满足正态分布的时候，
# 可以使用 取tokens的平均值并且加上两个tokens的标准差，来选用tokens的长度
tokens_length = np.mean(tokens_count) + 2 * np.std(tokens_count)
print(tokens_length)

# 可以看到当tokens的长度为244，大约95%的样本被覆盖，
# 我们需要对长度不足的tokens进行padding，超长的进行修剪
np.sum(tokens_count < tokens_length) / len(tokens_count)

# 定义一个把tokens转换成文本的方法
def reverse_tokens(tokens):
    text = ''
    for index in tokens:
        if index != 0:
            text = text + cn_model.index2word[index]
        else:
            text = text + ''
        pass
    return text
    pass

print(reverse_tokens(x_train_tokens[0]))
print(y_train[0])

embedding_matrix = np.zeros((50000, 300))
for i in range(50000):
    embedding_matrix[i, :] = cn_model[cn_model.index2word[i]]
    pass
embedding_matrix = embedding_matrix.astype('float32')


x_train_tokens_pad = tf.keras.preprocessing.sequence.pad_sequences(x_train_tokens,
                                                                  maxlen=int(tokens_length),
                                                                  padding='pre',
                                                                  truncating='pre')

x_train_tokens_pad[x_train_tokens_pad >= 50000] = 0
# x_train_tokens_pad.shape
np.sum(cn_model[cn_model.index2word[300]] == embedding_matrix[300])


# 使用90%进行训练，10%进行测试
# x_tokens_train = x_train_tokens_pad[:-int(x_train_tokens_pad.shape[0] / 10)]
# x_tokens_test = x_train_tokens_pad[-int(x_train_tokens_pad.shape[0] / 10):]
# y_tokens_train = y_train[:-int(y_train.shape[0] / 10)]
# y_tokens_test = y_train[-int(y_train.shape[0] / 10):]
from sklearn.model_selection import train_test_split
x_tokens_train, x_tokens_test, y_tokens_train, y_tokens_test = train_test_split(
    x_train_tokens_pad,
    y_train,
    test_size=0.1,
    random_state=12
)

print(x_tokens_train.shape)
print(embedding_matrix.shape)

# 构建模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(50000,300,
                              weights=[embedding_matrix],
                              input_length=int(tokens_length),
                              trainable=False,
                                        ),
#     tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            units=64, return_sequences=True
        )),
    tf.keras.layers.LSTM(32, return_sequences=False),
#     tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])
model.compile(optimizer='Adam',
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy']
             )
# , kernel_regularizer=tf.keras.regularizers.l2(0.01)

print(model.summary())
history = model.fit(x_tokens_train,
          y_tokens_train,
          batch_size=128,
          epochs=40,
          validation_split=0.1,
          validation_freq=1
         )
print(model.summary())

result = model.evaluate(x_tokens_test, y_tokens_test)
print(f'Accuracy : {result[1]}')

# history.history['loss']
# history.history['val_loss']
# history.history['sparse_categorical_accuracy']
# history.history['val_sparse_categorical_accuracy']
plt.plot(history.history['loss'],label="$Loss$")
plt.plot(history.history['val_loss'],label='$val_loss$')
plt.title('Loss')
plt.xlabel('epoch')
plt.ylabel('num')
plt.legend()
plt.show()
plt.plot(history.history['sparse_categorical_accuracy'],label="$sparse_categorical_accuracy$")
plt.plot(history.history['val_sparse_categorical_accuracy'],label='$val_sparse_categorical_accuracy$')
plt.title('Accuracy')
plt.xlabel('epoch')
plt.ylabel('num')
plt.legend()
plt.show()


def predict_sentiment(text):
    print(text)
    text = re.sub("[\s+.!/_,$%^(+\"'\”\“]+|[+——！，。？、~@#￥%……&（）]+", "", text)
    # 分词
    cut = jieba.cut(text)
    cut_list = [x for x in cut]
    for i, word in enumerate(cut_list):
        try:
            cut_list[i] = cn_model.vocab[word].index
        except KeyError:
            cut_list[i] = 0
        pass
    # padding
    tokens_pad = tf.keras.preprocessing.sequence.pad_sequences([cut_list],
                                                               maxlen=int(tokens_length),
                                                               padding='pre',
                                                               truncating='pre')
    return tokens_pad
    pass

test_list = [
'酒店设施不是新的，服务态度很不好',
'酒店卫生条件非常不好',
'床铺非常舒适',
'房间很冷，还不给开暖气',
'房间很凉爽，空调冷气很足',
'酒店环境不好，住宿体验很不好',
'房间隔音不到位' ,
'晚上回来发现没有打扫卫生,心情不好',
'因为过节所以要我临时加钱，比团购的价格贵',
'房间很温馨，前台服务很好,',
'国庆出游人比较多，各个景点都是，这个时候能欣赏到风景就不错了。行程整体比较满意，希望酒店可以稍微离市区近点就更完美了'
]

for text in test_list:
    try:
        tokens_pad = predict_sentiment(text)
        result = model.predict(x=tokens_pad)
        print(result)
        if result[0][0] <= result[0][1]:
            print(f'正:{result[0][1]}')
        else:
            print(f'负:{result[0][0]}')
    except Exception as ex:
        print(ex.args)
        pass
    pass