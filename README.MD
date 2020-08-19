# 使用TensorFlow进行中文情感分析
本文通过TensorFlow中的LSTM神经网络方法进行中文情感分析
需要依赖的库
* numpy
* jieba
* gensim
* tensorflow
* matplotlib
* sklearn
## 1.导入依赖包
```
# 导包
import re
import os

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jieba
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split

```
## 2.加载词向量
可以使用自己的词向量，也可以使用网上开源的词向量我这里使用的是北京师范大学中文信息处理研究所与中国人民大学DBIIR实验室的研究者开源的"chinese-word-vectors"
github链接为： https://github.com/Embedding/Chinese-Word-Vectors
![](https://img2020.cnblogs.com/blog/2008908/202008/2008908-20200814144833800-1632781547.png)
选择一个下载到本地，放到项目下的vectors目录下
```
# 使用gensim加载预训练中文分词，需要等待一段时间
cn_model = KeyedVectors.load_word2vec_format('vectors/sgns.zhihu.bigram',
                                            binary=False, unicode_errors='ignore')
```
## 3.读取训练语料
我这里使用的是谭松波老师的酒店评论语料
链接：https://pan.baidu.com/s/14hhPRrD96mkBQEbc_7Hqag  提取码：cx5q 
训练样本分别被放置在这两个文件夹里面pos和neg，每个文件夹里面有3000个txt文件，每个文件内有一段评语，共有6000个训练样本
下载好后，把pos和neg目录解压到data目录下
```
## 4.读取训练数据
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

```
## 5.生成对应的标签
```
x_train = np.array(x_train)
y_train = np.concatenate((np.ones(3000), np.zeros(3000)))  # 生成标签

# 打乱训练样本和标签的顺序
np.random.seed(116)
np.random.shuffle(x_train)

np.random.seed(116)
np.random.shuffle(y_train)

```

## 6.对训练数据进行分词操作
```
x_train_tokens = []
for text in x_train:
    # 使用jieba进行分词
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
```
## 7.索引长度标准化
因为每段评语的长度是不一样的，我们如果单纯取最长的一个评语，并把其他评语填充成同样的长度，这样十分浪费计算资源，所以我们去一个折衷的长度
```
# 获取每段语句的长度，并画图展示
tokens_count = [len(tokens) for tokens in x_train_tokens]
tokens_count.sort(reverse=True)
# 画图查看词的长度分布
plt.plot(tokens_count)
plt.ylabel('tokens count')
plt.xlabel('tokens length')
plt.show()
# 可以看出大部分词的长度都是在500以下的
```
![](https://img2020.cnblogs.com/blog/2008908/202008/2008908-20200814150204316-522860734.png)

```
# 当tokens长度分布满足正态分布的时候，
# 可以使用 取tokens的平均值并且加上两个tokens的标准差，来选用tokens的长度
tokens_length = np.mean(tokens_count) + 2 * np.std(tokens_count)
print(tokens_length)
输出：297.3980831340084
```
```
# 可以看到当tokens的长度为297.3980831340084，大约95%的样本被覆盖，
# 我们需要对长度不足的tokens进行padding，超长的进行修剪
np.sum(tokens_count < tokens_length) / len(tokens_count)
输出：0.9545
```
## 8.定义把tokens转换回文本的方法
```
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
# 测试
print(reverse_tokens(x_train_tokens[0]))
print(y_train[0])
# 输出：酒店的服务简直不好程度排第一！我住1702房间漏风。调整到1710房间！这个房间的空调是坏的！半夜给了。调整到房间明显比1702和1710房间小的很多而且房间不能够上网我已经被折磨的没有力气在调整房间了。想躺在床上看看电视我的天啊！看不了！总之我对这次携程的服务比较满意对该辽宁省沈阳市城市酒店提供的客房服务是特别特别特别的不满意！我希望携程能够好好考虑一下自己的加盟酒店总是这样我们还怎么相信携程这样的品牌。总体来说我很郁闷！也特别的伤心！
# 0.0

```
## 9.准备Embedding Matrix
现在我们来为模型准备词向量embedding matrix（词向量矩阵），根据keras的要求，我们需要准备一个维度为(numwords, 300)的矩阵，每一个词汇都用一个长度为300的向量表示。

注意我们只选择使用前50k个使用频率最高的词，在这个预训练词向量模型中，一共有260万词汇量，如果全部使用在分类问题上会很浪费计算资源，因为我们的训练样本很小，一种只用6k，如果我们有100k、200k甚至更多的训练样本时，在分类问题上可以考虑减少使用词汇量
```
embedding_matrix = np.zeros((50000, 300))
for i in range(50000):
    embedding_matrix[i, :] = cn_model[cn_model.index2word[i]]
    pass
embedding_matrix = embedding_matrix.astype('float32')
# 检查index是否对应
# 输出300意义为长度为300的embedding向量一一对应
print(np.sum(cn_model[cn_model.index2word[300]] == embedding_matrix[300]))
# 输出：300
```
## 10.对训练样本进行padding(填充)和truncating(修剪)
我们把文本转换为tokens（索引）之后，每一串索引的长度并不相等，所以为了方便模型的训练，我们需要把索引的长度标准化，上面我们选择了297这个可以覆盖95%训练样本的长度，接下来我们进行padding和truncating，我们一般采用'pre'的方法，这会在文本索引的前面填充0，因为根据一些研究材料中的实践，如果在文本索引后填充0的话，会对模型造成一些不良影响
```
x_train_tokens_pad = tf.keras.preprocessing.sequence.pad_sequences(x_train_tokens,
                                                                  maxlen=int(tokens_length),
                                                                  padding='pre',
                                                                  truncating='pre')
# 超出五万个词向量的词用0代替
train_pad[train_pad >= num_words] = 0
```
```
# 可见padding之后前面的tokens全变成0，文本在最后面
print(train_pad[33])
```
![](https://img2020.cnblogs.com/blog/2008908/202008/2008908-20200814151612091-259952856.png)

## 11.使用sklearn的train_test_split进行拆分训练集和测试集
```
# 使用90%进行训练，10%进行测试

x_tokens_train, x_tokens_test, y_tokens_train, y_tokens_test = train_test_split(
    x_train_tokens_pad,
    y_train,
    test_size=0.1,
    random_state=12
)
# 也可手动拆分
# x_tokens_train = x_train_tokens_pad[:-int(x_train_tokens_pad.shape[0] / 10)]
# x_tokens_test = x_train_tokens_pad[-int(x_train_tokens_pad.shape[0] / 10):]
# y_tokens_train = y_train[:-int(y_train.shape[0] / 10)]
# y_tokens_test = y_train[-int(y_train.shape[0] / 10):]
```
## 12.构建模型
```
# 构建模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(50000,300,
                                         weights=[embedding_matrix],
                                         input_length=int(tokens_length),
                                         trainable=False                                        
                                        ),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=64, return_sequences=True)),
    tf.keras.layers.LSTM(16, return_sequences=False),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])
model.compile(optimizer='Adam',
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy']
             )
```

## 13.进行训练
```
# 训练20轮，每轮都进行测试集的验证，使用1%用来测试集，每批128
history = model.fit(x_tokens_train, 
          y_tokens_train,
          batch_size=128,
          epochs=20, 
          validation_split=0.1,
          validation_freq=1
         )
```
![](https://img2020.cnblogs.com/blog/2008908/202008/2008908-20200814152128572-1532860348.png)

## 14.查看网络结构
```
model.summary()
```
![](https://img2020.cnblogs.com/blog/2008908/202008/2008908-20200814152359194-962396710.png)

# 15.使用测试集，验证模型准确率
```
result = model.evaluate(x_tokens_test, y_tokens_test)
print(f'Accuracy : {result[1]}')
```
可以看出我们模型的准确率达到了90%
![](https://img2020.cnblogs.com/blog/2008908/202008/2008908-20200814154252772-430694185.png)

## 16.画出训练的loss（训练误差）、val_loss（测试误差）和sparse_categorical_accuracy（训练准确度）、val_sparse_categorical_accuracy（测试准确度）
```
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
```
![](https://img2020.cnblogs.com/blog/2008908/202008/2008908-20200814154408449-1731531790.png)

## 17.准备一些文本，用训练好的模型进行测试
```
def predict_sentiment(text):
    
    print(text)
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
'房间很温馨，前台服务很好,'
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
```
![](https://img2020.cnblogs.com/blog/2008908/202008/2008908-20200814154835098-1556354728.png)
