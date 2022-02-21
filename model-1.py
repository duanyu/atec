# df1 df2 df3 df4类型为: pandas.core.frame.DataFrame.分别引用输入桩数据
# topai(1, df1)函数把df1内容写入第一个输出桩

# coding: utf-8
#该代码去掉了一些冗余无用的部分，对结果没有任何影响。

from keras import backend as K
from keras.engine.topology import Layer
from keras.activations import softmax
import os
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score

#由于模型使用了10fold-cv，所以用一个flag变量表示运行的是哪个fold
#每次进行不同的fold实验的时候都需要更改这个变量
fold = 3

#实验所用自动调整LR的学习率策略
class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    For more detail, please see paper.

    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```

    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=1e-5, max_lr=1e-2, step_size=1000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2. ** (x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** (x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(
                self.clr_iterations)

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, epoch, logs=None):

        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr())

#实验所用mult-head attention组件
class Attention(Layer):
    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head * size_per_head
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Attention, self).build(input_shape)

    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:, 0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape) - 2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12

    def call(self, x):
        # 如果只传入Q_seq,K_seq,V_seq，那么就不做Mask
        # 如果同时传入Q_seq,K_seq,V_seq,Q_len,V_len，那么对多余部分做Mask
        if len(x) == 3:
            Q_seq, K_seq, V_seq = x
            Q_len, V_len = None, None
        elif len(x) == 5:
            Q_seq, K_seq, V_seq, Q_len, V_len = x
        # 对Q、K、V做线性变换
        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0, 2, 1, 3))
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0, 2, 1, 3))
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0, 2, 1, 3))
        # 计算内积，然后mask，然后softmax
        A = K.batch_dot(Q_seq, K_seq, axes=[3, 3]) / self.size_per_head ** 0.5
        A = K.permute_dimensions(A, (0, 3, 2, 1))
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0, 3, 2, 1))
        A = K.softmax(A)
        # 输出并mask
        O_seq = K.batch_dot(A, V_seq, axes=[3, 2])
        O_seq = K.permute_dimensions(O_seq, (0, 2, 1, 3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)

    def get_config(self):
        config = {'nb_head': self.nb_head, 'size_per_head': self.size_per_head}
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

#mult head pool
#mult head attention之后使用，将每个head分开进行pooling操作
def multhead_pool(input_x, head_num, head_size, mlps):
    '''return the mult-head s2t pool(bs,h) of input matrix(bs, max_len, h)
    mlps means the list of dense layer to do pooling'''
    multhead_pool = []
    for i in range(head_num):
        mlp = mlps[i]
        onehead_x = Lambda(lambda x: x[:, :, i * head_size:(i + 1) * head_size])(input_x)
        # 沿着每一维求softmax
        s2t_w = Lambda(lambda x: softmax(x, axis=1))(mlp(onehead_x))
        onehead_pool = Lambda(lambda x: K.sum(x, axis=1))(Multiply()([onehead_x, s2t_w]))
        multhead_pool.append(onehead_pool)
    result_pool = Concatenate()(multhead_pool)
    return result_pool


from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
import sys
import jieba
import pandas as pd
import tensorflow as tf
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.models import *
import keras.backend as K
from keras.layers import *
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.optimizers import Adam

#参数
max_len = 30
emb_size = 300
lstm_size = 150
epochs = 4
mlp1_size = 300
num_head = 5
head_size = 80
bs = 256
beta1 = 0.9
lr = 2e-3
dp = 0.2
dp_emb = 0.2

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)

train = df1
val = df2
fulldata = df4

#获取train和val的数据
q1_train = train['sent1'].apply(lambda x: ' '.join([y for y in x])).values
q2_train = train['sent2'].apply(lambda x: ' '.join([y for y in x])).values
y_train = np.array(train['label'].values, dtype='int32')

q1_val = val['sent1'].apply(lambda x: ' '.join([y for y in x])).values
q2_val = val['sent2'].apply(lambda x: ' '.join([y for y in x])).values
y_val = np.array(val['label'].values, dtype='int32')

#获取全部训练集的数据，主要目的是得到每个fold共有的tokenizer
full1 = fulldata['sent1'].apply(lambda x: ' '.join([y for y in x])).values
full2 = fulldata['sent2'].apply(lambda x: ' '.join([y for y in x])).values

total_full = np.concatenate((full1, full2))

tokenizer = Tokenizer(filters='')
tokenizer.fit_on_texts(total_full)
print('the number of unique token is', len(tokenizer.word_index))

#对train和val的数据进行基本处理
q1_train = tokenizer.texts_to_sequences(q1_train)
q2_train = tokenizer.texts_to_sequences(q2_train)
q1_val = tokenizer.texts_to_sequences(q1_val)
q2_val = tokenizer.texts_to_sequences(q2_val)

q1_train_len = np.array([len(_q) for _q in q1_train])
q1_train = pad_sequences(q1_train, maxlen=max_len, padding='post')

q1_val_len = np.array([len(_q) for _q in q1_val])
q1_val = pad_sequences(q1_val, maxlen=max_len, padding='post')

q2_train_len = np.array([len(_q) for _q in q2_train])
q2_train = pad_sequences(q2_train, maxlen=max_len, padding='post')

q2_val_len = np.array([len(_q) for _q in q2_val])
q2_val = pad_sequences(q2_val, maxlen=max_len, padding='post')

#每一轮得到F1值的callback
class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(
            [np.array(self.validation_data[0]), np.array(self.validation_data[1]), np.array(self.validation_data[2]),
             np.array(self.validation_data[3])]))).round()
        val_targ = np.array(self.validation_data[4])
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        print('val_f1', _val_f1, 'val_recall', _val_recall, 'val_precision', _val_precision)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        return

metrics = Metrics()

#模型所用函数部分
def unchanged_shape(input_shape):
    "Function for Lambda layer"
    return input_shape


def soft_attention_alignment(input_1, input_2):
    "Align text representation with neural soft attention"
    attention = Dot(axes=-1)([input_1, input_2])
    w_att_1 = Lambda(lambda x: K.softmax(x, axis=1),
                     output_shape=unchanged_shape)(attention)
    w_att_2 = Permute((2, 1))(Lambda(lambda x: K.softmax(x, axis=2),
                                     output_shape=unchanged_shape)(attention))
    in1_aligned = Dot(axes=1)([w_att_1, input_1])
    in2_aligned = Dot(axes=1)([w_att_2, input_2])
    return in1_aligned, in2_aligned

#对两个向量，根据正交分解公式求相关以及不相关部分，分别为sim和dissim
def od_compare(input_1, input_2):
    "od compare of vector 1 and 2, which determines the similarity part and dissimiliarty part of vector"
    # input_1's shape is (bs, max_len, hidden_units)
    dot = Lambda(lambda x: K.sum(x[0] * x[1], axis=-1, keepdims=True))
    dot_12 = dot([input_1, input_2])
    dot_22 = dot([input_2, input_2])
    div = Lambda(lambda x: x[0] / x[1])([dot_12, dot_22])
    sim = Multiply()([div, input_2])
    dissim = Lambda(lambda x: x[0] - x[1])([input_1, sim])
    sim_mlp = mlp_sim(sim)
    sim = Add()([sim_mlp, sim])
    dissim_mlp = mlp_dissim(dissim)
    dissim = Add()([dissim_mlp, dissim])
    return sim, dissim

#读取预训练emb
def read_pre_emb(df_input, num_word):
    print('Indexing word vectors.')
    embeddings_index = {}
    words = df_input['word']
    vectors = df_input['vector']
    #   vectors = df_input.iloc[:,1:].values

    for index, word in enumerate(words):
        vec = vectors[index]
        #     vec = np.array(vec, dtype='float32')
        vec = np.array(vec.split(' '), dtype='float32')
        embeddings_index[word] = vec

    embedding_matrix = np.random.uniform(low=-0.05, high=0.05, size=(num_word, 300))
    for word, i in tokenizer.word_index.items():
        if i >= num_word:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


num_word = len(tokenizer.word_index) + 1
print('num of unique words', num_word)
pretrain_emb = read_pre_emb(df3, num_word)

# build DL model
# DL模型主体部分
x1 = Input(shape=(max_len,))
x2 = Input(shape=(max_len,))

x1_len = Input(shape=(1,), dtype='int32')
x2_len = Input(shape=(1,), dtype='int32')

emb_layer = Embedding(input_dim=num_word, output_dim=emb_size, name='emb_layer', weights=[pretrain_emb])
emb_x1 = emb_layer(x1)
emb_x2 = emb_layer(x2)

# bi-GRU model
lstm = Bidirectional(GRU(lstm_size, return_sequences=True, name='lstm', implementation=2))
lstm_x1 = lstm(emb_x1)
lstm_x2 = lstm(emb_x2)

# sigmoid(w1x1+w2x2+b)
# word vector和bi-GRU后的vector进行了gate加操作
x_w_mlp = Dense(lstm_size * 2, activation=None, use_bias=False)
x_w_b_mlp = Dense(lstm_size * 2, activation=None, use_bias=True)

x1_w = x_w_mlp(emb_x1)
x1_w_b = x_w_b_mlp(lstm_x1)
x1_gate = Activation('sigmoid')(Add()([x1_w, x1_w_b]))
inv_x1_gate = Lambda(lambda x: K.ones_like(x) - x)(x1_gate)
lstm_x1 = Add()([Multiply()([x1_gate, emb_x1]), Multiply()([inv_x1_gate, lstm_x1])])

x2_w = x_w_mlp(emb_x2)
x2_w_b = x_w_b_mlp(lstm_x2)
x2_gate = Activation('sigmoid')(Add()([x2_w, x2_w_b]))
inv_x2_gate = Lambda(lambda x: K.ones_like(x) - x)(x2_gate)
lstm_x2 = Add()([Multiply()([x2_gate, emb_x2]), Multiply()([inv_x2_gate, lstm_x2])])


# interaction
# 两个句子间的交互
x1_align, x2_align = soft_attention_alignment(lstm_x1, lstm_x2)

mlp_sim = Dense(lstm_size * 2, activation='relu')
mlp_dissim = Dense(lstm_size * 2, activation='relu')
x1_sim, x1_dissim = od_compare(lstm_x1, x2_align)
x2_sim, x2_dissim = od_compare(lstm_x2, x1_align)

#对相关／不相关部分的交互信息进行Attention建模，抓取长距离关系
sim_att = Attention(num_head, head_size, name='sim_att')
dissim_att = Attention(num_head, head_size, name='dissim_att')

x1_sim_att = sim_att([x1_sim, x1_sim, x1_sim, x1_len, x1_len])
x2_sim_att = sim_att([x2_sim, x2_sim, x2_sim, x2_len, x2_len])

x1_dissim_att = dissim_att([x1_dissim, x1_dissim, x1_dissim, x1_len, x1_len])
x2_dissim_att = dissim_att([x2_dissim, x2_dissim, x2_dissim, x2_len, x2_len])

mh_sim_mlps = []
for i in range(num_head):
    mh_sim_mlps.append(Dense(head_size, activation='relu'))

mh_dissim_mlps = []
for i in range(num_head):
    mh_dissim_mlps.append(Dense(head_size, activation='relu'))

#对相关、不相关部分进行pooling
x1_sim_pool = multhead_pool(x1_sim_att, num_head, head_size, mh_sim_mlps)
x2_sim_pool = multhead_pool(x2_sim_att, num_head, head_size, mh_sim_mlps)
sim_pool = Concatenate()([x1_sim_pool, x2_sim_pool])

x1_dissim_pool = multhead_pool(x1_dissim_att, num_head, head_size, mh_dissim_mlps)
x2_dissim_pool = multhead_pool(x2_dissim_att, num_head, head_size, mh_dissim_mlps)
dissim_pool = Concatenate()([x1_dissim_pool, x2_dissim_pool])

#对pooling结果过一层Relu，外加残差
sim_pool_mlp = Dense(num_head * head_size * 2, activation='relu')(sim_pool)
sim_pool = Add()([sim_pool_mlp, sim_pool])

dissim_pool_mlp = Dense(num_head * head_size * 2, activation='relu')(dissim_pool)
dissim_pool = Add()([dissim_pool_mlp, dissim_pool])

#对相关／不相关信息进行gate加操作
w_sim = Dense(num_head * head_size * 2, activation=None, use_bias=False)(sim_pool)
w_b_dissim = Dense(num_head * head_size * 2, activation=None, use_bias=True)(dissim_pool)
sim_gate = Activation('sigmoid')(Add()([w_sim, w_b_dissim]))
inv_sim_gate = Lambda(lambda x: K.ones_like(x) - x)(sim_gate)

combine = Add()([Multiply()([sim_gate, sim_pool]), Multiply()([inv_sim_gate, dissim_pool])])
combine = Dropout(dp)(combine)

#过两层MLP输出0-1的结果
mlp = Dense(mlp1_size, activation='relu')(combine)
output = Dense(1, activation='sigmoid')(mlp)

#compile模型
model = Model(inputs=(x1, x2, x1_len, x2_len), outputs=output)
model.compile(optimizer=Adam(lr, beta_1=beta1),
              loss='binary_crossentropy',
              metrics=['accuracy'])

#模型训练部分
clr = CyclicLR(base_lr=lr / 10, max_lr=lr, step_size=2 * len(q1_train) // bs, mode='triangular')
model.fit(x=[q1_train, q2_train, q1_train_len, q2_train_len],
          y=y_train,
          batch_size=bs,
          epochs=epochs,
          verbose=2,
          validation_data=([q1_val, q2_val, q1_val_len, q2_val_len], y_val),
          shuffle=True,
          class_weight={0: 0.8081, 1: 1.3115},#针对训练集不平衡现象进行加权
          callbacks=[clr, metrics]
          )
model.save_weights(model_dir + "extend-esim-plus-resmlp-test-" + str(fold) + ".h5")