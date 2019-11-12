import pandas as pd
import re
import math

import tensorflow as tf

print ('tf',tf.__version__)

import tensorflow_hub as hub

print ('tf_hub',hub.__version__)

from keras import backend as K

from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from keras.models import Model, Input
from keras.layers.merge import add
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Lambda

import numpy as np


data = pd.read_csv("home/ubuntu/sentences_eng.csv")

data.columns = ['unknown','Sno','Language','Text']
data = data[data['Language']=='eng']
data = data[['Sno','Language','Text']]

data['Sno'] = list(data.index)


def generate_text(all_sentences=None):
    data_tagged = []
    comb_sents_num = 2
    num_texts = math.ceil(len(all_sentences) / comb_sents_num)

    for i in range(0, num_texts):
        # print(i)
        l_limit = i * comb_sents_num
        if (i * comb_sents_num) + comb_sents_num > num_texts:
            u_limit = num_texts
        else:
            u_limit = (i * comb_sents_num) + comb_sents_num
        sentence_i = 0
        d = []
        for each_sent in all_sentences[l_limit:u_limit]:
            words = each_sent.split(" ")
            words[-1] = re.sub('[^0-9a-zA-Z]+', '', words[-1])
            tags = ['neos'] * (len(words) - 1)
            tags.append('eos')
            text_num = ['Text ' + str(i)] * len(words)
            sentence_num = ['Sentence ' + str(sentence_i)] * len(words)
            d.extend([(w, t) for w, t in zip(words, tags)])
            data_tagged.append(d)
            sentence_i = sentence_i + 1
    return data_tagged


texts = generate_text(data['Text'].values.tolist())

max_len = 50
tag2idx = {t: i for i, t in enumerate(['neos','eos'])}

X = [[w[0] for w in t] for t in texts]
new_X = []
for seq in X:
    new_seq = []
    for i in range(max_len):
        try:
            new_seq.append(seq[i])
        except:
            new_seq.append("__PAD__")
    new_X.append(new_seq)
X = new_X

y = [[tag2idx[w[1]] for w in t] for t in texts]

del data
del new_X
del texts

y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["neos"])

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1, random_state=2019)


del X
del y

batch_size = 32


sess = tf.Session()
K.set_session(sess)


elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())

def ElmoEmbedding(x):
    return elmo_model(inputs={
                            "tokens": tf.squeeze(tf.cast(x, tf.string)),
                            "sequence_len": tf.constant(batch_size*[max_len])
                      },
                      signature="tokens",
                      as_dict=True)["elmo"]


input_text = Input(shape=(max_len,), dtype='string')
embedding = Lambda(ElmoEmbedding, output_shape=(None, 1024))(input_text)
x = Bidirectional(LSTM(units=512, return_sequences=True,
                       recurrent_dropout=0.2, dropout=0.2))(embedding)
x_rnn = Bidirectional(LSTM(units=512, return_sequences=True,
                           recurrent_dropout=0.2, dropout=0.2))(x)
x = add([x, x_rnn])  # residual connection to the first biLSTM
out = TimeDistributed(Dense(2, activation="softmax"))(x)

model = Model(input_text, out)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

X_tr, X_val = X_tr[:(int(len(X_tr)/batch_size))*batch_size], X_tr[-135*batch_size:]
y_tr, y_val = y_tr[:int(len(X_tr)/batch_size)*batch_size], y_tr[-135*batch_size:]
y_tr = y_tr.reshape(y_tr.shape[0], y_tr.shape[1], 1)
y_val = y_val.reshape(y_val.shape[0], y_val.shape[1], 1)


history = model.fit(np.array(X_tr), y_tr, validation_data=(np.array(X_val), y_val),
                    batch_size=batch_size, epochs=2, verbose=1)

model_json = model.to_json()
model.save_weights("home/ubuntu/sentencesegmentation.h5")
