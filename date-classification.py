#!/usr/bin/env python
# coding: utf-8

# In[1]:


from faker import Faker
from tensorflow.keras.layers import RepeatVector, Concatenate, Dense, Dot, Activation
from tensorflow.keras.layers import Input, Bidirectional, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import numpy as np
import random
from babel.dates import format_date
import os


# In[2]:


faker = Faker()
np.random.seed(5)


# In[3]:


FORMATS = ['short',
           'medium',
           'medium',
           'medium',
           'long', 'long',
           'long', 'long',
           'long', 'full',
           'full', 'full',
           'd MMM YYY',
           'd MMMM YYY',
           'd MMMM YYY',
           'd MMMM YYY',
           'd MMMM YYY',
           'd MMMM YYY',
           'dd/MM/YYY',
           'EE d, MMM YYY',
           'EEEE d, MMMM YYY',
           'MMM d, YYY',
           'MMMM d, YYY',
           'YYY, d MMM',
           'YYY, d MMMM',
           'EE YYY, d MMMM',
           'EEEE YYY, d MM'
           ]
for format in FORMATS:
    print('%s => %s' % (format, format_date(faker.date_object(), format=format, locale='en')))


# In[4]:


def random_date():
    dt = faker.date_time_between(start_date='-500y', end_date='+50y')
    try:
        date = format_date(dt, format=random.choice(FORMATS), locale='en')
        human_readable = date.lower().replace(',', '')
        machine_readable = format_date(dt, format="YYYY-MM-dd", locale='en')
    except AttributeError as e:
        return None, None, None
    return human_readable, machine_readable


# In[5]:


random_date()


# In[6]:


human_vocab = set()
machine_vocab = set()
dataset = []
batch_size = 128
Tx = 30
Ty = 10
m = int(batch_size * 100 * 4)
test_size = int(m / 10)

for i in range(m):
    hd, md = random_date()
    dataset.append((hd, md))
    human_vocab.update(tuple(hd))
    machine_vocab.update(tuple(md))

# self-made tokenization

human_vocab.update(('<pad>', '<unk>'))
human_vocab = dict(enumerate(human_vocab))
# human_vocab = tokenizer.word_index
human_vocab = {v: i for i, v in human_vocab.items()}
human_vocab_index_word = dict(enumerate(human_vocab))

machine_vocab.add('<unk>')
machine_vocab = dict(enumerate(machine_vocab))
# word_index
inv_machine_vocab = {v: i for i, v in machine_vocab.items()}

print(len(dataset), len(human_vocab), len(machine_vocab))
dataset[:10]


# In[7]:


# test set
t = test_size
testset = []
for i in range(t):
    hd, md = random_date()
    testset.append((hd, md))


# In[8]:


HUMAN_VOCAB = len(human_vocab)
MACHINE_VOCAB = len(machine_vocab)

print(HUMAN_VOCAB, MACHINE_VOCAB)


# In[9]:


def string_to_ohe(string, T, vocab):
    # in a seq-to-seq model batch of one-hot vectors is the expected output of the final softmax layer
    # vocab play the role as tokenizer.word_index, i.e., word to index
    # in the past I work on tokenizing words, this time we tokenizer every single characters
    string = string.lower()
    arr = []
    while len(arr) < len(string):
        curr_index = len(arr)
        arr.append(vocab.get(string[curr_index], vocab['<unk>']))

    while len(arr) < T:
        arr.append(vocab['<pad>'])

    onehot = np.zeros((T, len(vocab)))
    for i in range(T):
        onehot[i, arr[i]] = 1

    return onehot, arr


def output_to_date(out, vocab):
    # this time the "vocab" is index_word
    arr = np.argmax(out, axis=-1)
    string = ''
    for i in arr:
        string += vocab[i]

    return string


# In[10]:


X = []
Y = []

for x, y in dataset:
    X.append(string_to_ohe(x, Tx, human_vocab)[0])
    Y.append(string_to_ohe(y, Ty, inv_machine_vocab)[0])

X = np.array(X)
Y = np.array(Y)
X.shape, Y.shape


# In[11]:


output_to_date(X[random.randint(0, 50000)], human_vocab_index_word)


# In[12]:


Xt = []
Yt = []

for x, y in testset:
    Xt.append(string_to_ohe(x, Tx, human_vocab)[0])
    Yt.append(string_to_ohe(y, Ty, inv_machine_vocab)[0])

Xt, Yt = np.array(Xt), np.array(Yt)
Xt.shape, Yt.shape


# In[13]:


# combines activations generated from BiLSTM with previous state of Post LSTM cell to get attention to be given to each timestep
# heart of attention model

def one_step_attention(a, s_prev):
    x = RepeatVector(Tx)(s_prev)
    # this is to make use of all the information from output hidden state from the first LSTM layer
    # and "current" hidden state of second LSTM layer\
    # shape of a: (none, T_x, 2*H_encoder), shape off current x: (none, T_x, H_decoder)
    x = Concatenate(axis=-1)([a, x])
    # shape of x: (none, T_x, 2*H_encoder + H_decoder)
    e = Dense(Ty, activation='tanh')(x)
    energy = Dense(1, activation='relu')(e)
    # shape of energy: (none, T_x, 1)
    alphas = Activation('softmax')(energy)
    # shape of alphas: (none, T_x)
    # alphas are the attention weight
    # therefore context is the weighted sum of our first LSTM layer's hidden state
    # context can be thought of as fine-tuned output hidden state that is to be fed into decoder LSTM model
    context = Dot(axes=1)([alphas, a])

    return context


# <img width="50%" src="./attention_mechanism_2.png"><img width="90%" src="./attention_mechanism_1.png">

# In[14]:


n_a = 32  # pre attention LSTM state, since Bi directional attention=64
n_s = 64  # post attention LSTM state

inp = Input((Tx, HUMAN_VOCAB))
s0 = Input((n_s,))
c0 = Input((n_s,))

outputs = []

s = s0
c = c0
a = Bidirectional(
    LSTM(
        n_a,
        batch_input_shape=(batch_size, Tx, HUMAN_VOCAB),
        return_sequences=True
    ))(inp)  # generate hidden state for every timestep

postLSTM = LSTM(n_s, return_state=True)

output = Dense(MACHINE_VOCAB, activation='softmax')  # our final output layer

for _ in range(Ty):  # iterate for every output step
    context = one_step_attention(a, s)  # get context
    s, _, c = postLSTM(context, initial_state=[s, c])  # generate cell_state_seq(currently 1), cell_state, memory
    out = output(s)
    outputs.append(out)

model = Model([inp, s0, c0], outputs)


# In[ ]:


model.compile(
    optimizer=Adam(lr=0.005, decay=0.01),
    loss='categorical_crossentropy',
    metrics=['acc']
)

s0 = np.zeros((m, n_s))
c0 = np.zeros((m, n_s))

Y = list(Y.swapaxes(0, 1))
Yt = list(Yt.swapaxes(0, 1))

history = model.fit([X, s0, c0], Y, epochs=100,
                    validation_data=([Xt, np.zeros((t, n_s)), np.zeros((t, n_s))], Yt),
                    batch_size=batch_size, verbose=1)

model.save_weights('attention_weights.h5')


# In[ ]:
