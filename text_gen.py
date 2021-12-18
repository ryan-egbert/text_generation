import numpy as np
# import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Flatten, Dropout
from collections import Counter
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import pickle as pck
from nltk.tokenize import word_tokenize

text = ''
with open('hp.txt', 'r') as f:
    for line in f:
        if line != '':
            text += line + ' '

char_to_replace = ['.',',','!','?',':',';','(',')','-','\n','\\','\'']
for char in char_to_replace:
    text = text.translate({ord(char):None})
    
words = word_tokenize(text.lower())
# words = [word.lower() for word in text.split()]
counter = Counter(words)
total_words = len(words)
sorted_words = counter.most_common(total_words)
vocab = {w:i+1 for i,(w,c) in enumerate(sorted_words)}
inv_vocab = {i+1:w for i,(w,c) in enumerate(sorted_words)}

text_num = []
for word in words:
    text_num.append(vocab[word])
    
X = []
y = []
seq_len = 100
for i in range(len(words) - seq_len):
    seq_in = text_num[i:i+seq_len]
    seq_out = text_num[i+seq_len]
    X.append(seq_in)
    y.append(seq_out)

n_patterns = len(X)
print("number of sequences: ", n_patterns)

Xr = np.reshape(X, (n_patterns, seq_len, 1))
Xr = Xr / float(len(words))
yr = np_utils.to_categorical(y)



model = Sequential()
model.add(LSTM(256, input_shape=(Xr.shape[1], Xr.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(yr.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy')
# model.compile(loss='categorical_crossentropy', optimizer='adam')

filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

print(Xr.shape)
print(yr.shape)
model.fit(Xr, yr, batch_size=256, epochs=100, validation_split=0.4)


with open("model.pck", 'wb') as f:
    pck.dump(model, f)

seq = text_num[:seq_len]
text = ' '.join(words[:seq_len])
for i in range(100):
    x = np.reshape(seq, (1, seq_len, 1))
    x = x / float(len(words))
    p = model.predict(x)
    index = np.argmax(p)
#     print(index)
    seq.append(index)
    seq = seq[1:]
    text += ' ' + inv_vocab[index]
    
print(text)