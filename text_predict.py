from keras.preprocessing.text import Tokenizer
import pickle
from keras.callbacks import ModelCheckpoint
import numpy as np

from attention.transformer import Transformer, LRSchedulerPerStep, LRSchedulerPerEpoch

with open('vect.pb', 'rb') as fp:
    texts = pickle.load(fp)

inputs = texts[0][0:6000]
targets = texts[1][0:6000]

tokenizer = Tokenizer(num_words=999, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~ ', lower=True, split=' ')
tokenizer.fit_on_texts(inputs)
indices = tokenizer.word_index
inputs = tokenizer.texts_to_sequences(inputs)

tokenizer = Tokenizer(num_words=999, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~ ', lower=True, split=' ')
tokenizer.fit_on_texts(targets)
labels = tokenizer.word_index
targets = tokenizer.texts_to_sequences(targets)

input_max_size = max(len(sentence) for sentence in inputs)
target_max_size = max(len(sentence) for sentence in targets)

for i in inputs:
    while True:     
        if len(i) < input_max_size:
            i.append(0)
        else:
            break
            
for i in targets:
    while True:     
        if len(i) < target_max_size:
            i.append(0)
        else:
            break

x_split = int(0.8 * len(inputs))
Xtrain, Xvalid = inputs[:x_split], inputs[x_split:]

y_split = int(0.8 * len(targets))
Ytrain, Yvalid = targets[:y_split], targets[y_split:]

d_model = 128
d_inner_hid = 128
limit = (max(input_max_size, target_max_size))

s2s = Transformer(inputs, targets, len_limit=limit, d_model=d_model, d_inner_hid=d_inner_hid, \
				   n_head=8, d_k=64, d_v=64, layers=2, dropout=0.1)

s2s.compile(optimizer='adam', fname='model.h5')

inpt = input('write: ')
text = str(inpt).split()

array = []
for t in text:
	for key, value in indices.items():
		if key == t:
			array.append(value)

while True:     
    if len(array) < target_max_size - 1:
        array.append(0)
    else:
        break

prediction = s2s.predict(array, delimiter=' ')

array = []
for p in prediction:
	for key, value in labels.items():
		if value == int(p):
			array.append(key)

array = [x for x in array if x != 'end']

print(' '.join(x for x in array))
	
