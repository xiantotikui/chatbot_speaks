from keras.preprocessing.text import Tokenizer
import pickle
from keras.callbacks import ModelCheckpoint

from attention.transformer import Transformer, LRSchedulerPerStep, LRSchedulerPerEpoch

with open('vect.pb', 'rb') as fp:
    texts = pickle.load(fp)

inputs = texts[0][0:5000]
targets = texts[1][0:5000]

tokenizer = Tokenizer(num_words=999)
tokenizer.fit_on_texts(inputs)
tokenizer.word_index
inputs = tokenizer.texts_to_sequences(inputs)

tokenizer = Tokenizer(num_words=999)
tokenizer.fit_on_texts(targets)
tokenizer.word_index
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
limit = (max(input_max_size, target_max_size))

s2s = Transformer(inputs, targets, len_limit=limit, d_model=d_model, d_inner_hid=512, \
				   n_head=8, d_k=64, d_v=64, layers=2, dropout=0.1)

lr_scheduler = LRSchedulerPerStep(d_model, 4000)   # there is a warning that it is slow, however, it's ok.
#lr_scheduler = LRSchedulerPerEpoch(d_model, 4000, Xtrain.shape[0]/64)  # this scheduler only update lr per epoch
model_saver = ModelCheckpoint('model.h5', save_best_only=True, save_weights_only=True)

s2s.compile(optimizer='adam')

s2s.model.fit([Xtrain, Ytrain], None, batch_size=64, epochs=30, \
				validation_data=([Xvalid, Yvalid], None), \
				callbacks=[lr_scheduler, model_saver])
