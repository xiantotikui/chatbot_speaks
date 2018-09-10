import pickle
import json

SENTENCE_START_TOKEN = "<START>"
SENTENCE_END_TOKEN = "<END>"

with open('./data/Frames-dataset/frames.json') as f:
    dialogues = json.load(f)

sentences = []
for text in dialogues:
    for point in text['turns']:
        sentences.append([point['author'], point['text']])

user = []
wizard = []

tmp0 = []
tmp1 = []

for sentence in sentences:
    if sentence[0] == 'user':
        tmp0.append(sentence[1])
        wizard.append(SENTENCE_START_TOKEN + ' '.join(tmp1) + SENTENCE_END_TOKEN)
        tmp1 = []
    else:
        tmp1.append(sentence[1])
        user.append(SENTENCE_START_TOKEN + ' '.join(tmp0) + SENTENCE_END_TOKEN)
        tmp0 = []
wizard.pop(0)
 
with open('vect.pb', 'wb') as fp:
    pickle.dump([user, wizard], fp)


