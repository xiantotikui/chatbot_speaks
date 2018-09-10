import pickle

SENTENCE_START_TOKEN = "\t"
SENTENCE_END_TOKEN = "\n"
min_sent_characters=0

sentences = []

with open('./data/RNN-twitter/tweets.txt', 'rt') as f:
    txt = f.read()
    txt = txt.split('\n')
    txt = [line.split(' |') for line in txt]
    txt.pop()
    txt.pop()
    for line in txt: line.pop()
    txt = [s for s in txt if len(s) >= min_sent_characters]
    for words in txt:
        for word in words:
            tmp = word.replace("'", "").replace("\"", "").split()
            tmp.pop(0)
            sentences.append(SENTENCE_START_TOKEN + ' '.join(tmp) + SENTENCE_END_TOKEN)

even = sentences[0:][::2]
odd = sentences[1:][::2]

with open('vect.pb', 'wb') as fp:
    pickle.dump([even, odd], fp)
