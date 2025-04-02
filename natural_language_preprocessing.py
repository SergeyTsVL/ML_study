import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import *
nltk.download('gutenberg')
nltk.download('punkt_tab')
nltk.download('stopwords')

import string

import gensim
from gensim.models.phrases import Phraser, Phrases
from gensim.models.word2vec import Word2Vec

from sklearn.manifold import TSNE

import pandas as pd
from bokeh.io import output_notebook, output_file
from bokeh.plotting import show, figure
# %matplotlib inline

from nltk.corpus import gutenberg

# print(len(gutenberg.fileids()))

# print(gutenberg.fileids())

# print(len(gutenberg.words()))

gberg_sent_tokens = sent_tokenize(gutenberg.raw())

# print(gberg_sent_tokens[0:6])

# print(gberg_sent_tokens[1])

# print(word_tokenize(gberg_sent_tokens[1]))

word_tokenize(gberg_sent_tokens[1])[14]

# a convenient method that handles newlines, as well as tokenizing sentences and words in one shot
gberg_sents = gutenberg.sents()

# print(gberg_sents[0:6])

# print(gberg_sents[4][14])

# print(gberg_sents[0:2])

# print([w.lower() for w in gberg_sents[4]])

stpwrds = stopwords.words('english') + list(string.punctuation)

# print(stpwrds)

# print([w.lower() for w in gberg_sents[4] if w.lower() not in stpwrds])

stemmer = PorterStemmer()

# print([stemmer.stem(w.lower()) for w in gberg_sents[4] if w.lower() not in stpwrds])

phrases = Phrases(gberg_sents) # train detector

bigram = Phraser(phrases) # create a more efficient Phraser object for transforming sentences

# print(bigram.phrasegrams) # output count and score of each bigram

tokenized_sentence = "Jon lives in New York City".split()

# print(tokenized_sentence)
#
# print(bigram[tokenized_sentence])

# as in Maas et al. (2001):
# - leave in stop words ("indicative of sentiment")
# - no stemming ("model learns similar representations of words of the same stem when data suggests it")
lower_sents = []
for s in gberg_sents:
    lower_sents.append([w.lower() for w in s if w.lower()
                        not in list(string.punctuation)])

# print(lower_sents[0:5])

lower_bigram = Phraser(Phrases(lower_sents))

# print(lower_bigram.phrasegrams) # miss taylor, mr woodhouse, mr weston

# print(lower_bigram["jon lives in new york city".split()])

lower_bigram = Phraser(Phrases(lower_sents,
                               min_count=32, threshold=64))
# print(lower_bigram.phrasegrams)

clean_sents = []
for s in lower_sents:
    clean_sents.append(lower_bigram[s])

# print(clean_sents[0:9])

# print(clean_sents[6])

# max_vocab_size can be used instead of min_count (which has increased here)
model = Word2Vec(sentences=clean_sents, vector_size=64,
                 sg=1, window=10, epochs=5,
                 min_count=10, workers=4)
model.save('clean_gutenberg_model.w2v')

# skip re-training the model with the next line:
model = gensim.models.Word2Vec.load('clean_gutenberg_model.w2v')

# print(len(model.wv.vocab)) # would be 17k if we carried out no preprocessing
# print(len(model.wv.index_to_key))
#
# print(model.wv['dog'])
#
# print(len(model.wv['dog']))

# print(model.wv.most_similar('dog', topn=3))
#
# print(model.wv.most_similar('eat', topn=3))
#
# print(model.wv.most_similar('day', topn=3))
#
# print(model.wv.most_similar('father', topn=3))
#
# print(model.wv.most_similar('ma_am', topn=3))
#
# print(model.wv.doesnt_match("mother father sister brother dog".split()))
#
# print(model.wv.similarity('father', 'dog'))
#
# print(model.wv.most_similar(positive=['father', 'woman'], negative=['man']))
#
# print(model.wv.most_similar(positive=['husband', 'woman'], negative=['man']))

tsne = TSNE(n_components=2, max_iter=1000)
# X_2d = tsne.fit_transform(model.wv[model.wv.vocab])
X_2d = tsne.fit_transform(model.wv[model.wv.index_to_key])
coords_df = pd.DataFrame(X_2d, columns=['x','y'])
coords_df['token'] = model.wv.index_to_key
coords_df.head()
coords_df.to_csv('clean_gutenberg_tsne.csv', index=False)

coords_df = pd.read_csv('clean_gutenberg_tsne.csv')




_ = coords_df.plot.scatter('x', 'y', figsize=(12,12),
                           marker='.', s=10, alpha=0.2)
subset_df = coords_df.sample(n=5000)
# print(subset_df)
p = figure(min_width=800, height=800)
_ = p.text(x=subset_df.x, y=subset_df.y, text=subset_df.token)
# print(_)
show(p)
# p.show()

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
# fig = plt.figure(figsize=(12, 12))
# p = fig.add_subplot(111, projection='3d')
# p.scatter(coords_df['x'], coords_df['y'], coords_df['z'])
# plt.show()











