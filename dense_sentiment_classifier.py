import keras
from keras.datasets import imdb # new!
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Embedding # new!
from keras.callbacks import ModelCheckpoint # new!
import os # new!
from sklearn.metrics import roc_auc_score, roc_curve # new!
import pandas as pd
import matplotlib.pyplot as plt # new!

# output directory name:
output_dir = 'model_output/dense'

# training:
epochs = 4
batch_size = 128

# vector-space embedding:
n_dim = 64
n_unique_words = 5000 # as per Maas et al. (2011); may not be optimal
n_words_to_skip = 50 # ditto
max_review_length = 100
pad_type = trunc_type = 'pre'

# neural network architecture:
n_dense = 64
dropout = 0.5

(x_train, y_train), (x_valid, y_valid) = imdb.load_data(num_words=n_unique_words,
                                                        skip_top=n_words_to_skip)

x_train[0:6] # 0 reserved for padding; 1 would be starting character; 2 is unknown; 3 is most common word, etc.

for x in x_train[0:6]:
    print(len(x))

y_train[0:6]

len(x_train), len(x_valid)

word_index = keras.datasets.imdb.get_word_index()
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["PAD"] = 0
word_index["START"] = 1
word_index["UNK"] = 2

word_index

index_word = {v:k for k,v in word_index.items()}

x_train[0]

' '.join(index_word[id] for id in x_train[0])

(all_x_train,_),(all_x_valid,_) = imdb.load_data()


' '.join(index_word[id] for id in all_x_train[0])

x_train = pad_sequences(x_train, maxlen=max_review_length,
                        padding=pad_type, truncating=trunc_type, value=0)
x_valid = pad_sequences(x_valid, maxlen=max_review_length,
                        padding=pad_type, truncating=trunc_type, value=0)

x_train[0:6]

for x in x_train[0:6]:
    print(len(x))

' '.join(index_word[id] for id in x_train[0])

' '.join(index_word[id] for id in x_train[5])

model = Sequential()
model.add(Embedding(n_unique_words, n_dim, input_length=max_review_length))
model.add(Flatten())
model.add(Dense(n_dense, activation='relu'))
model.add(Dropout(dropout))
# model.add(Dense(n_dense, activation='relu'))
# model.add(Dropout(dropout))
model.add(Dense(1, activation='sigmoid')) # mathematically equivalent to softmax with two classes

model.summary() # so many parameters!

# embedding layer dimensions and parameters:
n_dim, n_unique_words, n_dim*n_unique_words

# ...flatten:
max_review_length, n_dim, n_dim*max_review_length

# ...dense:
n_dense, n_dim*max_review_length*n_dense + n_dense # weights + biases

# ...and output:
n_dense + 1

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

modelcheckpoint = ModelCheckpoint(filepath=output_dir+
                                  "/weights.{epoch:02d}.hdf5")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model.fit(x_train, y_train,
          batch_size=batch_size, epochs=epochs, verbose=1,
          validation_data=(x_valid, y_valid),
          callbacks=[modelcheckpoint])

model.load_weights(output_dir+"/weights.02.hdf5") # NOT zero-indexed
y_hat = model.predict(x_valid)

len(y_hat)

y_hat[0]

y_valid[0]

plt.hist(y_hat)
k = plt.axvline(x=0.5, color='orange')
plt.show()

pct_auc = roc_auc_score(y_valid, y_hat)*100.0

"{:0.2f}".format(pct_auc)

float_y_hat = []
for y in y_hat:
    float_y_hat.append(y[0])

ydf = pd.DataFrame(list(zip(float_y_hat, y_valid)), columns=['y_hat', 'y'])

ydf.head(10)

' '.join(index_word[id] for id in all_x_valid[0])

' '.join(index_word[id] for id in all_x_valid[6])

ydf[(ydf.y == 0) & (ydf.y_hat > 0.9)].head(10)

' '.join(index_word[id] for id in all_x_valid[386])

ydf[(ydf.y == 1) & (ydf.y_hat < 0.1)].head(10)

' '.join(index_word[id] for id in all_x_valid[224])




