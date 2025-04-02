import numpy as np
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import BatchNormalization

(X_train, y_train), (X_valid, y_valid) = boston_housing.load_data()
X_train.shape
X_valid.shape
X_train[0]
y_train[0]

model = Sequential()

model.add(Dense(32, input_dim=13, activation='relu'))
model.add(BatchNormalization())

model.add(Dense(16, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(1, activation='linear'))

model.summary()

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_train, y_train,
          batch_size=8, epochs=32, verbose=1,
          validation_data=(X_valid, y_valid))

X_valid[42]

y_valid[42]

model.predict(np.reshape(X_valid[42], [1, 13]))
X_valid[43]

y_valid[43]
print(y_valid[42])
model.predict(np.reshape(X_valid[43], [1, 13]))
print(y_valid[43])