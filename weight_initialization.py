import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.initializers import Zeros, RandomNormal, glorot_normal, glorot_uniform

n_input = 784
n_dense = 256
b_init = Zeros()
# w_init = RandomNormal(stddev=1.0)
# w_init = glorot_normal()
w_init = glorot_uniform()

model = Sequential()
model.add(Dense(n_dense,
                input_dim=n_input,
                kernel_initializer=w_init,
                bias_initializer=b_init))
# model.add(Activation('sigmoid'))
# model.add(Activation('tanh'))
model.add(Activation('relu'))

x = np.random.random((1,n_input))

a = model.predict(x)
print(a)
_ = plt.hist(np.transpose(a))
plt.show()