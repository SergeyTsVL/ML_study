from matplotlib import pyplot as plt
import numpy as np
import keras
# import tensorflow
from keras.datasets import mnist


(X_train, y_train), (X_valid, y_valid) = mnist.load_data()
# sample = np.random.randint(0, X_train.shape[0])
sample = 39235
plt.figure(figsize = (10,10))
mnist_img = X_train[sample]
plt.imshow(mnist_img,cmap="Greys")
ax = plt.gca()

# First turn off the  major labels, but not the major ticks
plt.tick_params(
    axis='both',        # changes apply to the both x and y axes
    which='major',      # Change the major ticks only
    bottom=True,        # ticks along the bottom edge are on
    left=True,          # ticks along the top edge are on
    labelbottom=False,  # labels along the bottom edge are off
    labelleft=False)    # labels along the left edge are off

# Next turn off the minor ticks, but not the minor labels
plt.tick_params(
    axis='both',        # changes apply to both x and y axes
    which='minor',      # Change the minor ticks only
    bottom=False,       # ticks along the bottom edge are off
    left=False,         # ticks along the left edge are off
    labelbottom=True,   # labels along the bottom edge are on
    labelleft=True)     # labels along the left edge are on

# Set the major ticks, starting at 1 (the -0.5 tick gets hidden off the canvas)
ax.set_xticks(np.arange(-.5, 28, 1))
ax.set_yticks(np.arange(-.5, 28, 1))

# Set the minor ticks and labels
ax.set_xticks(np.arange(0, 28, 1), minor=True);
ax.set_xticklabels([str(i) for i in np.arange(0, 28, 1)], minor=True);
ax.set_yticks(np.arange(0, 28, 1), minor=True);
ax.set_yticklabels([str(i) for i in np.arange(0, 28, 1)], minor=True);

ax.grid(color='black', linestyle='-', linewidth=1.5)
_ = plt.colorbar(fraction=0.046, pad=0.04, ticks=[0,32,64,96,128,160,192,224,255])
# plt.show()
y_train[sample]


# # Проверьте размеры загруженных данных
# print(f"Размер обучающих изображений: {X_train.shape}")
# print(f"Размер тестовых изображений: {X_valid.shape, y_valid.shape}")
# print(f"Размер тестовых изображений: {X_valid[0]}")
plt.imshow (X_valid [0], cmap = 'Grays')
# plt.show()
# print(y_valid [0])


X_train = X_train.reshape(60000, 784).astype('float32')
X_valid = X_valid.reshape(10000, 784).astype('float32')
# print(X_valid[0])
X_train /= 255
X_valid /= 255
# print(X_valid[0])

n_classes = 10
y_train = keras.utils.to_categorical(y_train, n_classes)
y_valid = keras.utils.to_categorical(y_valid, n_classes)
print(y_valid)
