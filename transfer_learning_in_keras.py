# Загрузка зависимостей:
from keras import Sequential
from keras.applications.vgg19 import VGG19
# from keras.models import sequential
from keras.layers import Dense, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator
# Загрузка предварительно обученной модели VGG19:
vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling=None)
# Зафиксировать все слои в базовой модели VGGNet19:
for layer in vgg19.layers:
    layer.trainable = False
model = Sequential()
model.add(vgg19)
# Добавление новых слоев поверх модели VGG19:
model.add(Flatten(name='flattened'))
model.add(Dropout(0.5, name='dropout'))
model.add(Dense(2, activation='softmax', name='predictions'))
# Подготовка (компиляция) модели к последующему обучению:
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

import kagglehub

# Download latest version
path = kagglehub.dataset_download("dansbecker/hot-dog-not-hot-dog")

print("Path to dataset files:", path)

# Создание двух экземпляров класса-генератора:
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    data_format='channels_last',
    rotation_range=30,
    horizontal_flip=True,
    fill_mode='reflect')
valid_datagen = ImageDataGenerator(
    rescale=1.0/255,
    data_format='channels_last')
# Определение размера пакета:
batch_size=32
# Определение генераторов обучающих и проверочных данных:
train_generator = train_datagen.flow_from_directory(
    directory=r'./hot-dog-not-hot-dog/train',
    target_size=(224, 224),
    classes=['hot_dog','not_hot_dog'],
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=True,
    seed=42
)
valid_generator = valid_datagen.flow_from_directory(
    directory=r'./hot-dog-not-hot-dog/test',
    target_size=(224, 224),
    classes=['hot_dog','not_hot_dog'],
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=True,
    seed=42
)

model.fit(train_generator, steps_per_epoch=15,
    epochs=16, validation_data=valid_generator,
    validation_steps=15)