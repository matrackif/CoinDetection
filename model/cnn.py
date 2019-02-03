import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from typing import Tuple, List
NUM_CLASSES = 13


def cnn(input_shape: Tuple) -> Sequential:
    model = Sequential()
    model.add(Conv2D(256, kernel_size=(5, 5), strides=(1, 1),
                     activation='relu',
                     input_shape=input_shape, data_format='channels_last'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(128, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dense(250, activation='relu'))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(lr=0.01),
                  metrics=['accuracy'])
    return model
