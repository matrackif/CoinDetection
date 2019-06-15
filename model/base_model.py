import keras
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Activation, BatchNormalization
from keras.models import Sequential


class BaseModel:
    def __init__(self, shape, num_classes):
        self.model = None
        self.create_model(shape, num_classes)

    def create_model(self, shape, num_classes):
        self.model = Sequential()
        self.model.add(Conv2D(256, kernel_size=(5, 5), strides=(1, 1),
                              input_shape=shape, data_format='channels_last'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        # model.add(Dropout(0.2))
        ############
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Conv2D(128, (5, 5)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        # model.add(Dropout(0.2))
        ############
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(500))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        # model.add(Dropout(0.2))
        ############
        self.model.add(Dense(250))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dense(num_classes, activation='softmax'))
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=keras.optimizers.SGD(lr=0.01),
                           metrics=['accuracy'])
