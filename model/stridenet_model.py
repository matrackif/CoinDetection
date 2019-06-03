import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Conv2D, Flatten, Activation, BatchNormalization, Dropout
from keras.regularizers import l2
import model.base_model as base


class StrideNetModel(base.BaseModel):
    def create_model(self, shape, num_classes):
        init = 'he_normal'
        reg = l2(0.0005)
        opt = Adam(lr=1e-4, decay=1e-4 / 100)
        self.model = Sequential()
        self.model.add(Conv2D(16, (7, 7), strides=(2, 2), padding="valid",
                              kernel_initializer=init, kernel_regularizer=reg,
                              input_shape=shape))
        self.add_layer(32, init, reg)
        self.add_layer(64, init, reg)
        self.add_layer(128, init, reg)
        ############
        self.model.add(Flatten())
        self.model.add(Dense(512, kernel_initializer=init, activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(num_classes, activation='softmax'))
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=opt,
                           metrics=['accuracy'])

    def add_layer(self, filters, init, reg):
        self.model.add(Conv2D(filters, (3, 3), padding="same",
                              kernel_initializer=init, kernel_regularizer=reg,
                              activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(filters, (3, 3), strides=(2, 2), padding="same",
                              kernel_initializer=init, kernel_regularizer=reg,
                              activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.25))
