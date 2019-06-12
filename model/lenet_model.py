import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Activation, BatchNormalization, Dropout, MaxPooling2D
from keras.regularizers import l2
import model.base_model as base


class LeNetModel(base.BaseModel):
    def create_model(self, shape, num_classes):
        init = 'he_normal'
        reg = l2(0.0005)
        self.model = Sequential()
        self.model.add(Conv2D(16, (5, 5), strides=(1, 1), padding="valid",
                              kernel_initializer=init, kernel_regularizer=reg,
                              input_shape=shape, activation="tanh"))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Conv2D(32, (5, 5), strides=(1, 1), padding="valid",
                             kernel_initializer=init, kernel_regularizer=reg,
                             activation="tanh"))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Dense(256))
        self.model.add(BatchNormalization())
        self.model.add(Activation('tanh'))
        self.model.add(Dense(num_classes, activation='softmax'))
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=keras.optimizers.SGD(lr=0.01),
                           metrics=['accuracy'])


