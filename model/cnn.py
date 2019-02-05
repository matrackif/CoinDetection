import keras
import cv2
import os
import model.preprocessing as preprocessing
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Activation, BatchNormalization
from typing import Tuple, List
NUM_CLASSES = 13
MODEL_FILENAME = 'coin_det_model.h5'


def cnn(input_shape: Tuple) -> Sequential:
    model = Sequential()
    model.add(Conv2D(256, kernel_size=(5, 5), strides=(1, 1),
                     input_shape=input_shape, data_format='channels_last'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # model.add(Dropout(0.2))
    ############
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(128, (5, 5)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # model.add(Dropout(0.2))
    ############
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(500))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # model.add(Dropout(0.2))
    ############
    model.add(Dense(250))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(lr=0.01),
                  metrics=['accuracy'])
    return model


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []
        self.val_acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))


def init_training_data(width: int, height: int, grayscale: bool = True):
    dir_names = ['data/1_2_5_gr_tails', 'data/1_gr_heads', 'data/1_zl_heads', 'data/2_gr_heads', 'data/2_zl_heads',
                 'data/2_zl_tails',
                 'data/5_gr_heads', 'data/5_zl_heads', 'data/5_zl_tails', 'data/10_20_50_1_tails', 'data/10_gr_heads',
                 'data/20_gr_heads',
                 'data/50_gr_heads']
    x = None
    y = None
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    for i in range(len(dir_names)):
        for filename in os.listdir(dir_names[i]):
            cimg = cv2.imread(os.path.join(dir_names[i], filename),
                              cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
            resized_img = cv2.resize(cimg, (int(width), int(height)))
            if x is None:
                x = [resized_img]
                y = [i]
            else:
                x.append(resized_img)
                y.append(i)
    y = np.array(y).reshape(-1, 1)
    enc = OneHotEncoder(categories='auto', sparse=False)
    enc.fit(y)
    x = x[:, :, :, np.newaxis] if grayscale else x
    x = preprocessing.normalize(np.array(x))
    x_tr, x_te = None, None
    y_tr, y_te = None, None
    for train_index, test_index in sss.split(x, y):
        # print("TRAIN indices:", train_index, "TEST indices:", test_index)
        x_tr, x_te = x[train_index], x[test_index]
        y_tr, y_te = y[train_index], y[test_index]
        break
    y_tr = enc.transform(y_tr)
    y_te = enc.transform(y_te)
    return x_tr, x_te, y_tr, y_te


def train_model(args, save_model: bool = False, show_plot: bool = True):
    x_train, x_test, y_train, y_test = init_training_data(width=args['img_width'], height=args['img_height'],
                                                          grayscale=False)
    print('x_train.shape:', x_train.shape)
    print('x_test.shape:', x_test.shape)
    print('y_train.shape:', y_train.shape)
    print('y_test.shape:', y_test.shape)
    acc_history = AccuracyHistory()
    model = cnn(x_train[0].shape)
    err_history = model.fit(x_train, y_train,
                            batch_size=32,
                            epochs=args['epoch_count'],
                            verbose=1,
                            callbacks=[acc_history],
                            validation_data=(x_test, y_test))
    model.save(MODEL_FILENAME)
    if show_plot:
        plt.plot(range(args['epoch_count']), acc_history.acc, label='Training accuracy')
        plt.plot(range(args['epoch_count']), acc_history.val_acc, label='Test accuracy')
        plt.title('Accuracy of Coin Classifier During Training Epochs')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.show()
        ########
        plt.plot(err_history.history['loss'], label='Training loss (error)')
        plt.plot(err_history.history['val_loss'], label='Test loss (error)')
        plt.title('Training/test loss of coin classifier')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()
    return model
