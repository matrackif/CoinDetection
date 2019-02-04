import cv2
import numpy as np
import keras
import argparse
import os
import model.preprocessing as preprocessing
from model.cnn import cnn
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from copy import deepcopy


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))


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
        print("TRAIN indices:", train_index, "TEST indices:", test_index)
        x_tr, x_te = x[train_index], x[test_index]
        y_tr, y_te = y[train_index], y[test_index]
        break
    y_tr = enc.transform(y_tr)
    y_te = enc.transform(y_te)
    return x_tr, x_te, y_tr, y_te


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--epoch-count', nargs='?', type=int, default=60,
                        const=60, help='Number of epochs to train the CNN')
    parser.add_argument('-w', '--img-width', nargs='?', type=int, default=100,
                        const=100, help='All input images will have their width resized to this number')
    parser.add_argument('-ht', '--img-height', nargs='?', type=int, default=100,
                        const=100, help='All input images will have their height resized to this number')
    args = vars(parser.parse_args())
    print('Program arguments:', args)
    x_train, x_test, y_train, y_test = init_training_data(width=args['img_width'], height=args['img_height'], grayscale=False)
    print('x_train.shape:', x_train.shape)
    print('x_test.shape:', x_test.shape)
    print('y_train.shape:', y_train.shape)
    print('y_test.shape:', y_test.shape)
    acc_history = AccuracyHistory()
    model = cnn(x_train[0].shape)
    err_history = model.fit(x_train, y_train,
                            batch_size=5,
                            epochs=args['epoch_count'],
                            verbose=1,
                            callbacks=[acc_history],
                            validation_data=(x_test, y_test))
    plt.plot(range(args['epoch_count']), acc_history.acc)
    plt.title('Accuracy of Coin Classifier During Training Epochs')
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
