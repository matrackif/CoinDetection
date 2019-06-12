import keras
import cv2
import os
import model.preprocessing as preprocessing
import model.base_model as base
import model.lenet_model as lenet
import model.stridenet_model as stridenet
import numpy as np
import model.enums
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Activation, BatchNormalization
from typing import Tuple, List
from model.coin_image import CoinImage
from pathlib import Path
NUM_CLASSES = 13


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []
        self.val_acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))


class ModelManager:
    def __init__(self, args):
        self.x_tr, self.x_te = None, None
        self.y_tr, self.y_te = None, None
        self.args = args
        self.model = None

    # TODO make grayscale program argument
    def init_training_data(self):
        dir_names = ['data/1_2_5_gr_tails/output', 'data/1_gr_heads/output', 'data/1_zl_heads/output', 'data/2_gr_heads/output', 'data/2_zl_heads/output',
                     'data/2_zl_tails/output',
                     'data/5_gr_heads/output', 'data/5_zl_heads/output', 'data/5_zl_tails/output', 'data/10_20_50_1_tails/output',
                     'data/10_gr_heads/output',
                     'data/20_gr_heads/output',
                     'data/50_gr_heads/output']
        x = None
        y = None
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
        for i in range(len(dir_names)):
            for filename in os.listdir(dir_names[i]):
                cimg = cv2.imread(os.path.join(dir_names[i], filename),
                                  cv2.IMREAD_GRAYSCALE if self.args['greyscale'] else cv2.IMREAD_COLOR)
                resized_img = cv2.resize(cimg, (int(self.args['img_width']), int(self.args['img_height'])))
                if x is None:
                    x = [resized_img]
                    y = [i]
                else:
                    x.append(resized_img)
                    y.append(i)
        y = np.array(y).reshape(-1, 1)
        enc = OneHotEncoder(categories='auto', sparse=False)
        enc.fit(y)
        x = np.array(x)
        x = x[:, :, :, np.newaxis] if self.args['greyscale'] else x
        x = preprocessing.normalize(x)
        self.x_tr, self.x_te = None, None
        self.y_tr, self.y_te = None, None
        for train_index, test_index in sss.split(x, y):
            # print("TRAIN indices:", train_index, "TEST indices:", test_index)
            self.x_tr, self.x_te = x[train_index], x[test_index]
            self.y_tr, self.y_te = y[train_index], y[test_index]
            break
        self.y_tr = enc.transform(self.y_tr)
        self.y_te = enc.transform(self.y_te)

    def init_cnn(self):
        possible_models = {
            'base': base.BaseModel(self.x_tr.shape[1:], NUM_CLASSES),
            'lenet': lenet.LeNetModel(self.x_tr.shape[1:], NUM_CLASSES),
            'stridenet': stridenet.StrideNetModel(self.x_tr.shape[1:], NUM_CLASSES)
        }
        self.model = possible_models[self.args['model_name']].model


    def get_model(self, show_plot: bool = True):
        if self.args['train_model']:
            self.init_training_data()
            print('x_tr.shape:', self.x_tr.shape)
            print('x_te.shape:', self.x_te.shape)
            print('y_tr.shape:', self.y_tr.shape)
            print('y_te.shape:', self.y_te.shape)
            acc_history = AccuracyHistory()
            self.init_cnn()
            err_history = self.model.fit(self.x_tr, self.y_tr,
                                         batch_size=32,
                                         epochs=self.args['epoch_count'],
                                         verbose=1,
                                         callbacks=[acc_history],
                                         validation_data=(self.x_te, self.y_te))
            if self.args['save_model']:
                model_path = Path(self.args['model_file'])
                model_file_name = model_path.name
                if model_path.is_file():
                    print('Model file:', model_path.name, 'already exists.')
                    res = input('Overwrite file name? (y/n). If no then model will be saved with name: new_' + model_file_name)
                    if res.lower() == 'y' or res.lower() == 'yes':
                        pass
                    else:
                        model_file_name = 'new_' + model_file_name
                print('Saving model to file with name:', model_file_name)
                self.model.save(model_file_name)
            print('Model initialized directly from training')
        else:
            try:
                self.model = load_model(self.args['model_file'])
                print('Model successfully loaded from file')
            except OSError:
                print('Failed to load model with name:', self.args['model_file'], 'in directory:', os.getcwd())

    def classify_image(self, raw_images_as_arr: np.ndarray):
        if self.model is not None:
            input_width = self.model.layers[0].input_shape[1]
            input_height = self.model.layers[0].input_shape[2]
            for i in range(raw_images_as_arr.shape[0]):
                raw_images_as_arr[i] = cv2.resize(raw_images_as_arr[i], (input_width, input_height))

            raw_images_as_arr = preprocessing.normalize(raw_images_as_arr)
            pred = self.model.predict(raw_images_as_arr)
            # print(pred, pred.shape)
            class_indexes = pred.argmax(axis=1)
            # print(class_indexes)
            # print('Classified coins:')
            for class_idx in class_indexes:
                print(model.enums.CoinLabel(class_idx + 1))

    def classify_coin_images(self, coin_images: List[CoinImage]):
        input_width = self.model.layers[0].input_shape[1]
        input_height = self.model.layers[0].input_shape[2]
        model_is_greyscale = True
        if self.model.layers[0].input_shape[3] == 3:
            model_is_greyscale = False
        if model_is_greyscale and not self.args['greyscale']:
            print('Error: Model was trained on greyscale images, it cannot predict value of color images')
            return
        elif not model_is_greyscale and self.args['greyscale']:
            print('Error: Model was trained on color images, it cannot predict value of greyscale images')
            return
        print('Classified coins:')
        for i in range(len(coin_images)):
            coin_images[i].img_arr = cv2.resize(coin_images[i].img_arr, (input_width, input_height))
            coin_images[i].img_arr = preprocessing.normalize(coin_images[i].img_arr)
            if self.args['show_images']:
                cv2.imshow('Image of Detected Coin From Hough Transform', coin_images[i].img_arr)
                cv2.waitKey(0)
            if self.args['greyscale'] and len(coin_images[i].img_arr) != 4:
                coin_images[i].img_arr = coin_images[i].img_arr.reshape((1, self.args['img_width'], self.args['img_height'], 1))
            elif not self.args['greyscale'] and len(coin_images[i].img_arr) != 4:
                coin_images[i].img_arr = coin_images[i].img_arr.reshape((1, self.args['img_width'], self.args['img_height'], 3))
            pred = self.model.predict(coin_images[i].img_arr)

            class_indexes = pred.argmax(axis=1)
            for class_idx in class_indexes:
                coin_label_val = model.enums.CoinLabel(class_idx + 1)
                coin_images[i].label = coin_label_val
                print('Detected coin with label:', coin_label_val)
