import cv2
import numpy as np
import keras
from copy import deepcopy
import argparse
import os
from model.cnn import cnn
from matplotlib import pyplot as plt
DEFAULT_INPUT_WIDTH = 100
DEFAULT_INPUT_HEIGHT = 100


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))


def init_training_data(width: int = DEFAULT_INPUT_WIDTH, height: int = DEFAULT_INPUT_HEIGHT):
    dir_names = ['data/1_2_5_gr_tails', 'data/1_gr_heads', 'data/1_zl_heads', 'data/2_gr_heads', 'data/2_zl_heads',
                 'data/2_zl_tails',
                 'data/5_gr_heads', 'data/5_zl_heads', 'data/5_zl_tails', 'data/10_20_50_1_tails', 'data/10_gr_heads',
                 'data/20_gr_heads',
                 'data/50_gr_heads']
    x_train = None
    y_train = None
    for i in range(len(dir_names)):
        for filename in os.listdir(dir_names[i]):
            cimg = cv2.imread(os.path.join(dir_names[i], filename), cv2.IMREAD_COLOR)
            resized_img = cv2.resize(cimg, (int(width), int(height)))
            if x_train is None:
                x_train = [resized_img]
                y_train = np.zeros(shape=(1, len(dir_names)))
                y_train[0][i] = 1
            else:
                x_train.append(resized_img)
                newRow = np.zeros(shape=(1, len(dir_names)))
                newRow[0][i] = 1
                y_train = np.concatenate((y_train, newRow), axis=0)
            # print('resized_img.shape:', resized_img.shape)
    x_train = np.array(x_train)
    # print('x_train:', x_train)
    # print('y_train:', y_train)
    print('x_train.shape:', x_train.shape)
    print('y_train.shape:', y_train.shape)
    return x_train, y_train


if __name__ == '__main__':
    x_train, y_train = init_training_data()
    history = AccuracyHistory()
    model = cnn(x_train[0].shape)
    model.fit(x_train, y_train,
              batch_size=5,
              epochs=10,
              verbose=1,
              callbacks=[history])
    plt.plot(range(10), history.acc)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()
    # Instantiate the parser
    """
    parser = argparse.ArgumentParser(description='Circle extractor')
    parser.add_argument('--save', action='store_true',
                        help='Save scaled images')
    args = vars(parser.parse_args())
    print('Args:', args)
    img = cv2.imread('data/coins.jpg', 0)
    cimg = cv2.imread('data/coins.jpg', cv2.IMREAD_COLOR)
    cimg2 = deepcopy(cimg)
    cv2.imshow('color image', cimg)
    cv2.waitKey(0)
    ret, dst = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    circles = cv2.HoughCircles(dst, cv2.HOUGH_GRADIENT, dp=1, minDist=10, maxRadius=50, param1=50, param2=30, minRadius=10)
    circles = np.uint16(np.around(circles))
    j = 0
    for i in circles[0, :]:
        x = i[0] - i[2]
        y = i[1] - i[2]
        height = 2 * i[2]
        width = 2 * i[2]
        roi = cimg[y - 3:y + height + 6, x - 3:x + width + 6]
        new_roi = cv2.resize(roi, (100, 100))
        if args['save']:
            cv2.imwrite(str(j) + '.jpg', new_roi)
        else:
            pass
            # Uncomment to show scaled coin
            # cv2.imshow("Scaled coin", new_roi)
            # cv2.waitKey(0)

        # draw the outer circle
        cv2.circle(cimg2, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(cimg2, (i[0], i[1]), 2, (0, 0, 255), 3)
        j += 1

    cv2.imshow('detected circles', cimg2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
