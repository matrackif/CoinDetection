import argparse
import os
import numpy as np
import cv2
import model.preprocessing as preprocessing
import model.postprocessing as postprocessing
from model.model_manager import ModelManager
from scikitplot.metrics import plot_confusion_matrix
from matplotlib import pyplot as plt
DEFAULT_EPOCH_COUNT = 100
DEFAULT_IMAGE_HEIGHT_WIDTH = 150


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train-model', action='store_true', default=False,
                            help='Train Keras baseline model')
    parser.add_argument('-s', '--save-model', action='store_true', default=False,
                        help='Save trained model, if -t is not passed then this argument is ignored')
    parser.add_argument('-p', '--show-images', action='store_true', default=False,
                        help='Show the images found from Hough Transform')
    parser.add_argument('-n', '--epoch-count', nargs='?', type=int, default=DEFAULT_EPOCH_COUNT,
                        const=DEFAULT_EPOCH_COUNT, help='Number of epochs to train the CNN')
    parser.add_argument('-w', '--img-width', nargs='?', type=int, default=DEFAULT_IMAGE_HEIGHT_WIDTH,
                        const=DEFAULT_IMAGE_HEIGHT_WIDTH,
                        help='All input images will have their width resized to this number')
    parser.add_argument('-ht', '--img-height', nargs='?', type=int, default=DEFAULT_IMAGE_HEIGHT_WIDTH,
                        const=DEFAULT_IMAGE_HEIGHT_WIDTH,
                        help='All input images will have their height resized to this number')
    parser.add_argument('-f', '--input-file', nargs='?', type=str, default='data/images_of_multiple_coins/coins.jpg',
                        const='data/images_of_multiple_coins/coins.jpg',
                        help='Input file to image to be classified')
    args = vars(parser.parse_args())

    mm = ModelManager(args=args)
    mm.get_model()
    """
    mm.classify_image(np.ones(shape=(1, 150, 150, 3)))
    """
    coin_imgs = preprocessing.hough_transform(args['input_file'])
    mm.classify_coin_images(coin_imgs)
    print('Total coin value in zł:')
    print(postprocessing.get_total_count(coin_imgs))

    if not args['train_model']:
        mm.init_training_data(grayscale=False)
    y_te_pred = mm.model.predict(mm.x_te)
    y_actual = mm.y_te.argmax(axis=1).flatten().tolist()
    y_te_pred = y_te_pred.argmax(axis=1).flatten().tolist()
    # I'm sure there is a more elegant way
    classes = ['1,2,5gr tail', '1gr head', '1zl head', '2gr head', '2zl head', '2zl tail', '5gr head', '5zl head', '5zl tail', '10,20,50gr,1zl tail', '10gr head', '20gr head', '50gr head']
    for i in range(len(y_actual)):
        y_actual[i] = classes[y_actual[i]]
        y_te_pred[i] = classes[y_te_pred[i]]

    axes = plot_confusion_matrix(y_actual, y_te_pred)
    plt.setp(axes.get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.show()
