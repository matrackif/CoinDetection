import argparse
import os
import numpy as np
import cv2
import model.preprocessing as preprocessing
import model.postprocessing as postprocessing
from model.model_manager import ModelManager

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

