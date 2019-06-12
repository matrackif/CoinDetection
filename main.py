import argparse
import os
import numpy as np
import cv2
import model.preprocessing as preprocessing
import model.postprocessing as postprocessing
from model.coin_image import CoinImage
from model.model_manager import ModelManager
from glob import glob
from pathlib import Path
DEFAULT_EPOCH_COUNT = 100
DEFAULT_IMAGE_HEIGHT_WIDTH = 100
DEFAULT_MODEL_FILENAME = 'coin_det_model.h5'


if __name__ == '__main__':
    """
    Example arguments:
    
    Train greyscale model and save it:
    main.py -t -s -n 6 -g -mfn "greyscale_model.h5" -s
    
    Classify using greyscale model:
    main.py -d -f -show -g -mfn "greyscale_model.h5"
    
    Train color image model and save it:
    main.py -t -s -n 6
    
    Classify using color model:
    main.py -f -d -show
    
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train-model', action='store_true', default=False,
                            help='Train Keras baseline model')
    parser.add_argument('-s', '--save-model', action='store_true', default=False,
                        help='Save trained model, if -t is not passed then this argument is ignored')
    parser.add_argument('-show', '--show-images', action='store_true', default=False,
                        help='Show the images before classifying them')
    parser.add_argument('-n', '--epoch-count', nargs='?', type=int, default=DEFAULT_EPOCH_COUNT,
                        const=DEFAULT_EPOCH_COUNT, help='Number of epochs to train the CNN')
    parser.add_argument('-w', '--img-width', nargs='?', type=int, default=DEFAULT_IMAGE_HEIGHT_WIDTH,
                        const=DEFAULT_IMAGE_HEIGHT_WIDTH,
                        help='All input images will have their width resized to this number')
    parser.add_argument('-ht', '--img-height', nargs='?', type=int, default=DEFAULT_IMAGE_HEIGHT_WIDTH,
                        const=DEFAULT_IMAGE_HEIGHT_WIDTH,
                        help='All input images will have their height resized to this number')
    parser.add_argument('-f', '--input-file', nargs='?', type=str,
                        const='data/images_of_multiple_coins/coins.jpg',
                        help='Input file of image to be classified')
    parser.add_argument('-mfn', '--model-file', nargs='?', type=str, default=DEFAULT_MODEL_FILENAME,
                        const=DEFAULT_MODEL_FILENAME,
                        help='File name of .h5 file where Keras model will be saved to or loaded from')
    parser.add_argument('-d', '--directory', nargs='?', type=str,
                        const='data/validationSet',
                        help='Directory of coins to be classified')
    parser.add_argument('-g', '--greyscale', action='store_true', default=False,
                        help='Classify or train on greyscale images, if False then use RGB color scheme')
    args = vars(parser.parse_args())

    mm = ModelManager(args=args)
    mm.get_model()
    if args['directory'] is not None:
        validation_dir_path = Path(args['directory'])
        if validation_dir_path.is_dir():
            files = glob(os.path.join(args['directory'], '*.jpg'))
            imgs = []
            for f in files:
                if args['greyscale']:
                    img = CoinImage(img_arr=cv2.imread(f, cv2.IMREAD_GRAYSCALE))
                else:
                    img = CoinImage(img_arr=cv2.imread(f, cv2.IMREAD_COLOR))
                imgs.append(img)
            mm.classify_coin_images(imgs)
        else:
            print('Directory of validation set:', args['directory'], 'does not exist')

    if args['input_file'] is not None:
        input_file_path = Path(args['input_file'])
        if input_file_path.is_file():
            coin_imgs = preprocessing.hough_transform(args['input_file'], args['greyscale'])
            mm.classify_coin_images(coin_imgs)
            print('Total coin value in zł:')
            print(postprocessing.get_total_count(coin_imgs))
        else:
            print('Input file:', args['input_file'], 'does not exist')


