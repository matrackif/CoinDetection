import argparse
import os
from model.cnn import train_model, MODEL_FILENAME
from keras.models import load_model
DEFAULT_EPOCH_COUNT = 100
DEFAULT_IMAGE_HEIGHT_WIDTH = 150


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train-model', action='store_true', default=False,
                            help='Train Keras baseline model')
    parser.add_argument('-s', '--save-model', action='store_true', default=False,
                        help='Save trained model, if -t is not passed then this argument is ignored')
    parser.add_argument('-n', '--epoch-count', nargs='?', type=int, default=DEFAULT_EPOCH_COUNT,
                        const=DEFAULT_EPOCH_COUNT, help='Number of epochs to train the CNN')
    parser.add_argument('-w', '--img-width', nargs='?', type=int, default=DEFAULT_IMAGE_HEIGHT_WIDTH,
                        const=DEFAULT_IMAGE_HEIGHT_WIDTH,
                        help='All input images will have their width resized to this number')
    parser.add_argument('-ht', '--img-height', nargs='?', type=int, default=DEFAULT_IMAGE_HEIGHT_WIDTH,
                        const=DEFAULT_IMAGE_HEIGHT_WIDTH,
                        help='All input images will have their height resized to this number')
    args = vars(parser.parse_args())
    print('Program arguments:', args)
    model = None
    if args['train_model']:
        model = train_model(args=args, save_model=args['save_model'], show_plot=True)
    else:
        try:
            model = load_model(MODEL_FILENAME)
        except OSError:
            print('Failed to load model with name:', MODEL_FILENAME, 'in directory:', os.getcwd())
        if model is not None:
            print('Model successfully loaded from file')
