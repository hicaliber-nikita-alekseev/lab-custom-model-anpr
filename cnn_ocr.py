import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.layers import concatenate
import os
import argparse

import json
from PIL import Image

import keras.backend as K
K.set_image_data_format('channels_last')
import tensorflow as tf

from tensorflow.contrib.eager.python import tfe

tf.enable_eager_execution()
tf.set_random_seed(0)

from common import JOIN, IMAGE_WIDTH, IMAGE_HEIGHT, get_data_pair, model
n_chr=len(JOIN)

def parse_args():
    
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--verbose', type=int, default=1)
    
    # data directories
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument('--train_annotation', type=str, default=os.environ.get('SM_CHANNEL_TRAIN_ANNOTATION'))
    parser.add_argument('--validation_annotation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION_ANNOTATION'))
    
    # model directory: we will use the default set by SageMaker, /opt/ml/model
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    
    return parser.parse_known_args()


if __name__ == "__main__":

    args, _ = parse_args()
    print('args test: learning_rate = {}, batch_size = {}, epochs = {}'.format(args.learning_rate, args.batch_size, args.epochs))
    print(args)
    
    # change the path to use '/opt/ml/input/data/{channel}/...'
    t_imgs, t_annotations = get_data_pair(args.train, args.train_annotation)
    v_imgs, v_annotations = get_data_pair(args.validation, args.validation_annotation)
    
    input_shape = (IMAGE_WIDTH,IMAGE_HEIGHT,1)
    model_k = model(input_shape)
        
    # use hyperparameters ex) args.learning_rate
    model_k.compile(optimizer=tf.train.AdamOptimizer(args.learning_rate), 
            loss=['categorical_crossentropy','categorical_crossentropy','categorical_crossentropy','categorical_crossentropy',
                   'categorical_crossentropy','categorical_crossentropy','categorical_crossentropy'],
            metrics = ["accuracy"])
        
    # use hyperparameters ex) args.epochs, args.batch_size, etc.
    model_k.fit(t_imgs, [i.reshape([-1, n_chr]) for i in t_annotations],
                validation_data=(v_imgs, [i.reshape([-1, n_chr]) for i in v_annotations]),
                epochs=args.epochs, batch_size=args.batch_size, verbose=args.verbose)

    scores = model_k.evaluate(v_imgs, [i for i in v_annotations], batch_size=args.batch_size, verbose=1, sample_weight=None)
    print("scores: ", scores)
    
    # save checkpoint for locally loading in notebook
    saver = tfe.Saver(model_k.variables)
    saver.save(args.model_dir + '/weights.ckpt')
    
    # create a separate SavedModel for deployment to a SageMaker endpoint with TensorFlow Serving
    tf.contrib.saved_model.save_keras_model(model_k, args.model_dir)

            