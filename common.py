FONT_DIR = "./fonts"
FONT_HEIGHT = 32  # Pixel size to which the chars are resized
NUMS = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
CHARS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
SPACE = [' ']
JOIN = NUMS + CHARS + SPACE
LICENSE_MAX_LEN = 7

OUTPUT_SHAPE = (64, 128)
CLASSES = ['License Plate']


def get_data_pair(train_dir, annotation_dir):
    import os
    import json
    from PIL import Image
    import numpy as np

    n_chr = len(JOIN)
    t_jsns = os.listdir(annotation_dir)
    flg_first = True
    for i in t_jsns:
        ext = i.split('.')[1]
        if ext == 'json':
            with open(annotation_dir + '/' + i, "r") as jfile:
                jdata = json.load(jfile)
                # read image
                img = Image.open(train_dir + '/' + jdata['file'])
                img = img.resize((128, 64))
                imgs = img if flg_first else np.append(imgs, img)
                # read license plate numbers
                n_ = np.pad(jdata['nums'], (0, LICENSE_MAX_LEN - len(jdata['nums'])), 'constant',
                            constant_values=(JOIN.index(' ')))
                Y_t = np.zeros((n_chr, LICENSE_MAX_LEN))
                Y_t[n_, np.arange(LICENSE_MAX_LEN)] = 1
                Y_ = Y_t if flg_first else np.append(Y_, Y_t)
                flg_first = False
    Y_ = np.split(Y_.reshape([-1, LICENSE_MAX_LEN]), LICENSE_MAX_LEN, axis=1)
    return imgs.reshape([-1, 128, 64, 1]) / 255., [i.reshape([-1, n_chr]) for i in Y_]


def model(input_shape):
    import tensorflow as tf

    n_chr = len(JOIN)

    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = tf.keras.Input(input_shape)

    # Zero-Padding: pads the border of X_input with zeroes
    X = tf.keras.layers.ZeroPadding2D((2, 2))(X_input)

    # CONV0 -> BN -> RELU -> MAXPOOL applied to X
    X = tf.keras.layers.Conv2D(48, (5, 5), strides=(1, 1), name='conv0')(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name='bn0')(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='max_pool0')(X)

    # CONV1 -> BN -> RELU -> MAXPOOL applied to X
    X = tf.keras.layers.Conv2D(64, (5, 5), strides=(1, 1), name='conv1')(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name='bn1')(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.MaxPooling2D((1, 2), strides=(2, 2), name='max_pool1')(X)

    # CONV2 -> BN -> RELU -> MAXPOOL applied to X
    X = tf.keras.layers.Conv2D(64, (5, 5), strides=(1, 1), name='conv2')(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name='bn2')(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='max_pool2')(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = tf.keras.layers.Flatten()(X)
    # ----------------------------------------------------------
    d1 = tf.keras.layers.Dense(n_chr, activation='softmax', name='d1')(X)
    d2 = tf.keras.layers.Dense(n_chr, activation='softmax', name='d2')(X)
    d3 = tf.keras.layers.Dense(n_chr, activation='softmax', name='d3')(X)
    d4 = tf.keras.layers.Dense(n_chr, activation='softmax', name='d4')(X)
    d5 = tf.keras.layers.Dense(n_chr, activation='softmax', name='d5')(X)
    d6 = tf.keras.layers.Dense(n_chr, activation='softmax', name='d6')(X)
    d7 = tf.keras.layers.Dense(n_chr, activation='softmax', name='d7')(X)

    model = tf.keras.Model(inputs=X_input,
                           outputs=[d1, d2, d3, d4, d5, d6, d7],
                           name='CNN_ANPR')

    return model
