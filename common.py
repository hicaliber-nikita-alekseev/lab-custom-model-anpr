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
                n_ = np.pad(jdata['nums'], (0, LICENSE_MAX_LEN - len(jdata['nums'])), 'constant', constant_values=(JOIN.index(' ')))
                Y_t = np.zeros((n_chr, LICENSE_MAX_LEN))
                Y_t[n_, np.arange(LICENSE_MAX_LEN)] = 1
                Y_ = Y_t if flg_first else np.append(Y_, Y_t)
                flg_first = False
    Y_ = np.split(Y_.reshape([-1, LICENSE_MAX_LEN]), LICENSE_MAX_LEN, axis=1)
    return imgs.reshape([-1, 128, 64, 1]) / 255., [i.reshape([-1, n_chr]) for i in Y_]
