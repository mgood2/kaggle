from __future__ import print_function

import cv2
import numpy as np
import tensorflow as tf

from data import load_train_data, load_test_data



import tensorflow as tf

def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], imgs.shape[1], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i, 0] = cv2.resize(imgs[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
    return imgs_p


def train_and_predict():

    img_rows = 64
    img_cols = 80
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train, imgs_mask_train = load_train_data()
    print('before preprocess size: ')
    print('imgs_train : %s' %imgs_train.size)
    print('imgs_train shape[0] : %s' %imgs_train.shape[0])
    print('imgs_train shape[1] : %s' %imgs_train.shape[1])
    print('imgs_mask_train : %s' %imgs_mask_train.size)
    print('imgs_mask_train shape[0] : %s' %imgs_mask_train.shape[0])
    print('imgs_mask_train shape[1] : %s' %imgs_mask_train.shape[1])

    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)
    print('after preprocess size: ')
    print('imgs_train : %s' %imgs_train.size)
    print('imgs_train shape[0] : %s' %imgs_train.shape[0])
    print('imgs_train shape[1] : %s' %imgs_train.shape[1])
    print('imgs_mask_train : %s' %imgs_mask_train.size)
    print('imgs_mask_train shape[0] : %s' %imgs_mask_train.shape[0])
    print('imgs_mask_train shape[1] : %s' %imgs_mask_train.shape[1])

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]


    """
    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    imgs_test, imgs_id_test = load_test_data()
    imgs_test = preprocess(imgs_test)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std
    """


if __name__ == '__main__':
    train_and_predict()
