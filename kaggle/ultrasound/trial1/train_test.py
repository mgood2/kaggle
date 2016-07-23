from __future__ import print_function

import cv2
import numpy as np
import tensorflow as tf

from data import load_train_data, load_test_data

# Parameters

batch_size = 128
test_size = 256



#preprocessing image size
img_rows = 64
img_cols = 80
smooth = 1.
def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], imgs.shape[1], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i, 0] = cv2.resize(imgs[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
    return imgs_p

print('-'*30)
print('Loading and preprocessing train data...')
print('-'*30)
imgs, imgs_mask = load_train_data()

imgs = preprocess(imgs)
imgs_mask = preprocess(imgs_mask)

imgs = imgs.astype('float32')
mean = np.mean(imgs)  # mean for data centering
std = np.std(imgs)  # std for data normalization

imgs -= mean
imgs /= std

imgs_mask = imgs_mask.astype('float32')
imgs_mask /= 255.  # scale masks to [0, 1]


def dice_coef(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def init_weight(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w1, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):

    l1a = tf.nn.relu(tf.nn.conv2d(X, w,                       # l1a shape=(?, 64, 80, 32)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],              # l1 shape=(?, 14, 14, 32)
                        strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2,                     # l2a shape=(?, 14, 14, 64)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],              # l2 shape=(?, 7, 7, 64)
                        strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)

    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3,                     # l3a shape=(?, 7, 7, 128)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],              # l3 shape=(?, 4, 4, 128)
                        strides=[1, 2, 2, 1], padding='SAME')
    l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])    # reshape to (?, 2048)
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.relu(tf.matmul(l3, w4))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    pyx = tf.matmul(l4, w_o)
    return pyx


trX, trY, teX, teY = imgs[0:imgs.shape[0]*8/10], imgs_mask[0:imgs.shape[0]*8/10], imgs[imgs.shape[0]*8/10+1:imgs.shape[0]], imgs_mask[imgs.shape[0]*8/10+1:imgs.shape[0]]
trX = tf.transpose(trX,(0,2,3,1))
teX = tf.transpose(teX,(0,2,3,1))
trY = tf.transpose(trY,(0,2,3,1))
teY = tf.transpose(teY,(0,2,3,1))

X = tf.placeholder(tf.float32, [None, img_rows, img_cols, 1])
Y = tf.placeholder(tf.float32, [None, img_rows, img_cols, 1])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

w = init_weights([3, 3, 1, 32])       # 3x3x1 conv, 32 outputs
w2 = init_weights([3, 3, 32, 64])     # 3x3x32 conv, 64 outputs
w3 = init_weights([3, 3, 64, 128])    # 3x3x32 conv, 128 outputs
w4 = init_weights([128 * 4 * 4, 625]) # FC 128 * 4 * 4 inputs, 625 outputs
w_o = init_weights([625, img_rows * img_cols])         # FC 625 inputs, 10 outputs (labels)

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)

# Define loss and optimizer
cost = dice_coef_loss(py_x, Y)

train_op  = tf.train.AdamOptimizer(0.0001).minimize(cost)
predict_op = tf.argmax(py_x, 1)


print('-'*30)
print('Loading and preprocessing test data...')
print('-'*30)
imgs_test, imgs_id_test = load_test_data()
imgs_test = preprocess(imgs_test)

imgs_test = imgs_test.astype('float32')
imgs_test -= mean
imgs_test /= std

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()

    for i in range(100):
        training_batch = zip(range(0, len(trX), batch_size),
                             range(batch_size, len(trX), batch_size))
        for start, end in training_batch:
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          p_keep_conv: 0.8, p_keep_hidden: 0.5})

        test_indices = np.arange(len(teX)) # Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]

        print(i, np.mean(np.argmax(teY[test_indices], axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX[test_indices],
                                                         Y: teY[test_indices],
                                                         p_keep_conv: 1.0,
                                                         p_keep_hidden: 1.0})))
