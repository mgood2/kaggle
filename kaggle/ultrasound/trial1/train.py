from __future__ import print_function

import cv2
import numpy as np
import tensorflow as tf

from data import load_train_data, load_test_data



import tensorflow as tf

#preprocessing image size
img_rows = 64
img_cols = 80

smooth = 1.
def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], imgs.shape[1], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i, 0] = cv2.resize(imgs[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
    return imgs_p

# Parameters
LEARNING_RATE = 0.001
TRAINING_ITERATIONS = 20000
BATCH_SIZE = 128

VALIDATION_SIZE = 2000



#in tensorflow



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

validation_images = imgs[:VALIDATION_SIZE]
validation_labels = imgs_mask[:VALIDATION_SIZE]

imgs_train = imgs[VALIDATION_SIZE:]
imgs_mask_train = imgs_mask[VALIDATION_SIZE:]

def dice_coef(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

def conv2d_transpose(x, W, b, OS, strides=1):
    x = tf.nn.conv2d_transpose(x, W, OS, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

# Network Parameters

DROPOUT = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, 1, img_rows, img_cols])
y_ = tf.placeholder(tf.float32, [None, 1, img_rows, img_cols])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)



# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1,img_rows, img_cols, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    pool1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(pool1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    pool2 = maxpool2d(conv2, k=2)

    # Convolution Layer
    conv3 = conv2d(pool2, weights['wc3'], biases['bc3'])
    # Max Pooling (down-sampling)
    pool3 = maxpool2d(conv3, k=2)

    # Convolution Layer
    conv4 = conv2d(pool3, weights['wc4'], biases['bc4'])
    # Max Pooling (down-sampling)
    pool4 = maxpool2d(conv4, k=2)

    # Convolution Layer
    conv5 = conv2d(pool4, weights['wc5'], biases['bc5'])



    # Deconolution Layer
    conv6 = conv2d_transpose(conv5, weights['wc5'],  biases['bc4'], outputshape['os6'])
    # Pooling Up-sampling
    pool6 = conv2d_transpose(conv6, weights['pl6'],  biases['bc4'],  outputshape['up6'], strides=2)
    # Deconolution Layer
    conv7 = conv2d_transpose(pool6, weights['wc4'],  biases['bc3'], outputshape['os7'])
    # Pooling Up-sampling
    pool7 = conv2d_transpose(conv7, weights['pl7'],  biases['bc3'], outputshape['up7'], strides=2)
    # Deconolution Layer
    conv8 = conv2d_transpose(pool7, weights['wc3'],  biases['bc2'], outputshape['os8'])
    # Pooling Up-sampling
    pool8 = conv2d_transpose(conv8, weights['pl8'],  biases['bc2'], outputshape['up8'], strides=2)
    # Deconolution Layer
    conv9 = conv2d_transpose(pool8, weights['wc2'],  biases['bc1'], outputshape['os9'])
    # Pooling Up-sampling
    pool9 = conv2d_transpose(conv9, weights['pl9'],  biases['bc1'], outputshape['up9'], strides=2)

    conv10 = conv2d_transpose(pool9, weights['wc1'], biases['bc0'],outputshape['os10'])


    out = tf.reshape(conv10,[-1, 1, img_rows, img_cols])
    return out

# Store layers weight & bias
weights = {
    'wc1': tf.Variable(tf.random_normal([4, 5, 1, 32])),
    'wc2': tf.Variable(tf.random_normal([4, 5, 32, 64])),
    'wc3': tf.Variable(tf.random_normal([4, 5, 64, 128])),
    'wc4': tf.Variable(tf.random_normal([4, 5, 128, 256])),
    'wc5': tf.Variable(tf.random_normal([4, 5, 256, 512])),
    'wc6': tf.Variable(tf.random_normal([4, 5, 256, 512])),
    'wc7': tf.Variable(tf.random_normal([4, 5, 128, 256])),
    'wc8': tf.Variable(tf.random_normal([4, 5, 64, 128])),
    'wc9': tf.Variable(tf.random_normal([4, 5, 32, 64])),
    'wc10': tf.Variable(tf.random_normal([4, 5, 1, 32])),
    'pl6': tf.Variable(tf.random_normal([2, 2, 256, 256])),
    'pl7': tf.Variable(tf.random_normal([2, 2, 128, 128])),
    'pl8': tf.Variable(tf.random_normal([2, 2, 64, 64])),
    'pl9': tf.Variable(tf.random_normal([2, 2, 32, 32]))
}
outputshape = {
    'os6': [BATCH_SIZE, 4, 5, 256],
    'up6': [BATCH_SIZE, 8,10, 256],
    'os7': [BATCH_SIZE, 8,10, 128],
    'up7': [BATCH_SIZE,16,20, 128],
    'os8': [BATCH_SIZE,16,20,  64],
    'up8': [BATCH_SIZE,32,40,  64],
    'os9': [BATCH_SIZE,32,40,  32],
    'up9': [BATCH_SIZE,64,80,  32],
    'os10':[BATCH_SIZE,64,80,   1]
}
biases = {
    'bc0': tf.Variable(tf.random_normal([1])),
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bc3': tf.Variable(tf.random_normal([128])),
    'bc4': tf.Variable(tf.random_normal([256])),
    'bc5': tf.Variable(tf.random_normal([512])),
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = dice_coef_loss(y_, pred)
optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)


# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# serve data by batches

epochs_completed = 0
index_in_epoch = 0

num_examples =imgs_train.shape[0]
def next_batch(BATCH_SIZE):

    global imgs_train
    global imgs_mask_train
    global index_in_epoch
    global epochs_completed
    global num_examples

    start = index_in_epoch
    index_in_epoch += BATCH_SIZE

    # when all trainig data have been already used, it is reorder randomly
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        imgs_train = imgs_train[perm]
        imgs_mask_train = imgs_mask_train[perm]
        # start next epoch
        start = 0
        index_in_epoch = BATCH_SIZE
        assert BATCH_SIZE <= num_examples
    end = index_in_epoch
    return imgs_train[start:end], imgs_mask_train[start:end]





print('-'*30)
print('Creating and compiling model...')
print('-'*30)



print('-'*30)
print('Fitting model...')
print('-'*30)
init = tf.initialize_all_variables()
sess = tf.InteractiveSession()

sess.run(init)
train_accuracies = []
validation_accuracies = []
x_range = []

display_step=1

for i in range(TRAINING_ITERATIONS):

    #get new batch
    batch_xs, batch_ys = next_batch(BATCH_SIZE)

    # check progress on every 1st,2nd,...,10th,20th,...,100th... step
    if i%display_step == 0 or (i+1) == TRAINING_ITERATIONS:

        train_accuracy = accuracy.eval(feed_dict={x:batch_xs,
                                                  y_: batch_ys,
                                                  keep_prob: 1.0})
        if(VALIDATION_SIZE):
            validation_accuracy = accuracy.eval(feed_dict={ x: validation_images[0:BATCH_SIZE],
                                                            y_: validation_labels[0:BATCH_SIZE],
                                                            keep_prob: 1.0})
            print('training_accuracy / validation_accuracy => %.2f / %.2f for step %d'%(train_accuracy, validation_accuracy, i))

            validation_accuracies.append(validation_accuracy)
            x_range.append(i)
        else:
             print('training_accuracy => %.4f for step %d'%(train_accuracy, i))
        train_accuracies.append(train_accuracy)

        # increase display_step
        if i%(display_step*10) == 0 and i:
            display_step *= 10
    # train on batch
    sess.run(optimizer, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: DROPOUT})
# After training is done, it's good to check accuracy on data that wasn't used in training.
# check final accuracy on validation set
if(VALIDATION_SIZE):
    validation_accuracy = accuracy.eval(feed_dict={x: validation_images,
                                                   y_: validation_labels,
                                                   keep_prob: 1.0})
    print('validation_accuracy => %.4f'%validation_accuracy)

print('-'*30)
print('Loading and preprocessing test data...')
print('-'*30)
imgs_test, imgs_id_test = load_test_data()
imgs_test = preprocess(imgs_test)

imgs_test = imgs_test.astype('float32')
imgs_test -= mean
imgs_test /= std

print('-'*30)
print('Predicting masks on test data...')
print('-'*30)
print('imgs_test({0[0]},{0[1]})'.format(test_images.shape))

imgs_mask_test = np.zeros(imgs_test.shape[0])

for i in range(0,imgs_test.shape[0]//BATCH_SIZE):
    imgs_mask_test[i*BATCH_SIZE : (i+1)*BATCH_SIZE] = predict.eval(feed_dict={x: imgs_test[i*BATCH_SIZE : (i+1)*BATCH_SIZE], keep_prob: 1.0})

print('imgs_mask_test({0})'.format(len(imgs_mask_test)))

np.save('imgs_mask_test.npy', imgs_mask_test)

sess.close()
