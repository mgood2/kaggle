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
display_step = 10



#in tensorflow



print('-'*30)
print('Loading and preprocessing train data...')
print('-'*30)
imgs_train, imgs_mask_train = load_train_data()

imgs_train = preprocess(imgs_train)
imgs_mask_train = preprocess(imgs_mask_train)

imgs_train = imgs_train.astype('float32')
mean = np.mean(imgs_train)  # mean for data centering
std = np.std(imgs_train)  # std for data normalization

imgs_train -= mean
imgs_train /= std

imgs_mask_train = imgs_mask_train.astype('float32')
imgs_mask_train /= 255.  # scale masks to [0, 1]



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

def conv2d_transpose(x, W, OS, strides=1):
    return  tf.nn.conv2d_transpose(x, W, OS, strides=[1, strides, strides, 1], padding='SAME')


# Network Parameters

dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, img_rows*img_cols])
y_ = tf.placeholder(tf.float32, [None, img_rows*img_cols])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)



# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, img_cols, img_rows, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)
    # [-1, 40, 32, 32]

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)
    # [-1, 20, 16, 64]

    # Convolution Layer
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # Max Pooling (down-sampling)
    conv3 = maxpool2d(conv3, k=2)
    # [-1, 10, 8, 128]

    # Convolution Layer
    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'])
    # Max Pooling (down-sampling)
    conv4 = maxpool2d(conv4, k=2)
    # [ -1, 5, 4, 256]

    # Convolution Layer
    conv5 = conv2d(conv4, weights['wc5'], biases['bc5'])

    # Deconolution Layer
    conv6 = conv2d_transpose(conv5, weights['wc5'], outputshape['os6'])
    # Deconolution Layer
    conv7 = conv2d_transpose(conv6, weights['wc4'], outputshape['os7'])
    # Deconolution Layer
    conv8 = conv2d_transpose(conv7, weights['wc3'], outputshape['os8'])
    # Deconolution Layer
    conv9 = conv2d_transpose(conv8, weights['wc2'], outputshape['os9'])

    # Fully connected layer
    # Reshape conv9 output to fit fully connected layer input
    fc1 = tf.reshape(conv9, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.sigmoid(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 3x3 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 4, 1, 32])),
    # 3x3 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 4, 32, 64])),
    # 3x3 conv, 64 inputs, 128 outputs
    'wc3': tf.Variable(tf.random_normal([5, 4, 64, 128])),
    # 3x3 conv, 128 inputs, 256 outputs
    'wc4': tf.Variable(tf.random_normal([5, 4, 128, 256])),
    # 3x3 conv, 256 inputs, 512 outputs
    'wc5': tf.Variable(tf.random_normal([5, 4, 256, 512])),
    # fully connected, 512*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([5*4*4096, 1024])),
    # fully connected, 4*5*64 inputs, 1024 outputs
    'wd2': tf.Variable(tf.random_normal([4*5*64, 1024])),
    # 1024 inputs, img_rows*img_cols outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, img_rows*img_cols]))
}
outputshape = {
    'os6': [BATCH_SIZE, 5, 4, 256],
    'os7': [BATCH_SIZE, 5, 4, 128],
    'os8': [BATCH_SIZE, 5, 4,  64],
    'os9': [BATCH_SIZE, 5, 4,  32]
}
biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bc3': tf.Variable(tf.random_normal([128])),
    'bc4': tf.Variable(tf.random_normal([256])),
    'bc5': tf.Variable(tf.random_normal([512])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([img_rows*img_cols]))
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
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: DROPOUT})
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
