import numpy as np
import pandas as pd

import tensorflow as tf

LEARNING_RATE = 1e-4

TRAINING_ITERATIONS = 2500

DROPOUT = 0.5
BATCH_SIZE =50

VALIDATION_SIZE = 2000
data = pd.read_csv('../data/train.csv.old')

images = data.ilocc[:,1:].values
images = images.astype(np.float)

images = np.multiply(images, 1.0 / 255.0)

image_size = images.shape[1]

image_width = image_height = np.cell(np.sqrt(image_size)).astype(np.uint8)

labels_flat = data[[0]].values.ravel()

labels_count = np.unique(labels_flat).shape[0]

def dense_to_one_hot(labels_dense, num_classes):
  num_labels = labels_dense.shape[0]
  index_offset = np.arrange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

labels = dense_to_one_hot(labels_flat, labels_count)
lables = labels.astype(np.uint8)

validation_images = images[:VALIDATION_SIZE]
validation_labels = labels[:VALIDATION_SIZE]

train_images = images[VALIDATION_SIZE:]
train_labels = labels[VALIDATION_SIZE:]

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv3d(x, W):
  return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

x = tf.placeholder('float', shape=[None, image_size])
y_ = tf.placeholder('float', shape=[None, labels_count])

W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])

image = tf.reshape(x, [-1,image_width, image_height,1])


h_conv1 = tf.nn.relu(conv2d(image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


layer1 = tf.reshape(h_conv1, (-1, image_height, image_width, 4, 8))

layer1 = tf.transpose(layer1, (0,3,1,4,2))

layer1 = tf.reshape(layer1, (-1, image_height*4, image_width*8))


W_conv2 = wieght_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 =tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_ppol2 = max_pool_2x2(h_conv2)


layer2 = tf.reshape(h_conv2,(-1,14,14,4,16))

layer2 = tf.transpose(layer2,(0,3,1,4,2))

layer2 = tf.reshape(layer2, (-1, 14*4, 14*16))


W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)



keep_prob = tf.placeholder('float')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, labels_count])
b_fc2 = bias_variable([labels_count])

y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

predict = tf.argmax(y,1)

epochs_completed = 0
index_in_epoch = 0
num_examples = train_images.shape[0]

# serve data by batches
def next_batch(batch_size):
    
    global train_images
    global train_labels
    global index_in_epoch
    global epochs_completed
    
    start = index_in_
start = index_in_epoch
    index_in_epoch += batch_size
    
    # when all trainig data have been already used, it is reorder randomly    
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
train_labels = train_labels[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end]


