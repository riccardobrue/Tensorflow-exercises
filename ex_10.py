# =======================
# CONVOLUTIONAL NEURAL NETWORK (CNN)
# =======================

"""
https://youtu.be/hSDrJM6fJFM?list=PLXO45tsB95cJHXaDKpbwr5fC_CCYylw1f
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data  # imports the samples data


def add_layer(inputs, in_size, out_size, layer_name, activation_function=None):
    weights = tf.Variable(tf.random_normal([in_size, out_size], name="W"))
    biases = tf.Variable(tf.zeros([1, out_size], name="b") + 0.1)
    Wx_plus_b = tf.matmul(inputs, weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


# computes the tests on the testing set
def compute_accuracy(v_xs, v_ys):
    global prediction  # gets the prediction operation
    # we have to add the keep probability = 1 for the accuracy calculation
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})  # makes a prediction on the testing data
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))  # counts all the correct predicitons
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # calculate the accuracy
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})  # runs the accuracy calculation operation
    return result


# to get the weights in the model
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# to get the biases in the model
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# this is the convolutional layer
"""
https://classroom.udacity.com/courses/ud730/lessons/6377263405/concepts/64063017560923#
"""


def conv2d(x, W):
    # stride[1,x_movement,y_movement,1] : x and y movements are how many pixels are jumped by the filter each step
    # Must have strides[0]=strides[3]=1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


# decreases the width and the haight (pooling method)
def max_pool_2x2(x):
    # stride[1,x_movement,y_movement,1]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    # [NOTE: *1]: 2 means that decreased the size of the width and of the height by 2


# static parameters
number_of_features = 784  # the input data are images 28x28 pixels (784) so 784 features
number_of_outputs = 10  # numbers 1 to 10 (number of classes), one_hot output with 10 positions
learning_rate = 0.0001  # 1e-4
num_epochs = 1000
num_training_samples = 100

# import the data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)  # numbers from 1 to 10

# define the placeholders
xs = tf.placeholder(tf.float32, [None, number_of_features])
ys = tf.placeholder(tf.float32, [None, number_of_outputs])
keep_prob = tf.placeholder(tf.float32)

# we have to reshape the inputs into 28x28x1 (1 is the depth "input size channel", 1 means grayscale, 3 is RGB)
# -1 is the batch size: we are going to resize the last three dimensions and last the first one
# the batch size is the number of samples
x_image = tf.reshape(xs, [-1, 28, 28, 1])

# -------------------------------------------
# creation of the neural network layers
# convolutional layer 1 (conv1)
# we are going to give a shape to the W_conv1, which is a filter
W_conv1 = weight_variable([5, 5, 1, 32])  # patch 5x5 (the filter size "lens"), input size: 1, output size: 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # output size 28x28x32
h_pool1 = max_pool_2x2(h_conv1)  # output size 14x14x32: image dimension decreased by 2 [NOTE: *1]

# convolutional layer 2 (conv2)
W_conv2 = weight_variable([5, 5, 32, 64])  # now the out size is 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # output size is 14x14x64
h_pool2 = max_pool_2x2(h_conv2)  # output size 7x7x64

# create the fully connected layers after the convolutions ones
# fully connected layer 1 (fc1)
W_fc1 = weight_variable([7 * 7 * 64, 1024])  # this is a 2D
b_fc1 = bias_variable([1024])
# [n_samples,7,7,64] --> [n_samples,7*7*64] reshape
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# fully connected layer 2 (fc2)
W_fc2 = weight_variable([1024, number_of_outputs])
b_fc2 = bias_variable([number_of_outputs])
prediction = tf.nn.softmax(
    tf.matmul(h_fc1_drop, W_fc2) + b_fc2)  # softmax is used whenever we need to make a class prediction

# -------------------------------------------

# calculate the loss by cross entropy
# the prediction is from the fully connected layer 2 (fc2)
loss = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), axis=[1]))

# create the optimizer (with the adam optimizer algorithm)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# create the session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# training
for i in range(num_epochs):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
        print("ACCURACY: ", compute_accuracy(mnist.test.images, mnist.test.labels) * 100, "%")

sess.close()
# ==== # òòò