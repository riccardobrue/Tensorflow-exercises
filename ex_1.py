# =======================
# BASIC INTRODUCTION
# =======================

import tensorflow as tf
import numpy as np

# create random data with a static function
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3
# ===================================
# create tensorflow (static) structure start (defines all the "static" operations)
# ===================================
# initialize these weights and the biases (parameters)
# tf.random_uniform: gets [x] random values between two thresholds
# [1] is the "shape" of the following variable (1 element)
weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))  # a random value between -1.0 and 1.0

# tf.zeros: creates [x] zero values
biases = tf.Variable(tf.zeros([1]))  # a single zero value

y_pred = weights * x_data + biases  # makes a prediction with the initial values

# tf.reduce_mean: computes the mean among all the array values
# tf.square: computes the squared value
cost = tf.reduce_mean(tf.square(y_pred - y_data))  # calculate the loss

# minimize the loss (optimize the model)
learning_rate = 0.5
# we are going to use the gradient descent as optimizer algorithm
optimizer = tf.train.GradientDescentOptimizer(learning_rate)

"""
https://stackoverflow.com/questions/37890394/tensorflow-gradientdescentoptimizer-how-does-it-connect-to-tf-variables
"""
# training step through gradient descent
train = optimizer.minimize(cost)  # we want to reduce the cost

# whenever we have variables (except placeholders and constants) in the tensorflow stucture, we have to initialize them
init = tf.initialize_all_variables()  # initialize all the tensorflow variables
# ===================================
# create tensorflow structure end
# start a tensorflow session
# ===================================

# create a tensorflow session
sess = tf.Session()  # is as a pointer, we have to run functions (point to them) to execute them

# sess.run(...): runs a defined "static" operation
sess.run(init)  # run the initialize function (inizialize the variable structure defined above)

# train the tensorflow structure
for step in range(201):
    sess.run(train)  # each step runs the optimizer, which recurs to the cost calculation
    if step % 20 == 0:
        # prints the current step, the current weights and the current biases
        print(step, sess.run(weights), sess.run(biases))

# remember to close the session at the end
sess.close()


# NOTES
"""
Difference between placeholders and variables
(https://stackoverflow.com/questions/36693740/whats-the-difference-between-tf-placeholder-and-tf-variable)

- use tf.Variable for trainable variables such as weights (W) and biases (B) for your model
- use tf.placeholder to feed actual training examples


The difference is that with tf.Variable you have to provide an initial value when you declare it. 
With tf.placeholder you don't have to provide an initial value and you can specify it at run time 
    with the feed_dict argument inside Session.run
"""
