# =======================
# CLASSIFICATION PROBLEM
# =======================

"""
https://youtu.be/AhC6r4cwtq0?list=PLXO45tsB95cJHXaDKpbwr5fC_CCYylw1f
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data  # imports the samples data


def add_layer(inputs, in_size, out_size, layer_name, activation_function=None):
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


# computes the tests on the testing set
def compute_accuracy(v_xs, v_ys):
    global prediction  # gets the prediction operation
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})  # makes a prediction on the testing data
    print(y_pre[0])
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))  # counts all the correct predicitons
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # calculate the accuracy
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})  # runs the accuracy calculation operation
    return result


# static parameters
number_of_features = 784  # the input data are images 28x28 pixels (784) so 784 features
number_of_outputs = 10  # numbers 1 to 10 (number of classes), one_hot output with 10 positions
number_of_hidden_units_layer = 10
learning_rate = 0.5
num_epochs = 1000
num_training_samples = 200

# import the data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)  # numbers from 1 to 10

# define the placeholders for the inputs of the network
xs = tf.placeholder(tf.float32, [None, number_of_features])
ys = tf.placeholder(tf.float32, [None, number_of_outputs])

# add hidden layer
hidden_layer_output = add_layer(xs, number_of_features, number_of_hidden_units_layer, "hidden",
                                activation_function=tf.nn.sigmoid)

#  add output layer (adding only one layer)
# softmax: calculates the probability of each class and retrieves the highest one
# The softmax function is a more generalized logistic activation function which is used for multiclass classification.
prediction = add_layer(hidden_layer_output, number_of_hidden_units_layer, number_of_outputs, "output",
                       activation_function=tf.nn.softmax)

# the prediction must be considere as: [0.1,0.02,0.3,0.8,0.12,...] = [0,0,0,1,0,...]

# compute the error between prediction and real data trhough the cross entropy
loss = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), axis=[1]))

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

init = tf.initialize_all_variables()

# start the session
sess = tf.Session()
sess.run(init)

for i in range(num_epochs):
    # consider train data and test data separately
    # this generates a training way which uses a stochastic gradient descent concept
    # because changes the training data each step
    batch_xs, batch_ys = mnist.train.next_batch(num_training_samples)  # samples set for the training

    # run the training
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 50 == 0:
        # print("COST: ", sess.run(loss, feed_dict={xs: batch_xs, ys: batch_ys}))
        # the accuracy calculation is carried on a different set of images
        print("ACCURACY: ", compute_accuracy(mnist.test.images, mnist.test.labels) * 100, "%")
