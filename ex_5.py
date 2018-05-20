"""
https://youtu.be/Kd7gDHY_OUU?list=PLXO45tsB95cJHXaDKpbwr5fC_CCYylw1f
"""
# =======================
# ACTIVATION FUNCTION, OPTIMIZERS AND SIMPLE NEURAL NETWORK
# =======================

# activation function:
# x axis: represents the data passed from the previous layer
# y axis: represents the data which will be passed to the next layer

# there are several activation/transfer functions in tensorflow:
# relu, relu6, elu, softplus, softsign, dropout, bias_add, sigmooid, tanh


# there are several optimizers in tensorflow:
# GradientDescentOptimizer, AdadeltaOptimizer, AdagradOptimizer, MomentumOptimizer, ...
# ..., AdamOptimizer, FtrlOptimizer, RMSPropOptimizer


# -----------------------

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# adding a hidden layer to the neural network structure
# inputs from the last layer and its size (layer units) and how many out units there are after this layer
# so far we do not add any activation function
# this returns the computer values from the added layer
def add_layer(inputs, in_size, out_size, activation_function=None):
    # tf.random_normal: uses the normal distributed random values
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    # alternative: weights = tf.Variable(tf.random_uniform([in_size, out_size], -1.0, 1.0))

    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)

    Wx_plus_b = tf.matmul(inputs, weights) + biases

    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


# make up some real data
# np.linespace: generates numbers which fit into an interval with a given sample dimension
# [:, np.newaxis] organizes the generated data into a matrix of 300 rows and 1 column
# np.newaxis increased the dimension  of the existing array by one more dimension
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]  # creates 300 samples between -1 and 1 in matrix (300,1)
noise = np.random.normal(0, 0.05, x_data.shape)  # add a little bit of noise to the data to make it look a real data
y_data = np.square(x_data) - 0.5 + noise  # function which creates a y data, just created by our own

number_of_features = 1  # we only one feature (300x1) where 1 is the number of columns, so it is the number of features
number_of_outputs = 1  # we have only 1 output value (we are going to predict only one number)
number_of_hidden_units_layer = 10  # just pick a value for the number of units in the hidden layer

learning_rate = 0.1
num_epochs = 10000
# visualize the data
# plt.scatter(x_data, y_data)
# plt.show()

# define the tensorflow structure

# define the placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, number_of_features])  # None is used because we don't known the number of samples
ys = tf.placeholder(tf.float32, [None, number_of_outputs])

# add hidden layer
# tf.nn.relu: is a non-linear function
hidden_layer_output = add_layer(xs, number_of_features, number_of_hidden_units_layer, activation_function=tf.nn.relu)

# add output layer
# now the hidden layer is the new input (because add_layer return the computed output)
# None activation_function because we want to predict a value (regression problem)
prediction = add_layer(hidden_layer_output, number_of_hidden_units_layer, number_of_outputs)

# compute the error (called also loss or cost)
# axis (or the deprecated reduction_indices) =[1] is the dimension to reduce
# dimension: 0: y, 1: x, 2: z .... (dimension 2 is for 3D arrays and so on ...)
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), axis=[1]))  # sum over the dimension 1 (columns)
# the loss is computed by squaring the error (ys-prediction) and then sum all the samples together and compute a mean

# create the training operation
# note that there are a lot of optimizers, gradient descent is the basic one
# the optimizer knows which Variable to work on (Variables, not placeholders)
"""
https://stackoverflow.com/questions/44210561/how-do-backpropagation-works-in-tensorflow
"""
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# initialize the variables of the tensorflow structure
init = tf.initialize_all_variables()

# create the tensorflow session
sess = tf.Session()
sess.run(init)

# plot the real data
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)  # 1 row, 1 column, 1 figure
ax.scatter(x_data, y_data)

# state that the chart visualization doesn't block the execution of the program
# method 1
plt.ion()  # creates an unblocking chart
plt.show()
# method 2: old versions
# plt.show(block=False)

# train the neural network
for i in range(num_epochs):
    # we have to pass the variables for all the required placeholders
    # there is an internal "recursion" through the operations
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        # print the loss every 50 epochs, the loss should become smaller each epoch
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
        # we want to show the predicted values
        # the prediction operation is related only to the x_data placeholder
        prediction_values = sess.run(prediction, feed_dict={xs: x_data})

        # plot the predictions
        try:
            ax.lines.remove(lines[0])  # remove the previous lines from the chart
        except Exception:
            pass
        try:
            lines = ax.plot(x_data, prediction_values, 'r-', lw=2)  # lw: line width
            # pause the process in order to visualize the charts
            plt.pause(0.1)  # 0.1 seconds
        except Exception:
            pass

try:
    plt.waitforbuttonpress()  # wait for an action over the figure before closing the chart
except Exception:
    pass
