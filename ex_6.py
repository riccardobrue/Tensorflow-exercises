# =======================
# VISUALIZATION OF A TENSORBOARD (GRAPH and HISTOGRAM)
# =======================

"""
tutorial 14 (https://youtu.be/FtxpjxFi2vk?list=PLXO45tsB95cJHXaDKpbwr5fC_CCYylw1f)
"""
# NOTE:
# Just taken the code from ex_5

import tensorflow as tf
import numpy as np
import os


# added a layer name to identify the layer (hidden or output)
def add_layer(inputs, in_size, out_size, layer_name, activation_function=None):
    # create an "layer" set for the tensorflow visualization
    with tf.name_scope("layer"):
        # create an "weights" set for the tensorflow visualization
        with tf.name_scope("weights"):
            weights = tf.Variable(tf.random_normal([in_size, out_size], name="W"))
            # generate a chart for the weights histogram
            tf.summary.histogram(layer_name + "/weights", weights)

        # create an "biases" set for the tensorflow visualization
        with tf.name_scope("biases"):
            biases = tf.Variable(tf.zeros([1, out_size], name="b") + 0.1)
            # generate a chart for the biases histogram
            tf.summary.histogram(layer_name + "/biases", biases)

        # create an "Wx_plus_b" set for the tensorflow visualization
        with tf.name_scope("Wx_plus_b"):
            Wx_plus_b = tf.matmul(inputs, weights) + biases
            # generate a chart for the Wx_plus_b histogram
            tf.summary.histogram(layer_name + "/Wx_plus_b", Wx_plus_b)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        # generate a chart for the outputs histogram
        tf.summary.histogram(layer_name + "/outputs", outputs)
        return outputs


x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

number_of_features = 1
number_of_outputs = 1
number_of_hidden_units_layer = 10
learning_rate = 0.1
num_epochs = 1000

# create an "inputs" set for the tensorflow visualization
with tf.name_scope("inputs"):
    xs = tf.placeholder(tf.float32, [None, number_of_features], name="x_input")
    ys = tf.placeholder(tf.float32, [None, number_of_outputs], name="y_input")

hidden_layer_output = add_layer(xs, number_of_features, number_of_hidden_units_layer, "hidden",
                                activation_function=tf.nn.relu)
prediction = add_layer(hidden_layer_output, number_of_hidden_units_layer, number_of_outputs, "output")
# create an "loss" set for the tensorflow visualization
with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction, name="square"), axis=[1], name="sum"), name="mean")
    # generate a chart for the loss scalar chart (histogram section)
    tf.summary.scalar("loss", loss)

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
# merge the summaries
merged_summary = tf.summary.merge_all()

# add a writer to store in a file the structure of this tensorflow neural network
dir_path = os.path.dirname(os.path.realpath(__file__))  # get this project's dir path
# create a writer, which so far creater the graph for this neural network structure
writer = tf.summary.FileWriter(dir_path + "/logs/", sess.graph)  # sess.graph is one of the visualization tools
# to open the file:
# cd "path_to_this_project"
# tensorboard --logdir="logs"
# go to the link provided to visualize the tensorboard

for i in range(num_epochs):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
        # create the merged summary histogram
        histogram_result = sess.run(merged_summary, feed_dict={xs: x_data, ys: y_data})
        # add the histogram merged summary to the writer
        writer.add_summary(histogram_result, i)
