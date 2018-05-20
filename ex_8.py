# =======================
# AVOID OVERFITTING WITH DROPOUT
# =======================

"""
https://youtu.be/FbMtDHXPnPc?list=PLXO45tsB95cJHXaDKpbwr5fC_CCYylw1f
"""

import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


def add_layer(inputs, in_size, out_size, layer_name, activation_function=None):
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, weights) + biases

    # add the dropout for regularization
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)

    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)

    tf.summary.histogram(layer_name + "/outputs", outputs)
    return outputs


# load data
digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)  # transforms the output into one_hot arrays
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)  # 30% of the data used as testing set

# static parameters
number_of_features = 64  # the input data are images 8x8 pixels (64) so 64 features
number_of_outputs = 10  # numbers 1 to 10 (number of classes), one_hot output with 10 positions
number_of_hidden_units_layer = 50
learning_rate = 0.5
num_epochs = 1000
num_training_samples = 100

# define the placeholders for the inputs of the network
xs = tf.placeholder(tf.float32, [None, number_of_features])
ys = tf.placeholder(tf.float32, [None, number_of_outputs])
# define the placeholder for the keep probability
keep_prob = tf.placeholder(tf.float32)

# add output layer only
hidden_layer_output = add_layer(xs, number_of_features, number_of_hidden_units_layer, "hidden",
                                activation_function=tf.nn.tanh)
prediction = add_layer(hidden_layer_output, number_of_hidden_units_layer, number_of_outputs, "output",
                       activation_function=tf.nn.softmax)

loss = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), axis=[1]))
tf.summary.scalar("loss", loss)

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# init = tf.initialize_all_variables() # for tensorflow <0.12
init = tf.global_variables_initializer()

# start the session
sess = tf.Session()
sess.run(init)
merged_summary = tf.summary.merge_all()

# writes the summaries
train_writer = tf.summary.FileWriter("logs/train", sess.graph)
test_writer = tf.summary.FileWriter("logs/test", sess.graph)

for i in range(num_epochs):
    # during the training we want a keep probability of 50%
    sess.run(train_step, feed_dict={xs: X_train, ys: y_train, keep_prob: 1})
    if i % 50 == 0:
        # during the tests we want to take the whole neural network, so keep probability = 100%
        train_result = sess.run(merged_summary, feed_dict={xs: X_train, ys: y_train, keep_prob: 1})
        test_result = sess.run(merged_summary, feed_dict={xs: X_test, ys: y_test, keep_prob: 1})

        train_writer.add_summary(train_result, i)  # this is practically the train loss
        test_writer.add_summary(test_result, i)

# To run tensorflow:
# tensorboard --logdir="logs"
