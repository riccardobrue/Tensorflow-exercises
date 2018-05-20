# =======================
# SAVE AND RESTORE A NEURAL NETWORK
# =======================

import tensorflow as tf
import os

# -------------------------------
# if true: stores the data into files, if false: restores from files
# -------------------------------
# to_store = True
to_store = False
# -------------------------------

dir_path = os.path.dirname(os.path.realpath(__file__))  # get this project's dir path
saved_files_path = dir_path + "/my_net/save_net.ckpt"


def store():
    # save to file
    # remember to define the same dtype and the shape when restoring
    # assume that the following are the trained weights and biases of a neural network
    weights = tf.Variable([[1, 2, 3], [1, 2, 3]], dtype=tf.float32, name="weights")
    biases = tf.Variable([[1], [2], [3]], dtype=tf.float32, name="biases")

    init = tf.global_variables_initializer()

    # creates an object which saves all the variables (not the structure of the neural network)
    # used also for restore variables
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        save_path = saver.save(sess, saved_files_path)
        print("Saved to: ", save_path)


def restore():
    # restore the variables
    # create new variables with the same shape of the stored ones, populate them with zeros
    weights = tf.Variable(tf.zeros([2, 3]), dtype=tf.float32, name="weights")  # 2 rows, 3 columns
    biases = tf.Variable(tf.zeros([3, 1]), dtype=tf.float32, name="biases")  # 3 rows, 1 column

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, saved_files_path)
        print("restored weights: ", sess.run(weights))
        print("restored biases: ", sess.run(biases))


if to_store:
    store()
else:
    restore()
