# =======================
# PLACEHOLDERS
# =======================

import tensorflow as tf

# tf.placeholder: similar to the variable, but every time they hold a different value
# create two placeholders of the float32 type
matrix_input1 = tf.placeholder(tf.float32, [1, 3])  # indicates also the shape: 1 rows and 3 columns
matrix_input2 = tf.placeholder(tf.float32, [3, 1])  # placeholder with shape of 3 rows and 1 column

# another writing form is: tf.placeholder(tf.float32, shape=(1, 3))
# if we put None as a dimension in the shape, we can have any number of that dimension

input1 = tf.placeholder(tf.float32)  # this is a single float32 value
input2 = tf.placeholder(tf.float32)  # this is a single float32 value

simple_output = tf.multiply(input1, input2)  # multiplication of two values
matrix_output = tf.matmul(matrix_input1, matrix_input2)  # multiplication of two matrices

with tf.Session() as sess:
    # we must provide the placeholders values with "feed_dict={placeholder_name:value}"
    simple_result = sess.run(simple_output, feed_dict={input1: [7], input2: [2]})
    print(simple_result)
    matrix_result = sess.run(matrix_output, feed_dict={matrix_input1: [[1, 2, 5]], matrix_input2: [[2], [1], [3]]})
    print(matrix_result)
