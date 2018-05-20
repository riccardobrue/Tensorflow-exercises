# =======================
# CONSTANTS AND SESSIONS
# =======================

import tensorflow as tf

# tf.constant: defines a constant in the tensorflow structure
matrix1 = tf.constant([[3, 3]])  # 1x2 matrix (1 row x 2 columns)
matrix2 = tf.constant([[2], [2]])  # 2x1 matrix (2 rows x 1 column)

# to multiply two matrices (a,b) x (c,d) -> b==c and the result is an (a,d) matrix

# tf.matmul: multiplies two matrices
product = tf.matmul(matrix1, matrix2)  # this is similar to numpy.dot(m1,m2)

# usage of tensorflow sessions
# method 1
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()

# method 2
# we don't have to call the sess.close() manually
with tf.Session() as sess2:
    result2 = sess2.run(product)
    print(result2)
