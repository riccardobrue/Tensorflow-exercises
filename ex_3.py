# =======================
# VARIABLE USAGE
# =======================

import tensorflow as tf

state = tf.Variable(0, name='counter')  # create a variable, starting value = 0, set a name for the variable
print(state.name)

one = tf.constant(1)  # create a constant with value=1

new_value = tf.add(state, one)  # computes a sum and stores the result to new_value

update = tf.assign(state, new_value)  # assigns to the first element (state) the value of the second element (new_value)

# we have a variable, so we have to initialize the tensorflow variables
init = tf.initialize_all_variables()

# open the session to compute the "static" operations
with tf.Session() as sess:
    sess.run(init)  # initialization operation
    for _ in range(3):  # '_' is a convention as "useless" variable
        sess.run(update)
        print(sess.run(state))
