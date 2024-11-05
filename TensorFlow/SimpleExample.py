import numpy as np
import tensorflow as tf

w = tf.Variable(0, dtype=tf.float32)  # initialize to 0
optimizer = tf.keras.optimizers.legacy.Adam(0.1)  # set learning rate to 0.1 （legacy runs faster)

# Define the cost function
cost = w ** 2 - 10 * w + 25

# Note: the beautiful thing about tensorflow is you only need to write the code for forward prop.
