# Print out the first 4 lines of the tf.keras.datasets.mnist.load_data() dataset:

import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
for i in (x_train, y_train, x_test, y_test): print(i.shape, i[:4])
