
import tensorflow as tf
import numpy as np

# write a neural network that can predict the next number in a sequence
# the input is a sequence of numbers, the output is the next number in the sequence


# create a dataset of sequences
train_x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.uint8)
train_y = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=np.uint8)

# create a model that can predict the next number in a sequence
model = tf.keras.models.Sequential()
x = tf.keras.layers.Input(input_shape=(None,1))
model.add(tf.keras.layers.Dense(64, activation='linear'))
model.add(tf.keras.layers.Dense(1))

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])


# train the model
model.fit(train_x, train_y, epochs=1000, verbose=1)

# test the model
test_x = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20], dtype=np.uint8)
test_y = np.array([12, 13, 14, 15, 16, 17, 18, 19, 20, 21], dtype=np.uint8)

model.evaluate(test_x, test_y, verbose=2)

# save the model
model.save("models/prime0")

