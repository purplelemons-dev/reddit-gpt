
import tensorflow as tf
import numpy as np


def to_base256(x:int) -> np.ndarray[np.float32]:
    result = [0] * 4
    for i in range(3, -1, -1):
        result[i] = x % 256
        x //= 256
    return np.array(result, dtype=np.float32)

with tf.device("/GPU:0"):
    model:tf.keras.models.Sequential = tf.keras.models.load_model("models/checkpointprime.h5")
    model.summary()
    while True:
        try:
            x = int(input("Enter a number: "))
            x = to_base256(x) / np.float32(255.0)
            x = x.reshape((1, 4))
            print("survey says! ", model(x,training=False))
        except KeyboardInterrupt:
            break
    
