
import tensorflow as tf
import numpy as np

print(tf.config.list_physical_devices())

with tf.device("/GPU:0"):
    x_test, y_test = np.load("resources/test_x.npy"), np.load("resources/test_y.npy")

    if 1:
        x_train, y_train = np.load("resources/train_x.npy"), np.load("resources/train_y.npy")
        
        x_train = x_train.astype(np.float32)
        x_train/=np.float32(255.0)
        y_train = y_train.astype(np.float32)
        y_train/=np.float32(255.0)

        x_test = x_test.astype(np.float32)
        x_test/=np.float32(255.0)
        y_test = y_test.astype(np.float32)
        y_test/=np.float32(255.0)

        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath="models/checkpointprime.h5",
            verbose=1,
            save_weights_only=False,
            save_best_only=True,
            save_freq="epoch",
            monitor="val_loss",
            mode="min"
        )

        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(4,input_shape=(4,)),
            tf.keras.layers.Dense(8),
            tf.keras.layers.Dense(4,activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(2)
        ])
        model.compile(
            optimizer="adam",
            loss="mse",
            metrics=["accuracy"]
        )

        model.fit(x_train, y_train, batch_size=300, epochs=4, validation_data=(x_test, y_test), callbacks=[cp_callback], verbose=1)
        model.save("models/prime0")

    else:
        model = tf.keras.models.load_model("models/prime0")

    model.evaluate(x_test, y_test, verbose=2)


# Load and prepare the MNIST dataset. The pixel values of the images range from 0 through 255. Scale these values to a 
# range of 0 to 1 by dividing the values by 255.0. This also converts the sample data from integers to floating-point 
# numbers:
#mnist = tf.keras.datasets.mnist
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
#for i in (x_train, y_train, x_test, y_test): print(type(i), i.shape, i.dtype)
#print(x_train[0], y_train[0])
#x_train, x_test = x_train / 255.0, x_test / 255.0
#exit(0)
#if 1:
#    #tf.debugging.set_log_device_placement(True)
#    """
#    # Create some tensors
#    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
#    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
#    c = tf.matmul(a, b)
#
#    print(c)
#    exit(0)
#    """
#
#    # Build a tf.keras.Sequential model:
#    model = tf.keras.models.Sequential([
#        tf.keras.layers.Flatten(input_shape=(28, 28)),
#        tf.keras.layers.EinsumDense("ij,jk->ik", 256, activation='relu'),
#        tf.keras.layers.Dropout(0.2),
#        tf.keras.layers.Dense(10)
#    ])
#
#    #For each example, the model returns a vector of logits or log-odds scores, one for each class.
#    predictions:np.ndarray = model(x_train[:1]).numpy()
#
#    # The tf.nn.softmax function converts these logits to "probabilities" for each class:
#    tf.nn.softmax(predictions).numpy()
#
#    # Define a loss function for training using losses.SparseCategoricalCrossentropy:
#    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#
#    # Before you start training, configure and compile the model using Keras Model.compile. Set the optimizer class to 
#    # adam and the loss function to the loss_fn you defined earlier, and specify a metric to be evaluated for the model by 
#    # setting the metrics parameter to accuracy.
#    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
#
#
#    # Use the Model.fit method to adjust your model parameters and minimize the loss: 
#    model.fit(x_train, y_train, epochs=5)
#
#    model.save('models/test7')
#
#else:
#    # test1:
#    # 313/313 - 1s - loss: 0.0772 - accuracy: 0.9762 - 1s/epoch - 3ms/step
#    # test2:
#    # 313/313 - 1s - loss: 0.1078 - accuracy: 0.9680 - 553ms/epoch - 2ms/step
#    # test3:
#    # 313/313 - 1s - loss: 0.0974 - accuracy: 0.9705 - 597ms/epoch - 2ms/step
#    # test4: (einsum)
#    # 313/313 - 1s - loss: 0.0714 - accuracy: 0.9787 - 550ms/epoch - 2ms/step
#    # test5: (4x 28 dense)
#    # 313/313 - 1s - loss: 0.1430 - accuracy: 0.9592 - 645ms/epoch - 2ms/step
#    # test6: (100, (dropout), 50, 25, 12)
#    # 313/313 - 1s - loss: 0.1180 - accuracy: 0.9707 - 611ms/epoch - 2ms/step
#    # test7: (196 einsum)
#    # 313/313 - 1s - loss: 0.0710 - accuracy: 0.9797 - 537ms/epoch - 2ms/step
#    # test8: (256 einsum)
#    # 313/313 - 1s - loss: 0.0715 - accuracy: 0.9802 - 545ms/epoch - 2ms/step
#    model = tf.keras.models.load_model('models/test')
#
## Evaluate the model using the test data:
#print(model.evaluate(x_test,  y_test, verbose=2))
#
##print(predictions)
