from tensorflow.keras import layers, optimizers, losses, Sequential
from tensorflow import keras

import tensorflow as tf
from data_pre import data_pre
import numpy as np


class bp_model():

    def train_model(self, train_x, train_y):

        # train_y = train_y['0'].values
        train_y = tf.one_hot(train_y, depth=2)

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=data_pre().loadLog() + "people_npz-32-8-2",
                                                              histogram_freq=1)

        network = Sequential([
            layers.Dense(32, activation="sigmoid"),
            # layers.Dense(16, activation="sigmoid"),
            layers.Dense(8, activation="sigmoid"),
            layers.Dense(2)
        ])
        network.build(input_shape=(None, 400))

        network.summary()

        network.compile(optimizer=optimizers.Adam(0.001),
                        loss=losses.CategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath='my_model.h5',
            monitor='val_loss',
            verbose=1,
            save_weights_only=False,
            save_best_only=True,
            mode='min',
        )

        network.fit(x=train_x, y=train_y, epochs=100, validation_split=0.1, validation_freq=1, verbose=1,
                    callbacks=[checkpoint, tensorboard_callback])

        print(network.predict(train_x[:200]))

        network.save("my_model_npz.h5")

    pass


pass

if __name__ == '__main__':
    npz_filepath = r"/home/jry/MicroWave_right_npz/mydataset.npz"
    # x, y = data_pre().loadData()
    x, y =data_pre().load_npz_data(path=npz_filepath)
    bp_model().train_model(train_x=x, train_y=y)
