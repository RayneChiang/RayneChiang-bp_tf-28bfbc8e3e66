import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"


class pb_to_tflite:

    def __init__(self):
        self.module_path = os.path.dirname(__file__)

    def pb_converter(self, model_dir):
        my_model = tf.keras.models.load_model(model_dir)

        my_model.summary()

        converter = tf.lite.TFLiteConverter.from_keras_model(my_model)

        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]

        tf_model = converter.convert()

        open("converted_model.tflite", "wb").write(tf_model)


    pass


if __name__ == '__main__':
    pb_to_tflite().pb_converter('/home/jry/PycharmProjects/bp_tf/my_model')
