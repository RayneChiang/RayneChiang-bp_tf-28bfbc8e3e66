import tensorflow as tf
import tensorflow

import numpy as np

from tensorflow.examples.saved_model.integration_tests import mnist_util

import os


class bp_tf:

    # def __init__(inputnodes, ):

    # downLoad Mnist data as test
    def loadData(self):
        # module_path = os.path.dirname(__file__)

        # test_image = module_path + "t10k-images-idx3-ubyte"
        (x_train, y_train), (x_test, y_test) = mnist_util.load_reshaped_data(use_fashion_mnist=False,
                                                                             fake_tiny_data=False)
        x_train = np.reshape(x_train, (60000, 784))
        x_test = np.reshape(x_test, (10000, 784))

        return (x_train, y_train), (x_test, y_test)

    def __init__(self, input_Nodes, hidden_Nodes, output_Nodes, lamda):
        (train_image, train_label), (test_image, test_label) = self.loadData()

        self.input_Nodes = input_Nodes
        self.hidden_Nodes = hidden_Nodes
        self.output_Nodes = output_Nodes
        self.lamda = lamda

        # random normalization
        self.input_Weight = tf.Variable(tf.random.normal(
            [input_Nodes, hidden_Nodes], mean=0, stddev=1, dtype=tf.float32, seed=5))
        self.hidden_Weight = tf.Variable(tf.random.normal(
            [hidden_Nodes, output_Nodes], mean=0, stddev=1, dtype=tf.float32, seed=5))
        self.input_Threshold = tf.Variable(tf.random.normal(
            [1, self.hidden_Nodes], mean=0, stddev=1, dtype=tf.float32, seed=5))
        self.hidden_Threshold = tf.Variable(tf.random.normal(
            [1, self.output_Nodes], mean=0, stddev=1, dtype=tf.float32, seed=5))

        # L1 regularization, the bigger the lamda, the more impact bias has on output
        tf.compat.v1.add_to_collection('loss', tf.keras.regularizers.l1(self.lamda * self.input_Weight))
        tf.compat.v1.add_to_collection('loss', tf.keras.regularizers.l1(self.lamda * self.hidden_Weight))
        tf.compat.v1.add_to_collection('loss', tf.keras.regularizers.l1(self.lamda * self.input_Threshold))
        tf.compat.v1.add_to_collection('loss', tf.keras.regularizers.l1(self.lamda * self.hidden_Threshold))

        # forward propagation which uses sigmoid function as activation function
        # matrix-multi add Threshold as pre-activation // Threshold?

        # self.hidden_Cells = tf.math.sigmoid(tf.matmul(train_image, self.input_Weight)+self.input_Threshold)
        # self.output_Cells = tf.math.sigmoid(tf.matmul(self.hidden_Cells, self.hidden_Weight)+self.hidden_Threshold)

    def trainModel(self):
        (train_image, train_label), (test_image, test_label) = self.loadData()
        (train_image, train_label), (test_image, test_label)


def main():
    bp_tf(60000, 100, 10, 0.01).trainModel()
    pass


if __name__ == '__main__':
    main()
