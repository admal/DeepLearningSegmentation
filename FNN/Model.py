import logging
import os

import tensorflow as tf

from config import *


class Model:
    _model_graph = None
    _model = None
    model_dir = r"C:\Users\wpiot\PycharmProjects\DeepLearningSegmentation"

    def __init__(self, model_dir):
        """
        Initializes model
        :param model_dir: directory with model, it can be trained and used to predict rotation
        or with checkpoints to continue training
        """
        self.model_dir = model_dir

    @staticmethod
    def load_model():
        if not os.path.isfile('model.json'):
            return None

        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = tf.keras.models.model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("model.h5")
        print("Loaded model from disk")
        print(loaded_model.summary())
        return loaded_model

    def get_model(self):
        if self._model_graph is None:
            self._model_graph = self._build_model()
            self._model = self._model_graph

        return self._model

    def add_conv(self, model):
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(
            filters=NO_CLASSES,
            kernel_size=[3, 3],
            padding='same',
            activation='relu'
        ))


    def max_pooling(self):
        return tf.keras.layers.MaxPool2D(
            pool_size=(2, 2),
            padding='same')
    def up_sampling(self,model):
        model.add(tf.keras.layers.UpSampling2D(size=(2, 2)))
    def _model_keras(self):
        model = tf.keras.models.Sequential()
        #first
        model.add(tf.keras.layers.Conv2D(
            input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
            filters=NO_CLASSES,
            kernel_size=[3, 3],
            padding='same',
            activation='relu'
        ))
        model.add(tf.keras.layers.BatchNormalization())
        self.add_conv(model)
        model.add(self.max_pooling())

        # second
        self.add_conv(model)
        self.add_conv(model)
        model.add(self.max_pooling())
        # third
        self.add_conv(model)
        self.add_conv(model)
        self.add_conv(model)
        model.add(self.max_pooling())
        # fourth
        self.add_conv(model)
        self.add_conv(model)
        self.add_conv(model)
        model.add(self.max_pooling())

        # fifth
        self.add_conv(model)
        self.add_conv(model)
        self.add_conv(model)
        model.add(self.max_pooling())


        model.add(tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=(2, 2),
            padding='same',
            activation='relu'
        ))

        self.up_sampling(model)
        self.add_conv(model)
        self.add_conv(model)
        self.add_conv(model)

        self.up_sampling(model)
        self.add_conv(model)
        self.add_conv(model)
        self.add_conv(model)

        self.up_sampling(model)
        self.add_conv(model)
        self.add_conv(model)
        self.add_conv(model)

        self.up_sampling(model)
        self.add_conv(model)
        self.add_conv(model)

        self.up_sampling(model)
        self.add_conv(model)
        self.add_conv(model)

        model.add(tf.keras.layers.Dense(NO_CLASSES, activation='softmax'))
        model.compile(
            loss=tf.keras.losses.categorical_crossentropy,
            optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE),
            metrics=['accuracy']
        )
        return model

    def _model_fn(self, features, labels, mode):
        return
        input_layer = tf.reshape(features["x"], [-1, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS])

        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[3, 3],
            padding='same',
            activation=tf.nn.relu)

        max_pool_1 = tf.layers.max_pooling2d(
            inputs=conv1,
            pool_size=[2, 2],
            strides=2,
            padding='same'
        )

        conv2 = tf.layers.conv2d(
            inputs=max_pool_1,
            filters=32,
            kernel_size=[3, 3],
            padding='same',
            activation=tf.nn.relu)

        max_pool_2 = tf.layers.max_pooling2d(
            inputs=conv2,
            pool_size=[2, 2],
            strides=2,
            padding='same'
        )

        # fully connected
        conv3 = tf.layers.conv2d(
            inputs=max_pool_2,
            filters=128,
            kernel_size=[2, 2],
            padding='same',
            activation=tf.nn.relu
        )
        con4 = tf.layers.conv2d(
            inputs=conv3,
            filters=NO_CLASSES,
            kernel_size=[1, 1],
            padding='sane',
            activation=tf.nn.relu
        )

    def _build_model(self):
        return self._model_keras()

        network = input_data(shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
        # VGG Model
        logging.info("Input shape: {}".format(network.shape))
        network = conv_2d(network, 32, 3, padding="same", activation="relu")
        network = max_pool_2d(network, 2, padding='same')
        network = conv_2d(network, 32, 3, padding="same", activation="relu")
        network = max_pool_2d(network, 2, padding='same')

        # fully connected
        network = conv_2d(network, 128, 2, padding='same', activation='relu')
        logging.info("Conv shape: {}".format(network.shape))
        network = conv_2d(network, NO_CLASSES, 1)
        logging.info("Last conv shape: {}".format(network.shape))
        network = upsample_2d(network, 4)  # for now it is hardcoded, I will find better solution later ~AM
        network = regression(
            network,
            learning_rate=LEARNING_RATE,
            loss='categorical_crossentropy',
            name='targets',
            metric='accuracy'
        )

        return network

    def load(self):
        self.get_model().load(self.model_dir)
