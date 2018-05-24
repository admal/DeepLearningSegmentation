import logging

from keras.models import *
from keras.layers import *
from config import *
from keras.models import Model


class MyModel:
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
        loaded_model = model_from_json(loaded_model_json)
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

    def add_conv(self, model, filter_size=NO_CLASSES):
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(
            filters=filter_size,
            kernel_size=[3, 3],
            padding='same',
            activation='relu'
        ))

    def max_pooling(self):
        return tf.keras.layers.MaxPool2D(
            pool_size=(2, 2),
            padding='same')

    def up_sampling(self, model):
        model.add(tf.keras.layers.UpSampling2D(size=(2, 2)))

    def _model_keras(self):

        kernel = 3
        filter_size = 64
        pad = 1
        pool_size = 2

        model = Sequential()
        model.add(Layer(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH,3)))

        # encoder
        model.add(ZeroPadding2D(padding=(pad, pad)))
        model.add(Convolution2D(filter_size, kernel, kernel, border_mode='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

        model.add(ZeroPadding2D(padding=(pad, pad)))
        model.add(Convolution2D(128, kernel, kernel, border_mode='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

        model.add(ZeroPadding2D(padding=(pad, pad)))
        model.add(Convolution2D(256, kernel, kernel, border_mode='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

        model.add(ZeroPadding2D(padding=(pad, pad)))
        model.add(Convolution2D(512, kernel, kernel, border_mode='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        # decoder
        model.add(ZeroPadding2D(padding=(pad, pad)))
        model.add(Convolution2D(512, kernel, kernel, border_mode='valid'))
        model.add(BatchNormalization())

        model.add(UpSampling2D(size=(pool_size, pool_size)))
        model.add(ZeroPadding2D(padding=(pad, pad)))
        model.add(Convolution2D(256, kernel, kernel, border_mode='valid'))
        model.add(BatchNormalization())

        model.add(UpSampling2D(size=(pool_size, pool_size)))
        model.add(ZeroPadding2D(padding=(pad, pad)))
        model.add(Convolution2D(128, kernel, kernel, border_mode='valid'))
        model.add(BatchNormalization())

        model.add(UpSampling2D(size=(pool_size, pool_size)))
        model.add(ZeroPadding2D(padding=(pad, pad)))
        model.add(Convolution2D(filter_size, kernel, kernel, border_mode='valid'))
        model.add(BatchNormalization())

        model.add(Convolution2D(NO_CLASSES, 1, 1, border_mode='valid', ))

        model.outputHeight = model.output_shape[-2]
        model.outputWidth = model.output_shape[-1]

        # model.add(Reshape((NO_CLASSES, model.output_shape[-2] * model.output_shape[-1]),
        #                   input_shape=(NO_CLASSES, model.output_shape[-2], model.output_shape[-1])))

        # model.add(Permute((2, 1)))
        model.add(Activation('softmax'))

        model.compile(loss="categorical_crossentropy", optimizer=optimizers.adam(LEARNING_RATE), metrics=['accuracy'])

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
