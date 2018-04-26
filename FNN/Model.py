import tensorflow as tf
from config import *
import logging
import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, upsample_2d
from tflearn.layers.estimator import regression


class Model:
	_model_graph = None
	_model = None
	model_dir = "C:\\Users\\ASUS\\Documents\\PW\\SieciNeuronowe\\Projekt2\\Model"

	def __init__(self, model_dir):
		"""
        Initializes model
        :param model_dir: directory with model, it can be trained and used to predict rotation
        or with checkpoints to continue training
        """
		self.model_dir = model_dir

	def get_model(self):
		if self._model_graph is None:
			self._model_graph = self._build_model()
			self._model = tflearn.DNN(self._model_graph,
			                          tensorboard_dir=MODEL_DIRECTORY,
			                          tensorboard_verbose=3)

		return self._model

	def _build_model(self):
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
